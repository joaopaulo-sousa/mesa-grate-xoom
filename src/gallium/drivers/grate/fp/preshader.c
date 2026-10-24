/*
 * Preshader: units whose operands are all uniforms (or constants, or other
 * such units) give the same value for every pixel of a draw. They are taken
 * out of the program and evaluated on the CPU when the uniforms are uploaded
 * (grate_fp_eval_pre), into a free uniform slot their readers then use. This
 * is what the D3D9 effect compiler called a preshader, and what NIR's
 * nir_opt_preamble does for GPUs that run the preamble themselves.
 *
 * It saves ALU slots and, above all, registers: e.g. a shader selecting
 * texel channels by comparing uniforms otherwise keeps every comparison
 * result in a register of its own. Special functions of uniforms count too:
 * they would otherwise force a sequential program.
 *
 * The results a program branches on (comparisons, CMP conditions) can also
 * specialize it (fp_specialize): see grate_fp_spec.
 */
#include <math.h>
#include <string.h>

#include "fp_compiler.h"

#include "util/macros.h"

float
fp20_to_float(uint32_t v)
{
   union { uint32_t u; float f; } r;
   uint32_t exponent = (v >> 13) & 0x3f;

   if (!(v & 0x7ffff))
      return 0.0f;
   r.u = ((v >> 19) & 1) << 31 |
         (exponent == 0x3f ? 0xff : exponent - 31 + 127) << 23 |
         (v & 0x1fff) << 10;
   return r.f;
}

static bool
fp_pre_opnd(struct fp_compiler *c, const struct fp_opnd *o, const int *pre_of,
            struct grate_fp_pre_opnd *p)
{
   const struct grate_fp_info *info = &c->fp->info;

   *p = (struct grate_fp_pre_opnd) {
      .negate = o->negate,
      .absolute = o->absolute,
   };

   switch (o->kind) {
   case FP_OPND_ZERO:
   case FP_OPND_ONE:
      p->kind = GRATE_FP_PRE_CONST;
      p->value = o->kind == FP_OPND_ONE;
      return true;
   case FP_OPND_UNIFORM:
      if (BITSET_TEST(info->imm_mask, o->index)) {
         p->kind = GRATE_FP_PRE_CONST;
         p->value = fp20_to_float(info->consts[o->index]);
      } else {
         p->kind = GRATE_FP_PRE_USER;
         p->index = info->user_index[o->index];
      }
      return true;
   case FP_OPND_VALUE:
      if (pre_of[o->index] < 0)
         return false;
      p->kind = GRATE_FP_PRE_RESULT;
      p->index = pre_of[o->index];
      return true;
   default:
      return false;
   }
}

/* The preshader entry computing unit u, false if u depends on a pixel */
static bool
fp_pre_unit(struct fp_compiler *c, const struct fp_unit *u, const int *pre_of,
            struct grate_fp_pre *p)
{
   memset(p, 0, sizeof(*p));
   if (u->sfu) {
      p->sfu = u->sfu;
      return fp_pre_opnd(c, &u->slots[0].src[0], pre_of, &p->a[0]);
   }
   p->op = u->slots[0].op;
   p->saturate = u->saturate;
   p->num_products = u->num_slots;
   for (int k = 0; k < u->num_slots; k++)
      if (u->slots[k].square_c)
         return false;   /* the CPU evaluator has no squared term */

   for (int k = 0; k < u->num_slots; k++) {
      if (!fp_pre_opnd(c, &u->slots[k].src[0], pre_of, &p->a[k]) ||
          !fp_pre_opnd(c, &u->slots[k].src[1], pre_of, &p->b[k]))
         return false;
   }

   /* a DP4 sums the fourth product through slot 3 instead of rC */
   const struct fp_unit_slot *cs = &u->slots[u->num_slots == 4 ? 3 : 0];
   return fp_pre_opnd(c, &cs->src[2], pre_of, &p->c);
}

/* Whether a preshader operand is always 0.0 or 1.0 */
static bool
fp_pre_opnd_01(const struct grate_fp_info *info, const struct grate_fp_pre_opnd *o)
{
   if (o->negate)
      return false;
   if (o->kind == GRATE_FP_PRE_CONST)
      return o->value == 0.0f || o->value == 1.0f;
   return o->kind == GRATE_FP_PRE_RESULT && info->pre[o->index].value01;
}

/* Whether a preshader entry always gives 0.0 or 1.0: comparisons (a CMP
 * between 0 and 1, see emit_setcc) and their products, min and max */
static bool
fp_pre_value01(const struct grate_fp_info *info, const struct grate_fp_pre *p)
{
   if (p->sfu || p->num_products > 1)
      return false;
   switch (p->op) {
   case FP_ALU_OP_CSEL:
      return fp_pre_opnd_01(info, &p->b[0]) && fp_pre_opnd_01(info, &p->c);
   case FP_ALU_OP_MAD:
      return fp_pre_opnd_01(info, &p->a[0]) && fp_pre_opnd_01(info, &p->b[0]) &&
             p->c.kind == GRATE_FP_PRE_CONST && p->c.value == 0.0f;
   default:
      return fp_pre_opnd_01(info, &p->a[0]) && fp_pre_opnd_01(info, &p->b[0]) &&
             fp_pre_opnd_01(info, &p->c);
   }
}

static int
fp_pre_slot(struct grate_fp_info *info)
{
   for (int i = 0; i < GRATE_FP_MAX_SLOTS; i++)
      if (!BITSET_TEST(info->used_mask, i))
         return i;
   return -1;
}

void
fp_preshade(struct fp_compiler *c)
{
   struct grate_fp_info *info = &c->fp->info;
   int pre_of[FP_MAX_UNITS], slot_of[FP_MAX_UNITS];
   bool keep[FP_MAX_UNITS] = { false };

   memset(pre_of, -1, sizeof(pre_of));
   memset(slot_of, -1, sizeof(slot_of));

   /* values that must be in a register: texture coordinates */
   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < 2; k++)
         if (c->tex[n].coord[k].kind == FP_OPND_VALUE)
            keep[c->tex[n].coord[k].index] = true;

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      struct grate_fp_pre p;

      if (!fp_unit_has_value(u) || keep[i] ||
          (u->num_slots > 1 && u->slots[0].op != FP_ALU_OP_MAD) ||
          info->num_pre == GRATE_FP_MAX_PRE || !fp_pre_unit(c, u, pre_of, &p))
         continue;

      int slot = fp_pre_slot(info);
      if (slot < 0)
         break;
      p.dst = slot;
      p.value01 = fp_pre_value01(info, &p);
      BITSET_SET(info->used_mask, slot);
      pre_of[i] = info->num_pre;
      slot_of[i] = slot;
      info->pre[info->num_pre++] = p;
   }

   /* the readers take the uniform; the units themselves end up dead. A
    * special function works in place on a register: one left in the
    * program (out of preshader room) keeps its operand unit */
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (u->sfu)
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE && slot_of[o->index] >= 0) {
            o->kind = FP_OPND_UNIFORM;
            o->index = slot_of[o->index];
         }
      }
   }

   /* what the program left branches on */
   int pre_at_slot[GRATE_FP_MAX_SLOTS];
   memset(pre_at_slot, -1, sizeof(pre_at_slot));
   for (int i = 0; i < info->num_pre; i++)
      pre_at_slot[info->pre[i].dst] = i;

   info->num_spec = 0;
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (slot_of[i] >= 0 || u->sfu)
         continue;
      fp_unit_foreach_src(u, k, j) {
         const struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind != FP_OPND_UNIFORM || pre_at_slot[o->index] < 0)
            continue;
         struct grate_fp_spec sp = { .pre = pre_at_slot[o->index] };
         if (info->pre[sp.pre].value01)
            sp.value01 = true;
         else if (u->num_slots == 1 && u->slots[0].op == FP_ALU_OP_CSEL &&
                  k == 0 && j == 0) {
            sp.negate = o->negate;
            sp.absolute = o->absolute;
         } else
            continue;

         bool known = false;
         for (int n = 0; n < info->num_spec; n++)
            known |= !memcmp(&info->spec[n], &sp, sizeof(sp));
         if (!known && info->num_spec < GRATE_FP_MAX_SPEC)
            info->spec[info->num_spec++] = sp;
      }
   }
}

/* The specialized value of operand o, a preshader result, if any */
static const struct grate_fp_spec *
fp_spec_find(struct fp_compiler *c, const int *pre_at_slot,
             const struct fp_opnd *o, bool value01, int *bit)
{
   const struct grate_fp_info *info = &c->fp->info;

   if (o->kind != FP_OPND_UNIFORM || pre_at_slot[o->index] < 0)
      return NULL;
   for (int n = 0; n < info->num_spec; n++) {
      const struct grate_fp_spec *sp = &info->spec[n];
      if (sp->pre != pre_at_slot[o->index] || sp->value01 != value01 ||
          (!value01 && (sp->negate != o->negate || sp->absolute != o->absolute)))
         continue;
      *bit = (c->key->spec_bits >> n) & 1;
      return sp;
   }
   return NULL;
}

/* Value of operand o, when constant in this variant */
static bool
fp_spec_const(struct fp_compiler *c, const int *pre_at_slot, const bool *known,
              const float *kval, const struct fp_opnd *o, float *v)
{
   const struct grate_fp_info *info = &c->fp->info;
   int bit;

   switch (o->kind) {
   case FP_OPND_ZERO:
      *v = 0.0f;
      break;
   case FP_OPND_ONE:
      *v = 1.0f;
      break;
   case FP_OPND_UNIFORM:
      if (BITSET_TEST(info->imm_mask, o->index))
         *v = fp20_to_float(info->consts[o->index]);
      else if (fp_spec_find(c, pre_at_slot, o, true, &bit))
         *v = bit;
      else
         return false;
      break;
   case FP_OPND_VALUE:
      if (!known[o->index])
         return false;
      *v = kval[o->index];
      break;
   default:
      return false;
   }
   if (o->absolute)
      *v = fabsf(*v);
   if (o->negate)
      *v = -*v;
   return true;
}

/* Operand x read through operand r (a reader's modifiers on a copy of x):
 * absolute value first, then negation, as TGSI */
static struct fp_opnd
fp_spec_through(struct fp_opnd r, struct fp_opnd x)
{
   if (r.absolute) {
      x.absolute = true;
      x.negate = r.negate;
   } else
      x.negate ^= r.negate;
   return x;
}

static bool
fp_spec_has_slot(struct fp_compiler *c, float v)
{
   const struct grate_fp_info *info = &c->fp->info;
   uint32_t fp20 = grate_float_to_fp20(v);

   for (int i = 0; i < GRATE_FP_MAX_SLOTS; i++)
      if (!BITSET_TEST(info->used_mask, i) ||
          (BITSET_TEST(info->imm_mask, i) && info->consts[i] == fp20))
         return true;
   return false;
}

/*
 * Specialize the program for c->key->spec_bits: the specialized preshader
 * results become constants, which fold through the units in order (CMP on
 * a known condition is a copy, x * 0 drops x, x * 1 is x). Units folded to
 * a copy hand their readers the operand copied; fp_dead_code then drops
 * them and whatever only they read.
 */
void
fp_specialize(struct fp_compiler *c)
{
   int pre_at_slot[GRATE_FP_MAX_SLOTS];
   struct fp_opnd repl[FP_MAX_UNITS];
   bool has_repl[FP_MAX_UNITS] = { false }, known[FP_MAX_UNITS] = { false };
   float kval[FP_MAX_UNITS];

   if (!c->key || !c->key->spec || !c->fp->info.num_spec)
      return;

   memset(pre_at_slot, -1, sizeof(pre_at_slot));
   for (int i = 0; i < c->fp->info.num_pre; i++)
      pre_at_slot[c->fp->info.pre[i].dst] = i;

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];

      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind != FP_OPND_VALUE || !has_repl[o->index])
            continue;
         struct fp_opnd x = fp_spec_through(*o, repl[o->index]);
         /* special functions work in place on a value */
         if (u->sfu && (x.kind != FP_OPND_VALUE || x.negate || x.absolute))
            continue;
         *o = x;
      }

      if ((!fp_unit_has_value(u) && u->output_chan < 0) || u->sfu ||
          u->sfu_copy || u->num_slots != 1 || u->slots[0].accumulate_this ||
          u->slots[0].accumulate_other)
         continue;

      struct fp_unit_slot *s = &u->slots[0];
      struct fp_opnd x;
      bool have_x = false, have_v = false;
      float a, b, cc, v = 0.0f;
      bool ka = fp_spec_const(c, pre_at_slot, known, kval, &s->src[0], &a);
      bool kb = fp_spec_const(c, pre_at_slot, known, kval, &s->src[1], &b);
      bool kc = fp_spec_const(c, pre_at_slot, known, kval, &s->src[2], &cc);
      int bit;

      switch (s->op) {
      case FP_ALU_OP_CSEL:
         if (ka || fp_spec_find(c, pre_at_slot, &s->src[0], false, &bit)) {
            x = s->src[(ka ? a < 0.0f : bit) ? 1 : 2];
            have_x = true;
         }
         break;
      case FP_ALU_OP_MAD:
         if ((ka && a == 0.0f) || (kb && b == 0.0f)) {
            x = s->src[2];
            have_x = true;
         } else if (ka && kb && kc) {
            v = a * b + cc;
            have_v = true;
         } else if (kc && cc == 0.0f && ka && a == 1.0f) {
            x = s->src[1];
            have_x = true;
         } else if (kc && cc == 0.0f && kb && b == 1.0f) {
            x = s->src[0];
            have_x = true;
         }
         break;
      case FP_ALU_OP_MIN:
      case FP_ALU_OP_MAX:
         if (ka && kb && kc) {
            v = s->op == FP_ALU_OP_MIN ? fminf(a * b, cc) : fmaxf(a * b, cc);
            have_v = true;
         }
         break;
      }
      if (have_x)
         have_v = fp_spec_const(c, pre_at_slot, known, kval, &x, &v);
      if (!have_x && !have_v)
         continue;

      bool output = u->output_chan >= 0;
      if (have_v) {
         if (u->saturate)
            v = CLAMP(v, 0.0f, 1.0f);
         if (!output) {
            known[i] = true;
            kval[i] = v;
         }
         if (v == 0.0f)
            x = fp_opnd_zero(c, output);
         else if (v == 1.0f)
            x = fp_opnd_one(c, output);
         else if (fp_spec_has_slot(c, v))
            x = fp_opnd_const(c, v);
         else
            continue;   /* the unit stays, its readers fold all the same */
      }

      if (!output && !u->saturate) {
         repl[i] = x;
         has_repl[i] = true;
      } else {
         fp_unit_op(s, FP_ALU_OP_MAD, x, fp_opnd_one(c, output),
                    fp_opnd_zero(c, output));
      }
   }

   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < 2; k++) {
         struct fp_opnd *o = &c->tex[n].coord[k];
         if (o->kind != FP_OPND_VALUE || !has_repl[o->index])
            continue;
         struct fp_opnd x = fp_spec_through(*o, repl[o->index]);
         if (!x.negate && !x.absolute &&
             (x.kind == FP_OPND_VALUE || x.kind == FP_OPND_INPUT))
            *o = x;
      }
}

static float
fp_pre_read(const struct grate_fp_pre_opnd *o, const float *user,
            unsigned user_len, const float *results)
{
   float v;

   switch (o->kind) {
   case GRATE_FP_PRE_USER:
      v = o->index < user_len ? user[o->index] : 0.0f;
      break;
   case GRATE_FP_PRE_RESULT:
      v = results[o->index];
      break;
   default:
      v = o->value;
      break;
   }
   if (o->absolute)
      v = fabsf(v);
   return o->negate ? -v : v;
}

static float
fp_pre_sfu(enum fp_sfu_op op, float x)
{
   switch (op) {
   case FP_SFU_OP_RCP:  return 1.0f / x;
   case FP_SFU_OP_RSQ:  return 1.0f / sqrtf(x);
   case FP_SFU_OP_LG2:  return log2f(x);
   case FP_SFU_OP_EX2:  return exp2f(x);
   case FP_SFU_OP_SQRT: return sqrtf(x);
   case FP_SFU_OP_SIN:  return sinf(x);
   case FP_SFU_OP_COS:  return cosf(x);
   case FP_SFU_OP_FRC:  return x - floorf(x);
   default:             return x;   /* the first steps of ex2/sin/cos */
   }
}

static void
fp_pre_results(const struct grate_fp_info *info, const float *user,
               unsigned user_len, float *results)
{
   for (int i = 0; i < info->num_pre; i++) {
      const struct grate_fp_pre *p = &info->pre[i];
      float ab = 0.0f, r;

      if (p->sfu) {
         results[i] = fp_pre_sfu(p->sfu, fp_pre_read(&p->a[0], user, user_len,
                                                     results));
         continue;
      }

      for (int k = 0; k < p->num_products; k++)
         ab += fp_pre_read(&p->a[k], user, user_len, results) *
               fp_pre_read(&p->b[k], user, user_len, results);
      float cc = fp_pre_read(&p->c, user, user_len, results);

      switch (p->op) {
      case FP_ALU_OP_MIN:
         r = fminf(ab, cc);
         break;
      case FP_ALU_OP_MAX:
         r = fmaxf(ab, cc);
         break;
      case FP_ALU_OP_CSEL:
         /* rA < 0 ? rB : rC */
         r = fp_pre_read(&p->a[0], user, user_len, results) < 0.0f ?
             fp_pre_read(&p->b[0], user, user_len, results) : cc;
         break;
      default:
         r = ab + cc;
         break;
      }
      if (p->saturate)
         r = CLAMP(r, 0.0f, 1.0f);
      results[i] = r;
   }
}

void
grate_fp_eval_pre(const struct grate_fp_info *info, const float *user,
                  unsigned user_len, uint32_t *consts)
{
   float results[GRATE_FP_MAX_PRE];

   fp_pre_results(info, user, user_len, results);
   for (int i = 0; i < info->num_pre; i++)
      if (info->pre[i].dst != GRATE_FP_PRE_NO_DST)
         consts[info->pre[i].dst] = grate_float_to_fp20(results[i]);
}

uint32_t
grate_fp_eval_spec(const struct grate_fp_info *info, const float *user,
                   unsigned user_len)
{
   float results[GRATE_FP_MAX_PRE];
   uint32_t bits = 0;

   fp_pre_results(info, user, user_len, results);
   for (int n = 0; n < info->num_spec; n++) {
      const struct grate_fp_spec *sp = &info->spec[n];
      float v = results[sp->pre];
      bool bit;

      if (sp->value01) {
         bit = v != 0.0f;
      } else {
         if (sp->absolute)
            v = fabsf(v);
         if (sp->negate)
            v = -v;
         bit = v < 0.0f;
      }
      bits |= (uint32_t)bit << n;
   }
   return bits;
}
