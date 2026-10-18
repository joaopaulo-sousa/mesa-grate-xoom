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
 * result in a register of its own.
 */
#include <math.h>
#include <string.h>

#include "fp_compiler.h"

#include "util/macros.h"

static float
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
      if (info->imm_mask & (1u << o->index)) {
         p->kind = GRATE_FP_PRE_CONST;
         p->value = fp20_to_float(info->consts[o->index]);
      } else {
         p->kind = GRATE_FP_PRE_USER;
         p->index = o->index;
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
   p->op = u->slots[0].op;
   p->saturate = u->saturate;
   p->num_products = u->num_slots;

   for (int k = 0; k < u->num_slots; k++) {
      if (!fp_pre_opnd(c, &u->slots[k].src[0], pre_of, &p->a[k]) ||
          !fp_pre_opnd(c, &u->slots[k].src[1], pre_of, &p->b[k]))
         return false;
   }

   /* a DP4 sums the fourth product through slot 3 instead of rC */
   const struct fp_unit_slot *cs = &u->slots[u->num_slots == 4 ? 3 : 0];
   return fp_pre_opnd(c, &cs->src[2], pre_of, &p->c);
}

static int
fp_pre_slot(struct grate_fp_info *info)
{
   for (int i = GRATE_FP_NUM_CONSTS - 1; i >= 0; i--)
      if (!(info->used_mask & (1u << i)))
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

   /* values that must be in a register: texture coordinates and the
    * operands of special functions, which work in place */
   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < 2; k++)
         if (c->tex[n].coord[k].kind == FP_OPND_VALUE)
            keep[c->tex[n].coord[k].index] = true;
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->sfu)
         continue;
      fp_unit_foreach_src(u, k, j)
         if (u->slots[k].src[j].kind == FP_OPND_VALUE)
            keep[u->slots[k].src[j].index] = true;
   }

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      struct grate_fp_pre p;

      if (!fp_unit_has_value(u) || u->sfu || u->sfu_copy || keep[i] ||
          (u->num_slots > 1 && u->slots[0].op != FP_ALU_OP_MAD) ||
          info->num_pre == GRATE_FP_MAX_PRE || !fp_pre_unit(c, u, pre_of, &p))
         continue;

      int slot = fp_pre_slot(info);
      if (slot < 0)
         break;
      p.dst = slot;
      info->used_mask |= 1u << slot;
      pre_of[i] = info->num_pre;
      slot_of[i] = slot;
      info->pre[info->num_pre++] = p;
   }

   /* the readers take the uniform; the units themselves end up dead */
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE && slot_of[o->index] >= 0) {
            o->kind = FP_OPND_UNIFORM;
            o->index = slot_of[o->index];
         }
      }
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

void
grate_fp_eval_pre(const struct grate_fp_info *info, const float *user,
                  unsigned user_len, uint32_t *consts)
{
   float results[GRATE_FP_MAX_PRE];

   for (int i = 0; i < info->num_pre; i++) {
      const struct grate_fp_pre *p = &info->pre[i];
      float ab = 0.0f, r;

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
      consts[p->dst] = grate_float_to_fp20(r);
   }
}
