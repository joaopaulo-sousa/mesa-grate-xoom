/*
 * fixed10 temporaries: a register holds one fp20 value or two fixed10 ones
 * (10 bits, range [-2, 2), steps of 1/256), as the texels, the destination
 * colour and the colour output already are. The NVIDIA compiler keeps lowp
 * values that way; TGSI has lost the precision qualifiers, so the values that
 * may go to a half are found here instead:
 *
 * - their range, by interval arithmetic from texels, the destination colour,
 *   constants, comparisons and saturation, lies within the fixed10 one;
 * - the error of the store (1/512) reaches the colour output at most as is:
 *   every path to a colour write multiplies it by factors within [-1, 1]
 *   only, and none goes through a texture coordinate, a special function,
 *   a CSEL condition or a discard.
 *
 * - at most FX10_MAX_CHAIN of them lie on any path to the colour: each store
 *   rounds by up to half a step, as the colour output does once more.
 *
 * Only sequential programs (4 registers) use them, and only when the program
 * does not fit otherwise (fp_sched_seq), so programs that did fit before
 * come out the same.
 */
#include <math.h>
#include <stdio.h>

#include "grate_common.h"
#include "fp_compiler.h"

#include "util/macros.h"

#define FX10_MIN -2.0f
#define FX10_MAX (2.0f - 1.0f / 256.0f)
#define FX10_MAX_CHAIN 2

struct fx10_range {
   float lo, hi;
};

static const struct fx10_range fx10_unknown = { -INFINITY, INFINITY };

static struct fx10_range
fx10_hull(struct fx10_range a, struct fx10_range b)
{
   return (struct fx10_range) { MIN2(a.lo, b.lo), MAX2(a.hi, b.hi) };
}

static float
fx10_mul_bound(float a, float b)
{
   /* 0 * inf is no product here: a zero bound stays zero */
   return a == 0.0f || b == 0.0f ? 0.0f : a * b;
}

static struct fx10_range
fx10_mul(struct fx10_range a, struct fx10_range b)
{
   float p[4] = {
      fx10_mul_bound(a.lo, b.lo), fx10_mul_bound(a.lo, b.hi),
      fx10_mul_bound(a.hi, b.lo), fx10_mul_bound(a.hi, b.hi),
   };
   struct fx10_range r = { p[0], p[0] };
   for (int i = 1; i < 4; i++) {
      r.lo = MIN2(r.lo, p[i]);
      r.hi = MAX2(r.hi, p[i]);
   }
   return r;
}

static struct fx10_range
fx10_add(struct fx10_range a, struct fx10_range b)
{
   return (struct fx10_range) { a.lo + b.lo, a.hi + b.hi };
}

static bool
fx10_within(struct fx10_range r, float lo, float hi)
{
   return r.lo >= lo && r.hi <= hi;
}

/* Range of a uniform slot: immediates are known, preshader comparisons and
 * saturated preshader results are 0..1 */
static struct fx10_range
fx10_uniform_range(struct fp_compiler *c, int slot)
{
   const struct grate_fp_info *info = &c->fp->info;

   if (BITSET_TEST(info->imm_mask, slot)) {
      float v = fp20_to_float(info->consts[slot]);
      return (struct fx10_range) { v, v };
   }
   for (int i = 0; i < info->num_pre; i++)
      if (info->pre[i].dst == slot && (info->pre[i].value01 || info->pre[i].saturate))
         return (struct fx10_range) { 0.0f, 1.0f };
   return fx10_unknown;
}

static struct fx10_range
fx10_opnd_range(struct fp_compiler *c, const struct fx10_range *value,
                const struct fp_opnd *o)
{
   struct fx10_range r;

   switch (o->kind) {
   case FP_OPND_UNIFORM:
      r = fx10_uniform_range(c, o->index);
      break;
   case FP_OPND_VALUE:
      r = value[o->index];
      break;
   case FP_OPND_TEXEL:
   case FP_OPND_DST:
      /* the textures and render targets are UNORM formats
       * (grate_screen_is_format_supported) */
      r = (struct fx10_range) { 0.0f, 1.0f };
      break;
   case FP_OPND_ZERO:
      r = (struct fx10_range) { 0.0f, 0.0f };
      break;
   case FP_OPND_ONE:
      r = (struct fx10_range) { 1.0f, 1.0f };
      break;
   default:
      return fx10_unknown;
   }
   if (o->minus_one)
      r = (struct fx10_range) { r.lo - 1.0f, r.hi - 1.0f };
   if (o->twice)
      r = (struct fx10_range) { 2.0f * r.lo, 2.0f * r.hi };
   if (o->absolute)
      r = (struct fx10_range) { r.lo >= 0.0f ? r.lo : r.hi <= 0.0f ? -r.hi : 0.0f,
                                MAX2(fabsf(r.lo), fabsf(r.hi)) };
   if (o->negate)
      r = (struct fx10_range) { -r.hi, -r.lo };
   return r;
}

/* Range of the value of an ALU unit: op(sum of the slot products, c) */
static struct fx10_range
fx10_unit_range(struct fp_compiler *c, const struct fx10_range *value,
                const struct fp_unit *u)
{
   const struct fp_unit_slot *s = &u->slots[0];
   struct fx10_range a = fx10_opnd_range(c, value, &s->src[0]);
   struct fx10_range b = fx10_opnd_range(c, value, &s->src[1]);
   struct fx10_range cc = fx10_opnd_range(c, value, &s->src[2]);
   struct fx10_range r;

   if (u->sfu || u->num_slots == 0)
      return fx10_unknown;

   switch (s->op) {
   case FP_ALU_OP_MAD:
      r = fx10_add(fx10_mul(a, b), s->square_c ? fx10_mul(cc, cc) : cc);
      for (int k = 1; k < u->num_slots; k++) {
         const struct fp_unit_slot *sk = &u->slots[k];
         r = fx10_add(r, fx10_mul(fx10_opnd_range(c, value, &sk->src[0]),
                                  fx10_opnd_range(c, value, &sk->src[1])));
         if (sk->square_c) {
            struct fx10_range d = fx10_opnd_range(c, value, &sk->src[2]);
            r = fx10_add(r, fx10_mul(d, d));
         }
      }
      break;
   case FP_ALU_OP_MIN: {
      struct fx10_range p = fx10_mul(a, b);
      r = (struct fx10_range) { MIN2(p.lo, cc.lo), MIN2(p.hi, cc.hi) };
      break;
   }
   case FP_ALU_OP_MAX: {
      struct fx10_range p = fx10_mul(a, b);
      r = (struct fx10_range) { MAX2(p.lo, cc.lo), MAX2(p.hi, cc.hi) };
      break;
   }
   case FP_ALU_OP_CSEL:
      /* src[0] is the condition, the result src[1] or src[2] */
      r = fx10_hull(b, cc);
      break;
   default:
      return fx10_unknown;
   }
   static const float scale[] = { 1.0f, 2.0f, 4.0f, 0.5f };
   r = (struct fx10_range) { r.lo * scale[u->scale], r.hi * scale[u->scale] };
   if (u->condition)
      r = (struct fx10_range) { 0.0f, 1.0f };
   if (u->saturate)
      r = (struct fx10_range) { CLAMP(r.lo, 0.0f, 1.0f), CLAMP(r.hi, 0.0f, 1.0f) };
   if (isnan(r.lo) || isnan(r.hi))
      return fx10_unknown;
   return r;
}

/*
 * Whether an error in operand j of slot k of unit r reaches the result at
 * most as is: a factor whose partner stays within [-1, 1], the added term,
 * a value MIN/MAX/CSEL may pass on; never the CSEL condition.
 */
static bool
fx10_read_ok(struct fp_compiler *c, const struct fx10_range *value,
             const struct fp_unit *r, int k, int j)
{
   const struct fp_unit_slot *s = &r->slots[k];

   if (s->op == FP_ALU_OP_CSEL)
      return j != 0;
   if (j == 2)
      return !s->accumulate_this && !s->square_c;   /* a square doubles errors */
   return fx10_within(fx10_opnd_range(c, value, &s->src[1 - j]), -1.0f, 1.0f);
}

void
fp_mark_fx10(struct fp_compiler *c)
{
   struct fx10_range value[FP_MAX_UNITS];
   bool safe[FP_MAX_UNITS];   /* errors of the value reach the colour as is */
   int chain[FP_MAX_UNITS];   /* fixed10 values on the paths from it on */

   for (int i = 0; i < c->num_units; i++) {
      c->units[i].fx10 = false;
      value[i] = c->units[i].live ? fx10_unit_range(c, value, &c->units[i]) :
                                    fx10_unknown;
   }

   /* readers come after the units they read: backwards, every reader of a
    * unit is decided before it */
   for (int i = c->num_units - 1; i >= 0; i--) {
      struct fp_unit *u = &c->units[i];
      safe[i] = false;
      chain[i] = 0;
      if (!u->live || !fp_unit_has_value(u) || u->sfu || u->sfu_copy)
         continue;

      bool ok = true, read = false;
      int below = 0;
      for (int n = 0; n < c->num_tex && ok; n++)
         for (int k = 0; k < 2; k++)
            if (c->tex[n].live && c->tex[n].coord[k].kind == FP_OPND_VALUE &&
                c->tex[n].coord[k].index == i)
               ok = false;
      for (int r = i + 1; r < c->num_units && ok; r++) {
         struct fp_unit *ru = &c->units[r];
         if (!ru->live)
            continue;
         fp_unit_foreach_src(ru, k, j) {
            const struct fp_opnd *o = &ru->slots[k].src[j];
            if (o->kind != FP_OPND_VALUE || o->index != i)
               continue;
            read = true;
            below = MAX2(below, chain[r]);
            if (ru->sfu || ru->sfu_copy || ru->kill ||
                !fx10_read_ok(c, value, ru, k, j) ||
                (ru->output_chan < 0 && !safe[r]))
               ok = false;
         }
      }
      safe[i] = ok && read;
      u->fx10 = safe[i] && below < FX10_MAX_CHAIN &&
                fx10_within(value[i], FX10_MIN, FX10_MAX);
      chain[i] = below + u->fx10;
   }

   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "GRATE FRAG: fixed10:");
      for (int i = 0; i < c->num_units; i++)
         if (c->units[i].live && fp_unit_has_value(&c->units[i]))
            fprintf(stderr, " u%d[%g,%g]%s%s", i, value[i].lo, value[i].hi,
                    safe[i] ? "s" : "", c->units[i].fx10 ? "*" : "");
      fprintf(stderr, "\n");
   }
}
