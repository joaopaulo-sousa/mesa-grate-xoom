/*
 * Reassociation for shorter dependency chains. A multi-row program has 12
 * ALU packets at most (4 EXECs of 3), and a value one packet computes is
 * read by the next only, so the depth of the chains, more than the number
 * of operations, decides whether a shader fits. Two rewrites, on plain
 * MADs (no saturate on the inner one, no absolute values):
 *
 * - (x * y + k) * w + z with y, k, w, z uniform becomes x * (y * w) +
 *   (k * w + z), and (a * b + x) * w + z becomes x * w + (a * b * w + z):
 *   the uniform parts go to the preshader, one operation remains per pixel
 *   instead of two. E.g. smoothstep's (x - edge0) * (1 / (edge1 - edge0)).
 *
 * - a product of three factors (x * y) * z is regrouped so that the two
 *   available first are multiplied first: t * (t * (3 - 2t)) becomes
 *   (t * t) * (3 - 2t), t * t then going in parallel with 3 - 2t.
 *
 * Runs before fp_preshade, which takes the uniform-only units out.
 */
#include <math.h>
#include <string.h>

#include "fp_compiler.h"

#include "util/macros.h"

/* Move the last unit to position pos, renumbering the units after it */
static void
fp_move_last_unit(struct fp_compiler *c, int pos)
{
   int last = c->num_units - 1;
   struct fp_unit moved = c->units[last];

   memmove(&c->units[pos + 1], &c->units[pos], (last - pos) * sizeof(c->units[0]));
   c->units[pos] = moved;

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE && o->index >= pos && i != pos)
            o->index = o->index == last ? pos : o->index + 1;
      }
   }
   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < 2; k++) {
         struct fp_opnd *o = &c->tex[n].coord[k];
         if (o->kind == FP_OPND_VALUE && o->index >= pos)
            o->index = o->index == last ? pos : o->index + 1;
      }
}

struct fp_reassoc {
   bool alive[FP_MAX_UNITS];     /* read by a colour write, discard or TEX */
   bool uniform[FP_MAX_UNITS];   /* the same for every pixel (preshader) */
   int depth[FP_MAX_UNITS];      /* ALU operations on the longest path */
   int readers[FP_MAX_UNITS];    /* by live units */
};

/* A live single-slot MAD, which the rewrites may change: a value or, as
 * the outer unit, a colour write */
static bool
fp_plain_mad(const struct fp_reassoc *r, const struct fp_compiler *c, int i)
{
   const struct fp_unit *u = &c->units[i];
   const struct fp_unit_slot *s = &u->slots[0];
   return r->alive[i] && !u->kill && u->num_slots == 1 && !u->sfu &&
          !u->sfu_copy && s->op == FP_ALU_OP_MAD && !s->accumulate_this &&
          !s->accumulate_other && !s->square_c;
}

static bool
fp_opnd_uniform(const struct fp_reassoc *r, const struct fp_opnd *o)
{
   return o->kind == FP_OPND_UNIFORM || o->kind == FP_OPND_ZERO ||
          o->kind == FP_OPND_ONE ||
          (o->kind == FP_OPND_VALUE && r->uniform[o->index]);
}

static int
fp_opnd_depth(const struct fp_reassoc *r, const struct fp_opnd *o)
{
   return o->kind == FP_OPND_VALUE ? r->depth[o->index] : 0;
}

static void
fp_reassoc_scan(struct fp_compiler *c, struct fp_reassoc *r)
{
   memset(r->alive, 0, sizeof(r->alive));
   memset(r->readers, 0, sizeof(r->readers));
   for (int i = c->num_units - 1; i >= 0; i--) {
      struct fp_unit *u = &c->units[i];
      r->alive[i] |= u->output_chan >= 0 || u->kill;
      if (!r->alive[i])
         continue;
      fp_unit_foreach_src(u, k, j) {
         const struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE) {
            r->alive[o->index] = true;
            r->readers[o->index]++;
         } else if (o->kind == FP_OPND_TEXEL) {
            for (int t = 0; t < 2; t++)
               if (c->tex[o->index].coord[t].kind == FP_OPND_VALUE) {
                  r->alive[c->tex[o->index].coord[t].index] = true;
                  r->readers[c->tex[o->index].coord[t].index] += 2;   /* keep */
               }
         }
      }
   }

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      bool uniform = fp_unit_has_value(u);
      int depth = 0;
      fp_unit_foreach_src(u, k, j) {
         const struct fp_opnd *o = &u->slots[k].src[j];
         uniform &= fp_opnd_uniform(r, o);
         depth = MAX2(depth, fp_opnd_depth(r, o));
      }
      r->uniform[i] = uniform;
      r->depth[i] = uniform ? 0 : depth + 1;
   }
}

/* The inner unit an operand of a rewrite may fold: a plain MAD without
 * saturate that only this operand reads */
static struct fp_unit *
fp_inner(struct fp_compiler *c, const struct fp_reassoc *r, const struct fp_opnd *o)
{
   if (o->kind != FP_OPND_VALUE || o->absolute || o->minus_one || o->twice ||
       r->readers[o->index] != 1 ||
       !fp_plain_mad(r, c, o->index) || c->units[o->index].saturate ||
       !fp_unit_has_value(&c->units[o->index]))
      return NULL;
   return &c->units[o->index];
}

static struct fp_opnd
fp_negated(struct fp_opnd o, bool negate)
{
   o.negate ^= negate;
   return o;
}

/* The value of a constant operand: 0, 1 or an immediate */
static bool
fp_opnd_known(struct fp_compiler *c, const struct fp_opnd *o, float *v)
{
   if (o->kind == FP_OPND_ZERO)
      *v = 0.0f;
   else if (o->kind == FP_OPND_ONE)
      *v = 1.0f;
   else if (o->kind == FP_OPND_UNIFORM && BITSET_TEST(c->fp->info.imm_mask, o->index))
      *v = fp20_to_float(c->fp->info.consts[o->index]);
   else
      return false;
   if (o->minus_one)
      *v -= 1.0f;
   if (o->twice)
      *v *= 2.0f;
   if (o->absolute)
      *v = fabsf(*v);
   if (o->negate)
      *v = -*v;
   return true;
}

/*
 * a * b + cc: an immediate when all three are constants, else a new unit
 * put at *pos, before the unit being rewritten (*pos moves on with it)
 */
static struct fp_opnd
fp_mad_value(struct fp_compiler *c, struct fp_opnd a, struct fp_opnd b,
             struct fp_opnd cc, int *pos)
{
   float va, vb, vc;

   if (fp_opnd_known(c, &a, &va) && fp_opnd_known(c, &b, &vb) &&
       fp_opnd_known(c, &cc, &vc))
      return fp_opnd_const(c, va * vb + vc);

   struct fp_unit *n = fp_new_unit(c);
   if (c->failed)
      return fp_opnd_zero(c, false);
   fp_unit_op(&n->slots[0], FP_ALU_OP_MAD, a, b, cc);
   fp_move_last_unit(c, *pos);
   return fp_opnd_value(c, &c->units[(*pos)++]);
}

/*
 * (x * y + k) * w + z  -> x * (y * w) + (k * w + z)
 * (a * b + x) * w + z  -> x * w + (a * b * w + z)
 * with x the only operand that depends on the pixel
 */
static bool
fp_fold_affine(struct fp_compiler *c, struct fp_reassoc *r, int i)
{
   struct fp_unit_slot *s = &c->units[i].slots[0];

   if (!fp_opnd_uniform(r, &s->src[2]))
      return false;
   for (int at = 0; at < 2; at++) {
      struct fp_opnd w = s->src[1 - at];
      struct fp_unit *v = fp_inner(c, r, &s->src[at]);
      if (!v || !fp_opnd_uniform(r, &w) || w.absolute)
         continue;
      const struct fp_unit_slot *vs = &v->slots[0];
      int pixel = -1;
      for (int j = 0; j < 3; j++)
         if (!fp_opnd_uniform(r, &vs->src[j]))
            pixel = pixel < 0 ? j : 3;
      if (pixel < 0 || pixel == 3)
         continue;

      /* the sign of the folded operand goes with w */
      w = fp_negated(w, s->src[at].negate);
      struct fp_opnd x = vs->src[pixel], z = s->src[2];
      struct fp_opnd a0 = vs->src[0], a1 = vs->src[1], a2 = vs->src[2];
      struct fp_opnd zero = fp_opnd_zero(c, false), factor, term;
      int pos = i;   /* the rewritten unit, after the units put before it */

      if (pixel == 2) {
         /* (a * b + x) * w + z = x * w + (a * (b * w) + z) */
         factor = w;
         term = fp_mad_value(c, a0, fp_mad_value(c, a1, w, zero, &pos), z, &pos);
      } else {
         /* (x * y + k) * w + z = x * (y * w) + (k * w + z) */
         factor = fp_mad_value(c, pixel ? a0 : a1, w, zero, &pos);
         term = fp_mad_value(c, a2, w, z, &pos);
      }
      fp_unit_op(&c->units[pos].slots[0], FP_ALU_OP_MAD, x, factor, term);
      return true;
   }
   return false;
}

/* (x * y) * z -> the two shallower factors first */
static bool
fp_regroup_product(struct fp_compiler *c, struct fp_reassoc *r, int i)
{
   struct fp_unit_slot *s = &c->units[i].slots[0];

   if (!fp_opnd_is_const(c, &s->src[2], 0.0f))
      return false;
   for (int a = 0; a < 2; a++) {
      struct fp_unit *v = fp_inner(c, r, &s->src[a]);
      if (!v || !fp_opnd_is_const(c, &v->slots[0].src[2], 0.0f))
         continue;

      struct fp_opnd f[3] = { v->slots[0].src[0], v->slots[0].src[1], s->src[1 - a] };
      bool negate = s->src[a].negate;
      /* a level is gained when an inner factor is deeper than both others */
      int deepest = fp_opnd_depth(r, &f[0]) > fp_opnd_depth(r, &f[1]) ? 0 : 1;
      int dd = fp_opnd_depth(r, &f[deepest]);
      if (dd <= fp_opnd_depth(r, &f[1 - deepest]) || dd <= fp_opnd_depth(r, &f[2]))
         continue;

      /* the pair goes before this one (the outer factor may come after
       * the inner unit) */
      struct fp_opnd d = f[deepest];
      int pos = i;
      struct fp_opnd pair = fp_mad_value(c, fp_negated(f[deepest == 0 ? 1 : 0], negate), f[2],
                                         fp_opnd_zero(c, false), &pos);
      if (d.kind == FP_OPND_VALUE && d.index >= i)
         d.index += pos - i;
      fp_unit_op(&c->units[pos].slots[0], FP_ALU_OP_MAD, pair, d, fp_opnd_zero(c, false));
      return true;
   }
   return false;
}

void
fp_reassociate(struct fp_compiler *c)
{
   struct fp_reassoc r;
   bool progress = true;

   for (int pass = 0; progress && pass < 16 && !c->failed; pass++) {
      progress = false;
      fp_reassoc_scan(c, &r);
      for (int i = 0; i < c->num_units && !progress; i++) {
         if (!fp_plain_mad(&r, c, i) || r.uniform[i])
            continue;
         progress = fp_fold_affine(c, &r, i) || fp_regroup_product(c, &r, i);
      }
   }
}

/*
 * Scaling by 2, 4 or 1/2 is free on Tegra20: an ALU instruction scales its
 * result (fragment-shader-isa.md, "scale result") and an operand can be
 * read doubled ("scale by two"). After the preshader, a multiply x * k by
 * such a constant becomes
 * - the scaled instruction computing x, when nothing else reads x: one
 *   operation fewer on the path;
 * - for k = 2, x read doubled by every reader of the product, when those
 *   are all ALU operations: e.g. a texel * 2 - 1 is one MAD.
 */
static bool
fp_scale_const(struct fp_compiler *c, const struct fp_opnd *o, enum fp_scale *scale)
{
   if (fp_opnd_is_const(c, o, 2.0f))
      *scale = FP_SCALE_MUL2;
   else if (fp_opnd_is_const(c, o, 4.0f))
      *scale = FP_SCALE_MUL4;
   else if (fp_opnd_is_const(c, o, 0.5f))
      *scale = FP_SCALE_DIV2;
   else
      return false;
   return true;
}

/* The product's result scale moved into the unit computing x */
static bool
fp_scale_result(struct fp_compiler *c, struct fp_reassoc *r, int i,
                const struct fp_opnd *x, enum fp_scale scale)
{
   struct fp_unit *u = &c->units[i];

   if (x->kind != FP_OPND_VALUE || x->absolute || x->minus_one || x->twice ||
       r->readers[x->index] != 1)
      return false;
   struct fp_unit *v = &c->units[x->index];
   struct fp_unit_slot s = v->slots[0];
   if (!r->alive[x->index] || !fp_unit_has_value(v) || v->num_slots != 1 || v->sfu ||
       v->sfu_copy || v->saturate || v->scale != FP_SCALE_NONE ||
       s.accumulate_this || s.accumulate_other || s.square_c)
      return false;
   if (x->negate) {
      /* -(a * b + c) = -a * b - c; MIN, MAX and CSEL do not turn */
      if (s.op != FP_ALU_OP_MAD)
         return false;
      s.src[0].negate = !s.src[0].negate;
      s.src[2].negate = !s.src[2].negate;
   }
   u->slots[0] = s;
   u->scale = scale;
   return true;
}

/* x read doubled by the readers of the product i */
static bool
fp_scale_operand(struct fp_compiler *c, struct fp_reassoc *r, int i, const struct fp_opnd *x)
{
   const struct fp_unit *u = &c->units[i];

   if (!fp_unit_has_value(u) || u->saturate || x->absolute || x->twice || x->minus_one ||
       r->readers[i] == 0)
      return false;
   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < 2; k++)
         if (c->tex[n].coord[k].kind == FP_OPND_VALUE && c->tex[n].coord[k].index == i)
            return false;
   for (int n = i + 1; n < c->num_units; n++) {
      const struct fp_unit *v = &c->units[n];
      if (!r->alive[n])
         continue;
      fp_unit_foreach_src(v, k, j)
         if (v->slots[k].src[j].kind == FP_OPND_VALUE && v->slots[k].src[j].index == i &&
             (v->sfu || v->sfu_copy || v->slots[k].src[j].twice ||
              v->slots[k].src[j].minus_one))
            return false;
   }
   for (int n = i + 1; n < c->num_units; n++) {
      struct fp_unit *v = &c->units[n];
      if (!r->alive[n])
         continue;
      fp_unit_foreach_src(v, k, j) {
         struct fp_opnd *o = &v->slots[k].src[j];
         if (o->kind != FP_OPND_VALUE || o->index != i)
            continue;
         /* |2 * +-x| = |2 * x|, - (2 * -x) = 2 * x */
         bool negate = o->negate ^ (o->absolute ? false : x->negate);
         bool absolute = o->absolute;
         *o = *x;
         o->twice = true;
         o->negate = negate;
         o->absolute = absolute;
      }
   }
   return true;
}

void
fp_fold_scales(struct fp_compiler *c)
{
   struct fp_reassoc r;
   bool progress = true;

   while (progress && !c->failed) {
      progress = false;
      fp_reassoc_scan(c, &r);
      for (int i = 0; i < c->num_units && !progress; i++) {
         struct fp_unit *u = &c->units[i];
         struct fp_unit_slot *s = &u->slots[0];
         enum fp_scale scale;
         if (!fp_plain_mad(&r, c, i) || u->kill || u->scale != FP_SCALE_NONE ||
             !fp_opnd_is_const(c, &s->src[2], 0.0f))
            continue;
         for (int a = 0; a < 2 && !progress; a++) {
            if (!fp_scale_const(c, &s->src[1 - a], &scale))
               continue;
            struct fp_opnd x = s->src[a];
            progress = fp_scale_result(c, &r, i, &x, scale) ||
                       (scale == FP_SCALE_MUL2 && fp_scale_operand(c, &r, i, &x));
         }
      }
   }
}

/*
 * A blend's d * (1 - y) + ts reads y less one (one MAD). In a sequential
 * program the colour goes to the register pair a texel or the destination
 * colour may be in, and four channels reading y and ts to the end leave no
 * half free to write: there it becomes ts' = d + ts, then ts' - d * y,
 * taking ts in earlier.
 */
void
fp_expand_blend_inv(struct fp_compiler *c)
{
   for (int i = 0; i < c->num_units && !c->failed; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || !u->blend_inv)
         continue;
      struct fp_unit_slot s = u->slots[0];
      struct fp_opnd y = s.src[1];
      y.minus_one = false;
      y.negate = !y.negate;

      struct fp_unit *n = fp_new_unit(c);
      if (c->failed)
         return;
      fp_unit_op(&n->slots[0], FP_ALU_OP_MAD, s.src[0], (struct fp_opnd) { .kind = FP_OPND_ONE },
                 s.src[2]);
      n->live = true;
      fp_move_last_unit(c, i);
      u = &c->units[i + 1];
      u->blend_inv = false;
      fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, fp_negated(s.src[0], true), y,
                 fp_opnd_value(c, &c->units[i]));
      i++;
   }
}
