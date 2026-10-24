/*
 * Values computed in the vertex shader (see grate_moved in
 * grate_compiler.h). Run late, when the preshader has turned what depends
 * on uniforms only into uniform slots: a value is then a chain of MADs over
 * varyings, uniform slots and constants, and it is linear in the varyings
 * when no product has a varying on both sides. Interpolating such an
 * expression gives the same as computing it from the interpolated varyings
 * (all of them perspective-correct, the uniform parts the same over the
 * primitive), so the vertex shader can compute it as an extra output.
 *
 * Moved are the coordinates of dependent texture reads (they then need no
 * sequential program), and the linear values the ALU reads when that takes
 * fewer varying components to interpolate (a component is a register
 * column of an EXEC: e.g. a sum of two varyings takes one instead of two).
 */
#include <string.h>

#include "fp_compiler.h"

#include "pipe/p_shader_tokens.h"
#include "util/u_memory.h"

#define FP_MOVE_MAX_UNDO 1024

struct fp_move {
   struct grate_moved *m;
   int node_of[FP_MAX_UNITS];     /* unit -> node | pixel << 16, -1: none yet */
   int num_declared;              /* inputs not moved ones */

   /* operands changed, to undo them */
   struct {
      struct fp_opnd *at;
      struct fp_opnd old;
   } undo[FP_MOVE_MAX_UNDO];
   int num_undo;
   bool full;                     /* the undo log is: change nothing more */
};

static bool
fp_move_src(struct fp_compiler *c, struct fp_move *mv, struct fp_opnd o,
            struct grate_moved_src *out, bool *pixel)
{
   memset(out, 0, sizeof(*out));
   out->negate = o.negate;
   *pixel = false;
   if (o.absolute || o.twice || o.minus_one)
      return false;

   switch (o.kind) {
   case FP_OPND_ZERO:
      out->kind = GRATE_MOVED_ZERO;
      return true;
   case FP_OPND_ONE:
      out->kind = GRATE_MOVED_ONE;
      return true;
   case FP_OPND_UNIFORM:
      out->kind = GRATE_MOVED_UNIFORM;
      out->index = o.index;
      return true;
   case FP_OPND_INPUT: {
      unsigned name = c->input_semantic[o.index] >> 8;
      if (o.index >= mv->num_declared ||
          c->input_interp[o.index] != TGSI_INTERPOLATE_PERSPECTIVE ||
          (name != TGSI_SEMANTIC_GENERIC && name != TGSI_SEMANTIC_TEXCOORD))
         return false;
      out->kind = GRATE_MOVED_OUTPUT;
      out->semantic = c->input_semantic[o.index];
      out->comp = o.comp;
      *pixel = true;
      return true;
   }
   case FP_OPND_VALUE:
      break;
   default:
      return false;
   }

   int i = o.index;
   struct fp_unit *u = &c->units[i];
   struct fp_unit_slot *s = &u->slots[0];
   if (mv->node_of[i] < 0) {
      if (!fp_unit_has_value(u) || u->num_slots != 1 || u->sfu || u->saturate ||
          u->scale || u->condition || u->blend_inv || s->op != FP_ALU_OP_MAD ||
          s->accumulate_this || s->accumulate_other || s->square_c)
         return false;

      struct grate_moved_src src[3];
      bool p[3];
      for (int k = 0; k < 3; k++)
         if (!fp_move_src(c, mv, s->src[k], &src[k], &p[k]))
            return false;
      /* a varying times a varying is not linear */
      if (p[0] && p[1])
         return false;

      struct grate_moved *m = mv->m;
      if (m->num_nodes == GRATE_MOVED_MAX_NODES)
         return false;
      memcpy(m->nodes[m->num_nodes], src, sizeof(src));
      mv->node_of[i] = m->num_nodes++ | (p[0] || p[1] || p[2]) << 16;
   }
   out->kind = GRATE_MOVED_NODE;
   out->index = mv->node_of[i] & 0xffff;
   *pixel = mv->node_of[i] >> 16;
   return true;
}

static void
fp_move_set(struct fp_move *mv, struct fp_opnd *at, struct fp_opnd o)
{
   if (mv->num_undo == FP_MOVE_MAX_UNDO) {
      mv->full = true;
      return;
   }
   mv->undo[mv->num_undo].at = at;
   mv->undo[mv->num_undo].old = *at;
   mv->num_undo++;
   *at = o;
}

static void
fp_move_undo(struct fp_move *mv, int to)
{
   while (mv->num_undo > to) {
      mv->num_undo--;
      *mv->undo[mv->num_undo].at = mv->undo[mv->num_undo].old;
   }
}

/* Varying components the program interpolates (the live code's) */
static int
fp_count_inputs(struct fp_compiler *c, bool *live)
{
   bool tex_live[FP_MAX_TEXTURES] = { false };
   BITSET_DECLARE(read, FP_MAX_INPUTS * 4);

   BITSET_ZERO(read);
   memset(live, 0, c->num_units * sizeof(*live));
   for (int i = c->num_units - 1; i >= 0; i--) {
      struct fp_unit *u = &c->units[i];
      if (u->output_chan >= 0 || u->kill)
         live[i] = true;
      if (!live[i])
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE)
            live[o->index] = true;
         else if (o->kind == FP_OPND_INPUT)
            BITSET_SET(read, o->index * 4 + o->comp);
         else if (o->kind == FP_OPND_TEXEL && !tex_live[o->index]) {
            tex_live[o->index] = true;
            for (int t = 0; t < 2; t++) {
               struct fp_opnd *x = &c->tex[o->index].coord[t];
               if (x->kind == FP_OPND_VALUE)
                  live[x->index] = true;
               else if (x->kind == FP_OPND_INPUT)
                  BITSET_SET(read, x->index * 4 + x->comp);
            }
         }
      }
   }
   return __bitset_count(read, BITSET_WORDS(FP_MAX_INPUTS * 4));
}

/* Replace every read of unit i's value by component comp of input */
static void
fp_move_reads(struct fp_compiler *c, struct fp_move *mv, int i, int input, int comp)
{
   for (int n = 0; n < c->num_units; n++) {
      struct fp_unit *u = &c->units[n];
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind != FP_OPND_VALUE || o->index != i)
            continue;
         struct fp_opnd x = { .kind = FP_OPND_INPUT, .index = input, .comp = comp,
                              .negate = o->negate, .absolute = o->absolute };
         fp_move_set(mv, o, x);
      }
   }
   for (int n = 0; n < c->num_tex; n++)
      for (int t = 0; t < 2; t++) {
         struct fp_opnd *o = &c->tex[n].coord[t];
         if (o->kind == FP_OPND_VALUE && o->index == i)
            fp_move_set(mv, o, (struct fp_opnd) { .kind = FP_OPND_INPUT, .index = input,
                                                  .comp = comp, .negate = o->negate,
                                                  .absolute = o->absolute });
      }
}

/*
 * Find what the vertex shader could compute: info.can_move. For
 * grate_fp_key.move also make it so: the values become new inputs and what
 * computed them dies (fp_relayout).
 */
void
fp_move_coords(struct fp_compiler *c)
{
   struct grate_fp_info *info = &c->fp->info;
   struct grate_moved *m = &info->moved;
   struct fp_move *mv = CALLOC_STRUCT(fp_move);
   int input = c->num_inputs;

   mv->m = m;
   mv->num_declared = c->num_inputs;
   memset(m, 0, sizeof(*m));
   memset(mv->node_of, -1, sizeof(mv->node_of));
   info->can_move = false;

   /* the coordinates of dependent reads */
   for (int n = 0; n < c->num_tex; n++) {
      if (c->tex[n].input >= 0 || m->num_outs == GRATE_MOVED_MAX ||
          input == FP_MAX_INPUTS)
         continue;

      /* both coordinates, or none: drop what a failed one added */
      struct grate_moved saved = *m;
      int saved_node_of[FP_MAX_UNITS];
      memcpy(saved_node_of, mv->node_of, sizeof(saved_node_of));
      struct grate_moved_src root[2];
      bool p[2];
      if (!fp_move_src(c, mv, c->tex[n].coord[0], &root[0], &p[0]) ||
          !fp_move_src(c, mv, c->tex[n].coord[1], &root[1], &p[1])) {
         *m = saved;
         memcpy(mv->node_of, saved_node_of, sizeof(saved_node_of));
         continue;
      }

      if (mv->num_undo + 2 > FP_MOVE_MAX_UNDO)
         break;
      int k = m->num_outs++;
      m->out[k].root[0] = root[0];
      m->out[k].root[1] = root[1];
      m->out[k].num_roots = 2;
      m->out[k].input = input;
      for (int j = 0; j < 2; j++)
         fp_move_set(mv, &c->tex[n].coord[j], (struct fp_opnd) {
            .kind = FP_OPND_INPUT, .index = input, .comp = j });
      input++;
   }

   /* linear values the ALU reads, outermost first, while that saves
    * varying components */
   bool *live = MALLOC(FP_MAX_UNITS * sizeof(*live));
   int count = fp_count_inputs(c, live);
   int alu = -1;           /* moved input taking them */
   for (int i = c->num_units - 1; i >= 0; i--) {
      struct fp_unit *u = &c->units[i];
      if (!live[i] || !fp_unit_has_value(u))
         continue;
      if (alu < 0 && (m->num_outs == GRATE_MOVED_MAX || input == FP_MAX_INPUTS))
         break;

      struct grate_moved saved = *m;
      int saved_node_of[FP_MAX_UNITS];
      memcpy(saved_node_of, mv->node_of, sizeof(saved_node_of));
      int undo = mv->num_undo;
      struct grate_moved_src root;
      bool pixel;
      if (!fp_move_src(c, mv, (struct fp_opnd) { .kind = FP_OPND_VALUE, .index = i },
                       &root, &pixel) ||
          !pixel || root.kind != GRATE_MOVED_NODE) {
         *m = saved;
         memcpy(mv->node_of, saved_node_of, sizeof(saved_node_of));
         continue;
      }

      int k = alu >= 0 ? alu : m->num_outs;
      int in = alu >= 0 ? m->out[k].input : input;
      int comp = alu >= 0 ? m->out[k].num_roots : 0;
      fp_move_reads(c, mv, i, in, comp);
      int now = fp_count_inputs(c, live);
      if (now >= count || mv->full) {
         fp_move_undo(mv, undo);
         *m = saved;
         memcpy(mv->node_of, saved_node_of, sizeof(saved_node_of));
         fp_count_inputs(c, live);
         continue;
      }
      count = now;
      if (alu < 0) {
         alu = m->num_outs++;
         m->out[alu].input = input++;
      }
      m->out[alu].root[m->out[alu].num_roots++] = root;
      if (m->out[alu].num_roots == 4)
         alu = -1;
   }
   FREE(live);

   info->can_move = m->num_outs > 0;
   if (c->key->move) {
      for (int k = 0; k < m->num_outs; k++) {
         c->input_semantic[m->out[k].input] = GRATE_MOVED_SEMANTIC(k);
         c->input_interp[m->out[k].input] = TGSI_INTERPOLATE_PERSPECTIVE;
      }
      c->num_inputs = input;
   } else {
      fp_move_undo(mv, 0);
      memset(m, 0, sizeof(*m));
   }
   FREE(mv);
}
