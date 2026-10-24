/*
 * Texture coordinates computed in the vertex shader (see grate_moved in
 * grate_compiler.h). Run late, when the preshader has turned what depends
 * on uniforms only into uniform slots: the coordinates of a dependent read
 * are then a chain of MADs over varyings, uniform slots and constants, and
 * they are linear in the varyings when no product has a varying on both
 * sides. Interpolating such an expression gives the same as computing it
 * from the interpolated varyings (all of them perspective-correct, the
 * expression's uniform parts the same over the primitive), so the vertex
 * shader can compute it as an extra output.
 */
#include <string.h>

#include "fp_compiler.h"

#include "pipe/p_shader_tokens.h"

struct fp_move {
   struct grate_moved *m;
   int node_of[FP_MAX_UNITS];     /* unit -> node, -1 if not one yet */
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
      if (c->input_interp[o.index] != TGSI_INTERPOLATE_PERSPECTIVE ||
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
      mv->node_of[i] = m->num_nodes++;
      /* whether it depends on a varying, in the same array */
      mv->node_of[i] |= (p[0] || p[1] || p[2]) << 16;
   }
   out->kind = GRATE_MOVED_NODE;
   out->index = mv->node_of[i] & 0xffff;
   *pixel = mv->node_of[i] >> 16;
   return true;
}

/*
 * Find the dependent reads the vertex shader could compute the coordinates
 * of: info.can_move. For grate_fp_key.move, also make them so: the
 * coordinates become new inputs, what computed them dies (fp_relayout).
 */
void
fp_move_coords(struct fp_compiler *c)
{
   struct grate_fp_info *info = &c->fp->info;
   struct fp_move mv = { .m = &info->moved };
   int input = c->num_inputs;

   memset(&info->moved, 0, sizeof(info->moved));
   memset(mv.node_of, -1, sizeof(mv.node_of));
   info->can_move = false;

   for (int n = 0; n < c->num_tex; n++) {
      struct grate_moved *m = &info->moved;
      if (c->tex[n].input >= 0 || m->num_coords == GRATE_MOVED_MAX ||
          input == FP_MAX_INPUTS)
         continue;

      /* both coordinates, or none: drop what a failed one added */
      struct grate_moved saved = *m;
      int saved_node_of[FP_MAX_UNITS];
      memcpy(saved_node_of, mv.node_of, sizeof(saved_node_of));
      struct grate_moved_src root[2];
      bool p[2];
      if (!fp_move_src(c, &mv, c->tex[n].coord[0], &root[0], &p[0]) ||
          !fp_move_src(c, &mv, c->tex[n].coord[1], &root[1], &p[1])) {
         *m = saved;
         memcpy(mv.node_of, saved_node_of, sizeof(saved_node_of));
         continue;
      }

      info->can_move = true;
      int k = m->num_coords++;
      m->coord[k].root[0] = root[0];
      m->coord[k].root[1] = root[1];
      m->coord[k].input = input;
      if (c->key->move) {
         c->input_semantic[input] = GRATE_MOVED_SEMANTIC(k);
         c->input_interp[input] = TGSI_INTERPOLATE_PERSPECTIVE;
         for (int j = 0; j < 2; j++)
            c->tex[n].coord[j] = (struct fp_opnd) {
               .kind = FP_OPND_INPUT, .index = input, .comp = j };
      }
      input++;
   }

   if (!c->key->move)
      memset(&info->moved, 0, sizeof(info->moved));
   else
      c->num_inputs = input;
}
