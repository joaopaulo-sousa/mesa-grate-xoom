#ifndef GRATE_PROGRAM_H
#define GRATE_PROGRAM_H

#include "pipe/p_context.h"
#include "pipe/p_state.h"

#include "grate_compiler.h"

struct grate_shader_blob {
   uint32_t *commands;
   int num_commands;
};

struct grate_vertex_shader_state {
   struct pipe_shader_state base;
   struct grate_shader_blob blob;
   uint16_t output_mask;
   uint16_t output_semantic[16];
   float immediates[GRATE_VP_MAX_IMMEDIATES][4];
   int num_immediates;
};

/* A fragment program compiled for one blend state (see grate_fp_key) */
struct grate_fs_variant {
   struct grate_fp_key key;
   struct grate_shader_blob blob;
   struct grate_fp_info info;
};

#define GRATE_FS_MAX_VARIANTS 16

struct grate_fragment_shader_state {
   struct pipe_shader_state base;     /* owns base.tokens */

   /* the variant for the bound blend state, set by grate_fs_update() */
   struct grate_shader_blob blob;
   struct grate_fp_info info;

   bool discards;                     /* KILL/KILL_IF: discards fragments */

   struct grate_fs_variant variants[GRATE_FS_MAX_VARIANTS];
   unsigned num_variants;
   unsigned next_evict;
};

struct grate_context;

/* Select (compiling if needed) the fragment program variant for the bound
 * blend state; call before emitting a draw. */
void
grate_fs_update(struct grate_context *context);

void
grate_context_program_init(struct pipe_context *pcontext);

#endif
