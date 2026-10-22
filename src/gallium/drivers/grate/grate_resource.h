#ifndef GRATE_RESOURCE_H
#define GRATE_RESOURCE_H

#include "pipe/p_screen.h"
#include "util/u_transfer.h"

#define GRATE_MAX_MIP_LEVELS 13

struct grate_resource {
   struct pipe_resource b;
   struct grate_bo *bo;
   unsigned int pitch;
   unsigned int tiled : 1;
   unsigned int format : 5;

   /* Z24_UNORM_S8_UINT: the hardware has no packed depth/stencil surface, the
    * resource itself is the 16 bit depth plane and this the stencil one */
   struct grate_resource *stencil;

   /* byte offset and row pitch of each mip level inside the object */
   unsigned int level_offset[GRATE_MAX_MIP_LEVELS];
   unsigned int level_pitch[GRATE_MAX_MIP_LEVELS];
};

static inline struct grate_resource *
grate_resource(struct pipe_resource *resource)
{
   return (struct grate_resource *)resource;
}

int
grate_pixel_format(enum pipe_format format);

unsigned
grate_resource_blocksize(enum pipe_format format);

void
grate_context_resource_init(struct pipe_context *pcontext);

void
grate_screen_resource_init(struct pipe_screen *pscreen);

#endif
