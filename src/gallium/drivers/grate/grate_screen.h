#ifndef GRATE_SCREEN_H
#define GRATE_SCREEN_H

#include "pipe/p_screen.h"
#include "util/slab.h"

struct grate_device;

extern uint32_t grate_debug;

#define GRATE_DEBUG_UNIMPLEMENTED 0x1
#define GRATE_DEBUG_TGSI 0x2
#define GRATE_DEBUG_TRACE 0x4

/* Largest point (pixels). The hardware draws at least 10000 exactly (a point
 * whose centre is past the guard band is dropped, as GL allows); this is the
 * framebuffer size, so that dEQP rasterization.limits.points can draw one at
 * its edge. Vertex shaders clamp gl_PointSize to it, the hardware does not. */
#define GRATE_MAX_POINT_SIZE 2048.0f

struct disk_cache;

struct grate_screen {
   struct pipe_screen base;

   /* compiled fragment programs (grate_program.c) and Mesa's own shader
    * cache, keyed to this build of the driver */
   struct disk_cache *disk_cache;

   struct slab_parent_pool transfer_pool;

   int fd;
   struct grate_device *drm;
};

static inline struct grate_screen *
grate_screen(struct pipe_screen *screen)
{
   return (struct grate_screen *)screen;
}

struct pipe_screen *
grate_screen_create(int fd);

/* Whether the TEX unit samples the format */
bool
grate_sampler_format(enum pipe_format format);

/* Texel channel the TEX unit returns channel c of the format in */
unsigned
grate_sampler_channel(enum pipe_format format, unsigned c);

#endif
