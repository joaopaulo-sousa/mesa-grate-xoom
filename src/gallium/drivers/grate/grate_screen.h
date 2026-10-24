#ifndef GRATE_SCREEN_H
#define GRATE_SCREEN_H

#include "pipe/p_screen.h"
#include "util/slab.h"

struct grate_device;

extern uint32_t grate_debug;

#define GRATE_DEBUG_UNIMPLEMENTED 0x1
#define GRATE_DEBUG_TGSI 0x2
#define GRATE_DEBUG_TRACE 0x4

struct grate_screen {
   struct pipe_screen base;

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
