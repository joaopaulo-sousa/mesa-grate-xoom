#include <stdlib.h>
#include <stdio.h>

#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_pack_color.h"
#include "util/u_surface.h"
#include "util/u_transfer.h"
#include "util/u_inlines.h"

#include "grate_common.h"
#include "grate_bo.h"
#include "grate_context.h"
#include "grate_resource.h"
#include "grate_screen.h"

#include "host1x01_hardware.h"

/*
 * XXX Required to access winsys_handle internals. Should go away in favour
 * of some abstraction to handle handles in a Tegra-specific winsys
 * implementation.
 */
#include "drm-uapi/drm_fourcc.h"

#include "frontend/drm_driver.h"


static bool
grate_resource_get_handle(struct pipe_screen *pscreen,
                          struct pipe_context *context,
                          struct pipe_resource *presource,
                          struct winsys_handle *handle,
                          unsigned usage)
{
   struct grate_resource *resource;
   int ret;

   if (presource->target == PIPE_BUFFER)
      return false;

   resource = grate_resource(presource);

   if (handle->type == WINSYS_HANDLE_TYPE_KMS) {
      ret = grate_bo_get_handle(resource->bo, &handle->handle);
      if (ret < 0) {
         fprintf(stderr, "grate_bo_get_handle() failed: %d\n", ret);
         return false;
      }
   } else if (handle->type == WINSYS_HANDLE_TYPE_FD) {
      ret = grate_bo_export(resource->bo, 0);
      if (0 < ret) {
         handle->handle = ret;
      } else {
         fprintf(stderr, "grate_bo_export() failed: %d\n", ret);
         return false;
      }
   } else {
      fprintf(stdout, "unsupported handle type: %d\n", handle->type);
      return false;
   }

   handle->stride = resource->pitch;
   handle->offset = 0;
   handle->modifier = DRM_FORMAT_MOD_LINEAR;
   return true;
}

/*
 * DRI3 and the dmabuf export path ask for the layout through this rather than
 * through get_handle, and without it they end up passing DRM_FORMAT_MOD_INVALID
 * to the X server, which answers BadAlloc. We only ever hand out single plane
 * linear buffers, so the answers are all short.
 * (from libre-tegra/mesa grate-tmp, Ion Agorria)
 */
static bool
grate_resource_get_param(struct pipe_screen *pscreen,
                         struct pipe_context *pcontext,
                         struct pipe_resource *presource,
                         unsigned plane, unsigned layer, unsigned level,
                         enum pipe_resource_param param,
                         unsigned usage, uint64_t *value)
{
   struct grate_resource *resource = grate_resource(presource);
   struct winsys_handle handle;

   switch (param) {
   case PIPE_RESOURCE_PARAM_NPLANES:
      *value = 1;
      return true;
   case PIPE_RESOURCE_PARAM_STRIDE:
      *value = resource->pitch;
      return true;
   case PIPE_RESOURCE_PARAM_OFFSET:
      *value = 0;
      return true;
   case PIPE_RESOURCE_PARAM_MODIFIER:
      *value = DRM_FORMAT_MOD_LINEAR;
      return true;
   case PIPE_RESOURCE_PARAM_LAYER_STRIDE:
      *value = (uint64_t)resource->pitch * presource->height0;
      return true;
   case PIPE_RESOURCE_PARAM_HANDLE_TYPE_SHARED:
   case PIPE_RESOURCE_PARAM_HANDLE_TYPE_KMS:
   case PIPE_RESOURCE_PARAM_HANDLE_TYPE_FD:
      memset(&handle, 0, sizeof(handle));
      if (param == PIPE_RESOURCE_PARAM_HANDLE_TYPE_FD)
         handle.type = WINSYS_HANDLE_TYPE_FD;
      else if (param == PIPE_RESOURCE_PARAM_HANDLE_TYPE_KMS)
         handle.type = WINSYS_HANDLE_TYPE_KMS;
      else
         handle.type = WINSYS_HANDLE_TYPE_SHARED;

      if (!grate_resource_get_handle(pscreen, pcontext, presource, &handle, usage))
         return false;

      *value = handle.handle;
      return true;
   default:
      return false;
   }
}

static void
grate_resource_destroy(struct pipe_screen *pscreen,
                       struct pipe_resource *presource)
{
   struct grate_resource *resource = grate_resource(presource);

   grate_bo_unref(resource->bo);
   FREE(resource);
}

static void *
grate_resource_transfer_map(struct pipe_context *pcontext,
                            struct pipe_resource *presource,
                            unsigned level, unsigned usage,
                            const struct pipe_box *box,
                            struct pipe_transfer **transfer)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_resource *resource = grate_resource(presource);
   void *ret = NULL;
   struct pipe_transfer *ptrans;

   if (usage & PIPE_MAP_DIRECTLY)
      return NULL;

   /*
    * Nothing records which resource a submitted job touched, so the only safe
    * thing is to let everything already submitted land before handing out a
    * CPU pointer. Otherwise a readback races the GPU still writing the render
    * target, which is what made large frames come back half drawn.
    */
   if (!(usage & PIPE_MAP_UNSYNCHRONIZED))
      grate_context_flush_streams(context);

   ptrans = slab_alloc(&context->transfer_pool);
   if (!ptrans)
      return NULL;

   if (grate_bo_map(resource->bo, &ret))
      return NULL;

   memset(ptrans, 0, sizeof(*ptrans));

   pipe_resource_reference(&ptrans->resource, presource);
   ptrans->resource = presource;
   ptrans->level = level;
   ptrans->usage = usage;
   ptrans->box = *box;
   unsigned lvl = MIN2(level, (unsigned)(GRATE_MAX_MIP_LEVELS - 1));
   unsigned lpitch = resource->level_pitch[lvl] ? resource->level_pitch[lvl]
                                                : resource->pitch;

   ptrans->stride = lpitch;
   ptrans->layer_stride = ptrans->stride;
   *transfer = ptrans;

   return (uint8_t *)ret + resource->level_offset[lvl] +
          box->y * lpitch +
          box->x * util_format_get_blocksize(presource->format);
}

static void
grate_resource_transfer_flush_region(struct pipe_context *pcontext,
                                     struct pipe_transfer *transfer,
                                     const struct pipe_box *box)
{
   grate_unimplemented();
}

static void
grate_resource_transfer_unmap(struct pipe_context *pcontext,
                              struct pipe_transfer *transfer)
{
   struct grate_context *context = grate_context(pcontext);

   grate_bo_unmap(grate_resource(transfer->resource)->bo);

   pipe_resource_reference(&transfer->resource, NULL);
   slab_free(&context->transfer_pool, transfer);
}

int
grate_pixel_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_A8_UNORM:
      return TGR3D_SURF_FORMAT_A8;
   case PIPE_FORMAT_L8_UNORM:
      return TGR3D_SURF_FORMAT_L8;
    case PIPE_FORMAT_S8_UINT:
      return TGR3D_SURF_FORMAT_S8;
    case PIPE_FORMAT_L8A8_UNORM:
      return TGR3D_SURF_FORMAT_L8A8;
   case PIPE_FORMAT_B5G6R5_UNORM:
      return TGR3D_SURF_FORMAT_B5G6R5;
   case PIPE_FORMAT_B5G5R5A1_UNORM:
      return TGR3D_SURF_FORMAT_A1B5G5R5;
   case PIPE_FORMAT_B4G4R4A4_UNORM:
      return TGR3D_SURF_FORMAT_A4B4G4R4;
   case PIPE_FORMAT_Z16_UNORM:
      return TGR3D_SURF_FORMAT_Z16;
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return TGR3D_SURF_FORMAT_R8G8B8A8;
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return TGR3D_SURF_FORMAT_R16G16B16A16_float;
   default:
      return -1;
   }
}

static struct pipe_resource *
grate_screen_resource_create(struct pipe_screen *pscreen,
                             const struct pipe_resource *template)
{
   struct grate_screen *screen = grate_screen(pscreen);
   struct grate_resource *resource;
   uint32_t flags = 0, height, size;

   resource = CALLOC_STRUCT(grate_resource);
   if (!resource)
      return NULL;

   resource->b = *template;

   pipe_reference_init(&resource->b.reference, 1);
   resource->b.screen = pscreen;

   resource->pitch = template->width0 * util_format_get_blocksize(template->format);
   height = template->height0;

   resource->tiled = 0;
   if (template->bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW |
                         PIPE_BIND_SCANOUT | PIPE_BIND_DEPTH_STENCIL)) {
      if (template->bind & PIPE_BIND_DEPTH_STENCIL)
         resource->pitch = align(resource->pitch, 256);
      else if (!(template->bind & PIPE_BIND_SAMPLER_VIEW))
         resource->pitch = align(resource->pitch, 32);
      else
         /*
          * The TEX unit derives the row pitch from the width (measured with
          * RGBA8888): power-of-two widths are aligned to 16 bytes, others to
          * 64 bytes.
          */
         resource->pitch = align(resource->pitch,
                                 util_is_power_of_two_nonzero(template->width0) ?
                                 16 : 64);

      /*
       * The sampler addresses rows within the power-of-two extent enclosing
       * the height, and reading past the end of the object faults the SMMU
       * (tegra-mc texsrd2 page faults): back those rows with memory.
       * (from libre-tegra/mesa grate-tmp)
       */
      if ((template->bind & PIPE_BIND_SAMPLER_VIEW) &&
          !(template->bind & (PIPE_BIND_SCANOUT | PIPE_BIND_DISPLAY_TARGET)))
         height = util_next_power_of_two(height);

      /*
       * No DRM_TEGRA_GEM_CREATE_BOTTOM_UP: it makes the display controller
       * scan the buffer out bottom up, but the frontend already flips window
       * surfaces in the viewport, so with it a KMS client (mpv, RetroArch)
       * was upside down on the panel. EXPERIMENT: GRATE_BOTTOM_UP=1.
       */
      if (getenv("GRATE_BOTTOM_UP"))
         flags = DRM_TEGRA_GEM_CREATE_BOTTOM_UP;
   }

   if (template->target != PIPE_BUFFER) {
      /* pick pixel-format */
      int format = grate_pixel_format(template->format);
      assert(format >= 0);
      resource->format = format;
   }

   /*
    * Every mip level gets storage of its own: sized for level 0 only, the
    * uploads of the other levels would overwrite level 0 and a sampler
    * reading them would fault (from libre-tegra/mesa grate-tmp).
    */
   unsigned levels = MIN2(template->last_level + 1, GRATE_MAX_MIP_LEVELS);
   unsigned lw = template->width0, lh = height;
   unsigned blocksize = util_format_get_blocksize(template->format);

   size = 0;
   for (unsigned l = 0; l < levels; ++l) {
      unsigned lpitch = lw * blocksize;

      if (l == 0)
         lpitch = resource->pitch;
      else
         lpitch = align(lpitch, util_is_power_of_two_nonzero(lw) ? 16 : 64);

      resource->level_offset[l] = size;
      resource->level_pitch[l] = lpitch;
      size += lpitch * MAX2(lh, 1u);

      lw = MAX2(lw >> 1, 1u);
      lh = MAX2(lh >> 1, 1u);
   }

   resource->bo = grate_bo_alloc(screen->drm, size, flags);
   if (!resource->bo) {
      fprintf(stderr, "grate_bo_alloc() failed\n");
      return NULL;
   }

   return &resource->b;
}

/* Only linear buffers exist: take any list that allows them (GBM surfaces,
 * DRI3) */
static struct pipe_resource *
grate_screen_resource_create_with_modifiers(struct pipe_screen *pscreen,
                                            const struct pipe_resource *template,
                                            const uint64_t *modifiers,
                                            int count)
{
   bool linear = count == 0;

   for (int i = 0; i < count; i++)
      linear |= modifiers[i] == DRM_FORMAT_MOD_LINEAR ||
                modifiers[i] == DRM_FORMAT_MOD_INVALID;
   if (!linear)
      return NULL;

   return grate_screen_resource_create(pscreen, template);
}

static struct pipe_resource *
grate_screen_resource_from_handle(struct pipe_screen *pscreen,
                                  const struct pipe_resource *template,
                                  struct winsys_handle *handle,
                                  unsigned usage)
{
   struct grate_screen *screen = grate_screen(pscreen);
   struct grate_resource *resource;
   int format;

   resource = CALLOC_STRUCT(grate_resource);
   if (!resource)
      return NULL;

   resource->b = *template;

   pipe_reference_init(&resource->b.reference, 1);
   resource->b.screen = pscreen;

   switch (handle->type) {
   case WINSYS_HANDLE_TYPE_FD:
      resource->bo = grate_bo_import(screen->drm, handle->handle);
      if (!resource->bo) {
         fprintf(stderr, "grate_bo_import() failed\n");
         goto fail;
      }
      break;
   default:
      UNREACHABLE("invalid winsys handle type");
   }
   if (!resource->bo || handle->offset != 0)
      goto fail;

   resource->pitch = handle->stride;

   format = grate_pixel_format(template->format);
   assert(format >= 0);
   resource->format = format;

   return &resource->b;
   
fail:
   FREE(resource);
   return NULL;
}

void
grate_screen_resource_init(struct pipe_screen *pscreen)
{
   grate_trace();
   pscreen->resource_create = grate_screen_resource_create;
   pscreen->resource_from_handle = grate_screen_resource_from_handle;
   pscreen->resource_get_handle = grate_resource_get_handle;
   pscreen->resource_get_param = grate_resource_get_param;
   pscreen->resource_create_with_modifiers = grate_screen_resource_create_with_modifiers;
   pscreen->resource_destroy = grate_resource_destroy;
}

static void
grate_resource_copy_region(struct pipe_context *pcontext,
                           struct pipe_resource *dst,
                           unsigned int dst_level,
                           unsigned int dstx, unsigned dsty,
                           unsigned int dstz,
                           struct pipe_resource *src,
                           unsigned int src_level,
                           const struct pipe_box *box)
{
   /* through the CPU: mapping waits for the GPU, so it is correct if slow
    * (from libre-tegra/mesa grate-tmp) */
   util_resource_copy_region(pcontext, dst, dst_level, dstx, dsty, dstz,
                             src, src_level, box);
}

static void
grate_blit(struct pipe_context *pcontext, const struct pipe_blit_info *info)
{
   int err;
   uint32_t value;
   uint32_t *ptr;
   struct grate_context *context = grate_context(pcontext);
   struct grate_channel *gr2d = context->gr2d;
   struct grate_resource *dst, *src;

   dst = grate_resource(info->dst.resource);
   src = grate_resource(info->src.resource);

   err = grate_stream_begin(&gr2d->stream, &ptr);
   if (err < 0) {
      fprintf(stderr, "grate_stream_begin() failed: %d\n", err);
      return;
   }

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x009, 0x9));
   GRATE_PUSHBUF_WORD(ptr, 0x0000003a);            /* 0x009 - trigger */
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);            /* 0x00c - cmdsel */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x01e, 0x7));
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);            /* 0x01e - controlsecond */
   /*
    * [20:20] source color depth (0: mono, 1: same)
    * [17:16] destination color depth (0: 8 bpp, 1: 16 bpp, 2: 32 bpp)
    */

   value = 1 << 20;
   switch (util_format_get_blocksize(dst->b.format)) {
   case 1:
      value |= 0 << 16;
      break;
   case 2:
      value |= 1 << 16;
      break;
   case 4:
      value |= 2 << 16;
      break;
   default:
      assert(0);
   }

   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x01f - controlmain */
   GRATE_PUSHBUF_WORD(ptr, 0x000000cc);            /* 0x020 - ropfade */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x046, 1));

   /*
    * [20:20] destination write tile mode (0: linear, 1: tiled)
    * [ 0: 0] tile mode Y/RGB (0: linear, 1: tiled)
    */
   value = (dst->tiled << 20) | src->tiled;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x046 - tilemode */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x02b, 0xe149));
   grate_stream_push_reloc(&gr2d->stream, &ptr, dst->bo, 0);      /* 0x02b - dstba */

   GRATE_PUSHBUF_WORD(ptr, dst->pitch);            /* 0x02e - dstst */

   grate_stream_push_reloc(&gr2d->stream, &ptr, src->bo, 0);      /* 0x031 - srcba */

   GRATE_PUSHBUF_WORD(ptr, src->pitch);            /* 0x033 - srcst */

   value = info->dst.box.height << 16 | info->dst.box.width;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x038 - dstsize */

   value = info->src.box.y << 16 | info->src.box.x;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x039 - srcps */

   value = info->dst.box.y << 16 | info->dst.box.x;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x03a - dstps */

   grate_stream_end(&gr2d->stream, &ptr);

   grate_stream_flush(&gr2d->stream, true);
}

static uint32_t
pack_color(enum pipe_format format, const float *rgba)
{
   union util_color uc;
   util_pack_color(rgba, format, &uc);
   return uc.ui[0];
}

static void
fill(struct grate_stream *stream, uint32_t **ptrp,
           struct grate_resource *dst,
           uint32_t fill_value, int blocksize,
           unsigned dstx, unsigned dsty,
           unsigned width, unsigned height)
{
   uint32_t value;
   uint32_t *ptr = *ptrp;

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x09, 0x09));
   GRATE_PUSHBUF_WORD(ptr, 0x0000003a);           /* 0x009 - trigger */
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);           /* 0x00C - cmdsel */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x1e, 0x07));
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);           /* 0x01e - controlsecond */

   value  = 1 << 6; /* fill mode */
   value |= 1 << 2; /* turbofill */
   switch (blocksize) {
   case 1:
      value |= 0 << 16;
      fill_value = (fill_value & 0xff) * 0x01010101u;
      break;
   case 2:
      value |= 1 << 16;
      fill_value = (fill_value & 0xffff) * 0x00010001u;
      break;
   case 4:
      value |= 2 << 16;
      break;
   default:
      UNREACHABLE("invalid blocksize");
   }
   /*
    * srcfgc is a 32 bit pattern whatever the pixel size, so a narrower fill
    * has to be replicated across the word: a bare 16 bit value left every
    * second pixel zero, e.g. a depth buffer cleared to 1.0 came out as
    * alternating 1.0 and 0.0 (from libre-tegra/mesa grate-tmp).
    */
   GRATE_PUSHBUF_WORD(ptr, value);           /* 0x01f - controlmain */

   GRATE_PUSHBUF_WORD(ptr, 0x000000cc);      /* 0x020 - ropfade */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x2b, 0x09));
   grate_stream_push_reloc(stream, &ptr, dst->bo, 0);/* 0x02b - dstba */
   GRATE_PUSHBUF_WORD(ptr, dst->pitch);      /* 0x02e - dstst */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x35, 1));

   GRATE_PUSHBUF_WORD(ptr, fill_value);           /* 0x035 - srcfgc */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x46, 1));
   GRATE_PUSHBUF_WORD(ptr, dst->tiled << 20);     /* 0x046 - tilemode */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x38, 0x05));
   GRATE_PUSHBUF_WORD(ptr, height << 16 | width); /* 0x038 - dstsize */
   GRATE_PUSHBUF_WORD(ptr, dsty << 16 | dstx);    /* 0x03a - dstps */
   
   *ptrp = ptr;
}

static void
grate_clear(struct pipe_context *pcontext, unsigned int buffers,
            uint32_t color_clear_mask, uint8_t stencil_clear_mask,
            const struct pipe_scissor_state *scissor_state,
            const union pipe_color_union *color, double depth,
            unsigned int stencil)
{
   (void) color_clear_mask;
   (void) stencil_clear_mask;
   struct grate_context *context = grate_context(pcontext);
   struct grate_stream *stream = &context->gr2d->stream;
   uint32_t *ptr;
   int err;
   struct pipe_framebuffer_state *fb;

   fb = &context->framebuffer.base;

   err = grate_stream_begin(stream, &ptr);
   if (err < 0) {
      grate_loge("grate_stream_begin() failed: %d\n", err);
      return;
   }

   if (buffers & PIPE_CLEAR_COLOR) {
      int i;
      for (i = 0; i < fb->nr_cbufs; ++i) {
         struct pipe_surface *dst = &fb->cbufs[i];
         fill(stream, &ptr, 
                  grate_resource(dst->texture),
                  pack_color(dst->format, color->f),
                  util_format_get_blocksize(dst->format),
                  0, 0, fb->width, fb->height);
      }
   }

   if (buffers & PIPE_CLEAR_DEPTH || buffers & PIPE_CLEAR_STENCIL) {
      if (fb->zsbuf.texture) {
         /* TODO: handle the case where both are not set! */
         fill(stream, &ptr, 
                  grate_resource(fb->zsbuf.texture),
                  util_pack_z_stencil(fb->zsbuf.format, depth, stencil),
                  util_format_get_blocksize(fb->zsbuf.format),
                  0, 0, fb->width, fb->height);
      }
   }
   
   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, true);
}

static void
grate_clear_render_target(struct pipe_context *pipe,
                          struct pipe_surface *dst,
                          const union pipe_color_union *color,
                          unsigned dstx, unsigned dsty,
                          unsigned width, unsigned height,
                          bool render_condition_enabled)
{
   struct grate_context *context = grate_context(pipe);
   struct grate_stream *stream = &context->gr2d->stream;
   uint32_t *ptr;
   int err;
   assert(!render_condition_enabled);
   
   err = grate_stream_begin(stream, &ptr);
   if (err < 0) {
      grate_loge("grate_stream_begin() failed: %d\n", err);
      return;
   }
   
   fill(&context->gr2d->stream, &ptr,
        grate_resource(dst->texture),
        pack_color(dst->format, color->f),
        util_format_get_blocksize(dst->format),
        dstx, dsty, width, height);
   
   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, true);
}

static void
grate_clear_depth_stencil(struct pipe_context *pipe,
                          struct pipe_surface *dst,
                          unsigned clear_flags,
                          double depth,
                          unsigned stencil,
                          unsigned dstx, unsigned dsty,
                          unsigned width, unsigned height,
                          bool render_condition_enabled)
{
   struct grate_context *context = grate_context(pipe);
   struct grate_stream *stream = &context->gr2d->stream;
   uint32_t *ptr;
   int err;
   assert(!render_condition_enabled);
   
   err = grate_stream_begin(stream, &ptr);
   if (err < 0) {
      grate_loge("grate_stream_begin() failed: %d\n", err);
      return;
   }
   
   fill(&context->gr2d->stream, &ptr,
        grate_resource(dst->texture),
        util_pack_z_stencil(dst->format, depth, stencil),
        util_format_get_blocksize(dst->format),
        dstx, dsty, width, height);
   
   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, true);
}

static void
grate_flush_resource(struct pipe_context *ctx, struct pipe_resource *resource)
{
   /*
    * Called when a resource is about to be handed to the window system - for
    * weston, the buffer it is about to scan out. Nothing else waits for the
    * GPU on that path, so without this the compositor flips a frame that is
    * still being drawn, and the display shows part of the previous one.
    */
   grate_context_flush_streams(grate_context(ctx));
}

void
grate_context_resource_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->buffer_map = grate_resource_transfer_map;
   pcontext->texture_map = grate_resource_transfer_map;
   pcontext->transfer_flush_region = grate_resource_transfer_flush_region;
   pcontext->buffer_unmap = grate_resource_transfer_unmap;
   pcontext->texture_unmap = grate_resource_transfer_unmap;
   pcontext->buffer_subdata = u_default_buffer_subdata;
   pcontext->texture_subdata = u_default_texture_subdata;

   pcontext->resource_copy_region = grate_resource_copy_region;
   pcontext->blit = grate_blit;
   pcontext->clear = grate_clear;
   pcontext->clear_buffer = u_default_clear_buffer;
   pcontext->flush_resource = grate_flush_resource;
   pcontext->resource_release = u_default_resource_release;
   pcontext->clear_render_target = grate_clear_render_target;
   pcontext->clear_depth_stencil = grate_clear_depth_stencil;
}
