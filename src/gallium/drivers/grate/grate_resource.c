#include <math.h>
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

   if (resource->stencil)
      grate_resource_destroy(pscreen, &resource->stencil->b);
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

   /* the two planes have no packed CPU view; GLES 2 does not read depth */
   if (resource->stencil) {
      grate_loge("no CPU access to %s\n", util_format_short_name(presource->format));
      return NULL;
   }

   /*
    * Let the jobs that used the BO land before handing out a CPU pointer:
    * a readback otherwise races the GPU still writing the render target, and
    * an upload overwrites a texture a draw is still sampling.
    */
   if (!(usage & PIPE_MAP_UNSYNCHRONIZED))
      grate_bo_wait(resource->bo);

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
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:   /* the depth plane, see stencil */
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

/* Bytes per pixel of the surface the hardware uses */
unsigned
grate_resource_blocksize(enum pipe_format format)
{
   return format == PIPE_FORMAT_Z24_UNORM_S8_UINT ? 2 : util_format_get_blocksize(format);
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

   resource->pitch = template->width0 * grate_resource_blocksize(template->format);
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
   unsigned blocksize = grate_resource_blocksize(template->format);

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
      FREE(resource);
      return NULL;
   }

   if (template->format == PIPE_FORMAT_Z24_UNORM_S8_UINT) {
      struct pipe_resource plane = *template;
      plane.format = PIPE_FORMAT_S8_UINT;
      struct pipe_resource *s = grate_screen_resource_create(pscreen, &plane);
      if (!s) {
         grate_resource_destroy(pscreen, &resource->b);
         return NULL;
      }
      resource->stencil = grate_resource(s);
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

/* Colour of the source box at (x, y) of it, 0..1 across (bilinear or
 * nearest), from the mapped rows */
static void
blit_sample(const struct pipe_blit_info *info, const uint8_t *map, unsigned stride,
            float x, float y, float *rgba)
{
   enum pipe_format format = info->src.format;
   unsigned bs = util_format_get_blocksize(format);
   int w = abs(info->src.box.width), h = abs(info->src.box.height);
   float fx = x * w - 0.5f, fy = y * h - 0.5f;

   if (info->filter != PIPE_TEX_FILTER_LINEAR) {
      int ix = CLAMP((int)(x * w), 0, w - 1), iy = CLAMP((int)(y * h), 0, h - 1);
      util_format_unpack_rgba(format, rgba, map + iy * stride + ix * bs, 1);
      return;
   }

   int x0 = CLAMP((int)floorf(fx), 0, w - 1), y0 = CLAMP((int)floorf(fy), 0, h - 1);
   int x1 = MIN2(x0 + 1, w - 1), y1 = MIN2(y0 + 1, h - 1);
   float ax = CLAMP(fx - floorf(fx), 0.0f, 1.0f), ay = CLAMP(fy - floorf(fy), 0.0f, 1.0f);
   float t[4][4];
   util_format_unpack_rgba(format, t[0], map + y0 * stride + x0 * bs, 1);
   util_format_unpack_rgba(format, t[1], map + y0 * stride + x1 * bs, 1);
   util_format_unpack_rgba(format, t[2], map + y1 * stride + x0 * bs, 1);
   util_format_unpack_rgba(format, t[3], map + y1 * stride + x1 * bs, 1);
   for (int c = 0; c < 4; c++)
      rgba[c] = (t[0][c] * (1 - ax) + t[1][c] * ax) * (1 - ay) +
                (t[2][c] * (1 - ax) + t[3][c] * ax) * ay;
}

/*
 * Blits the 2D engine cannot do (it copies bytes, same size, level 0):
 * scaling (mipmap generation), flips, other levels, format conversion. On
 * the CPU, colour only.
 */
static void
grate_blit_cpu(struct pipe_context *pcontext, const struct pipe_blit_info *info)
{
   struct pipe_transfer *st, *dt;
   const struct pipe_box *sb = &info->src.box, *db = &info->dst.box;

   if (!(info->mask & PIPE_MASK_RGBA) || util_format_is_depth_or_stencil(info->dst.format) ||
       util_format_is_compressed(info->src.format) || util_format_is_compressed(info->dst.format)) {
      grate_loge("blit %s -> %s unsupported\n", util_format_short_name(info->src.format),
                 util_format_short_name(info->dst.format));
      return;
   }

   struct pipe_box sbox = *sb, dbox = *db;
   /* map the boxes the right way up, flip in the sampling */
   if (sbox.width < 0) { sbox.x += sbox.width; sbox.width = -sbox.width; }
   if (sbox.height < 0) { sbox.y += sbox.height; sbox.height = -sbox.height; }
   if (dbox.width < 0) { dbox.x += dbox.width; dbox.width = -dbox.width; }
   if (dbox.height < 0) { dbox.y += dbox.height; dbox.height = -dbox.height; }
   sbox.depth = dbox.depth = 1;

   const uint8_t *smap = pcontext->texture_map(pcontext, info->src.resource, info->src.level,
                                               PIPE_MAP_READ, &sbox, &st);
   uint8_t *dmap = pcontext->texture_map(pcontext, info->dst.resource, info->dst.level,
                                         PIPE_MAP_WRITE, &dbox, &dt);
   if (smap && dmap) {
      unsigned dbs = util_format_get_blocksize(info->dst.format);
      bool sflip_x = sb->width < 0, sflip_y = sb->height < 0;
      bool dflip_x = db->width < 0, dflip_y = db->height < 0;
      const struct pipe_scissor_state *sc = &info->scissor;
      for (int y = 0; y < dbox.height; y++) {
         for (int x = 0; x < dbox.width; x++) {
            if (info->scissor_enable &&
                (dbox.x + x < sc->minx || dbox.x + x >= sc->maxx ||
                 dbox.y + y < sc->miny || dbox.y + y >= sc->maxy))
               continue;
            float u = (x + 0.5f) / dbox.width, v = (y + 0.5f) / dbox.height;
            if (sflip_x != dflip_x)
               u = 1.0f - u;
            if (sflip_y != dflip_y)
               v = 1.0f - v;
            float rgba[4];
            blit_sample(info, smap, st->stride, u, v, rgba);
            util_format_pack_rgba(info->dst.format, dmap + y * dt->stride + x * dbs, rgba, 1);
         }
      }
   }
   if (smap)
      pcontext->texture_unmap(pcontext, st);
   if (dmap)
      pcontext->texture_unmap(pcontext, dt);
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

   /* the 2D engine copies bytes, same size, at level 0 */
   if (info->src.level || info->dst.level || info->src.format != info->dst.format ||
       info->src.box.width != info->dst.box.width ||
       info->src.box.height != info->dst.box.height ||
       info->src.box.width < 0 || info->src.box.height < 0 || info->scissor_enable ||
       util_format_is_compressed(info->src.format)) {
      grate_blit_cpu(pcontext, info);
      return;
   }

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
   grate_stream_push_reloc(&gr2d->stream, &ptr, dst->bo, 0, true); /* 0x02b - dstba */

   GRATE_PUSHBUF_WORD(ptr, dst->pitch);            /* 0x02e - dstst */

   grate_stream_push_reloc(&gr2d->stream, &ptr, src->bo, 0, false); /* 0x031 - srcba */

   GRATE_PUSHBUF_WORD(ptr, src->pitch);            /* 0x033 - srcst */

   value = info->dst.box.height << 16 | info->dst.box.width;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x038 - dstsize */

   value = info->src.box.y << 16 | info->src.box.x;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x039 - srcps */

   value = info->dst.box.y << 16 | info->dst.box.x;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x03a - dstps */

   grate_stream_end(&gr2d->stream, &ptr);

   grate_stream_flush(&gr2d->stream, false);
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
           struct grate_resource *dst, unsigned level,
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
   /* a mip level: where grate_screen_resource_create put it */
   level = MIN2(level, GRATE_MAX_MIP_LEVELS - 1);
   grate_stream_push_reloc(stream, &ptr, dst->bo, dst->level_offset[level], true); /* 0x02b - dstba */
   GRATE_PUSHBUF_WORD(ptr, level ? dst->level_pitch[level] : dst->pitch); /* 0x02e - dstst */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x35, 1));

   GRATE_PUSHBUF_WORD(ptr, fill_value);           /* 0x035 - srcfgc */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x46, 1));
   GRATE_PUSHBUF_WORD(ptr, dst->tiled << 20);     /* 0x046 - tilemode */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x38, 0x05));
   GRATE_PUSHBUF_WORD(ptr, height << 16 | width); /* 0x038 - dstsize */
   GRATE_PUSHBUF_WORD(ptr, dsty << 16 | dstx);    /* 0x03a - dstps */
   
   *ptrp = ptr;
}

/* Clear the depth and/or stencil of a depth/stencil surface, plane by plane */
static void
fill_zs(struct grate_stream *stream, uint32_t **ptrp, struct pipe_surface *dst,
        unsigned flags, double depth, unsigned stencil,
        unsigned x, unsigned y, unsigned width, unsigned height)
{
   struct grate_resource *res = grate_resource(dst->texture);
   const struct util_format_description *desc = util_format_description(dst->format);
   bool clear_depth = (flags & PIPE_CLEAR_DEPTH) && util_format_has_depth(desc);
   bool clear_stencil = (flags & PIPE_CLEAR_STENCIL) && util_format_has_stencil(desc);

   if (res->stencil) {
      if (clear_depth)
         fill(stream, ptrp, res, dst->level, util_pack_z(PIPE_FORMAT_Z16_UNORM, depth), 2,
              x, y, width, height);
      if (clear_stencil)
         fill(stream, ptrp, res->stencil, dst->level, stencil, 1, x, y, width, height);
   } else if (clear_depth || clear_stencil) {
      fill(stream, ptrp, res, dst->level, util_pack_z_stencil(dst->format, depth, stencil),
           util_format_get_blocksize(dst->format), x, y, width, height);
   }
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

   /* a scissored clear (compositors clear the damaged region only) */
   unsigned x = 0, y = 0, w = fb->width, h = fb->height;
   if (scissor_state) {
      unsigned maxx = MIN2(scissor_state->maxx, fb->width);
      unsigned maxy = MIN2(scissor_state->maxy, fb->height);
      x = MIN2(scissor_state->minx, maxx);
      y = MIN2(scissor_state->miny, maxy);
      w = maxx - x;
      h = maxy - y;
      if (!w || !h)
         return;
   }

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
                  grate_resource(dst->texture), dst->level,
                  pack_color(dst->format, color->f),
                  util_format_get_blocksize(dst->format),
                  x, y, w, h);
      }
   }

   if (buffers & PIPE_CLEAR_DEPTH || buffers & PIPE_CLEAR_STENCIL) {
      if (fb->zsbuf.texture)
         fill_zs(stream, &ptr, &fb->zsbuf, buffers, depth, stencil, x, y, w, h);
   }
   
   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, false);
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
        grate_resource(dst->texture), dst->level,
        pack_color(dst->format, color->f),
        util_format_get_blocksize(dst->format),
        dstx, dsty, width, height);
   
   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, false);
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
   
   fill_zs(stream, &ptr, dst, clear_flags, depth, stencil, dstx, dsty, width, height);
   
   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, false);
}

static void
grate_flush_resource(struct pipe_context *ctx, struct pipe_resource *resource)
{
   /*
    * Called when a resource is about to be handed to the window system - for
    * weston, the buffer it is about to scan out. A kernel with implicit sync
    * has the job fences on the BO, and the display waits for them; without
    * it nothing else waits for the GPU on that path, and the compositor would
    * flip a frame that is still being drawn.
    */
   if (grate_context(ctx)->drm->implicit_sync != 1)
      grate_bo_wait(grate_resource(resource)->bo);
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
