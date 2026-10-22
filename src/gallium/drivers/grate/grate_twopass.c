/*
 * Blending in two passes, for fragment programs that only fit without it.
 *
 * Tegra20 blends in the fragment program (the PSEQ loads the destination
 * colour), which costs a register pair, an EXEC and two levels of ALU
 * operations: a long program, e.g. Qt Quick's MultiEffect, may fit opaque
 * but not blended. Such a draw is then done as the drivers of the time did
 * long effects:
 *
 * 1. the draw itself, unblended, into a colour buffer of the framebuffer's
 *    size cleared to transparent black, with the draw's own depth/stencil
 *    buffer, viewport and scissor;
 * 2. a rectangle over the whole framebuffer copying that buffer onto the
 *    real one with the draw's blend state.
 *
 * The pixels the draw does not cover come out (0, 0, 0, 0), which leaves the
 * destination as it was for the blend states where a source of zero adds
 * nothing and keeps the destination's weight at one: src * f + dst * g (or
 * dst * g - src * f) with g one, one minus source alpha or one minus source
 * colour. That covers classic and premultiplied alpha blending; other blend
 * states keep drawing the program that does not fit (magenta). A program
 * that fits blended never comes here.
 */
#include "pipe/p_shader_tokens.h"
#include "pipe/p_state.h"
#include "tgsi/tgsi_text.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

#include "grate_common.h"
#include "grate_context.h"
#include "grate_program.h"
#include "grate_state.h"
#include "grate_twopass.h"

struct grate_twopass {
   struct pipe_resource *target;     /* the first pass' colour */
   struct pipe_sampler_view *view;
   struct pipe_resource *quad;
   void *sampler, *elements, *zsa, *rast, *vs, *fs;
   struct pipe_blend_state opaque;   /* no blending, every channel */
   struct tgsi_token vs_tokens[64], fs_tokens[64];
};

/* the position, and the texture coordinate 0..1 from it */
static const char grate_twopass_vs_text[] =
   "VERT\n"
   "DCL IN[0]\n"
   "DCL OUT[0], POSITION\n"
   "DCL OUT[1], GENERIC[0]\n"
   "IMM[0] FLT32 { 0.5, 0.0, 0.0, 0.0 }\n"
   "  0: MOV OUT[0], IN[0]\n"
   "  1: MAD OUT[1], IN[0], IMM[0].xxxx, IMM[0].xxxx\n"
   "  2: END\n";

static const char grate_twopass_fs_text[] =
   "FRAG\n"
   "DCL IN[0], GENERIC[0], PERSPECTIVE\n"
   "DCL OUT[0], COLOR\n"
   "DCL SAMP[0]\n"
   "DCL SVIEW[0], 2D, FLOAT\n"
   "DCL TEMP[0]\n"
   "  0: TEX TEMP[0], IN[0], SAMP[0], 2D\n"
   "  1: MOV OUT[0], TEMP[0]\n"
   "  2: END\n";

/* Whether a source of zero leaves the destination as it is */
static bool
grate_twopass_keeps_dst(unsigned func, unsigned dst_factor)
{
   return (func == PIPE_BLEND_ADD || func == PIPE_BLEND_REVERSE_SUBTRACT) &&
          (dst_factor == PIPE_BLENDFACTOR_ONE ||
           dst_factor == PIPE_BLENDFACTOR_INV_SRC_ALPHA ||
           dst_factor == PIPE_BLENDFACTOR_INV_SRC_COLOR);
}

static struct grate_twopass *
grate_twopass_init(struct grate_context *context)
{
   struct pipe_context *pipe = &context->base;
   struct grate_twopass *tp = CALLOC_STRUCT(grate_twopass);
   if (!tp)
      return NULL;

   tp->opaque.rt[0].colormask = PIPE_MASK_RGBA;

   if (!tgsi_text_translate(grate_twopass_vs_text, tp->vs_tokens, ARRAY_SIZE(tp->vs_tokens)) ||
       !tgsi_text_translate(grate_twopass_fs_text, tp->fs_tokens, ARRAY_SIZE(tp->fs_tokens))) {
      FREE(tp);
      return NULL;
   }
   struct pipe_shader_state vs = { .type = PIPE_SHADER_IR_TGSI, .tokens = tp->vs_tokens };
   struct pipe_shader_state fs = { .type = PIPE_SHADER_IR_TGSI, .tokens = tp->fs_tokens };
   tp->vs = pipe->create_vs_state(pipe, &vs);
   tp->fs = pipe->create_fs_state(pipe, &fs);

   struct pipe_vertex_element element = {
      .src_format = PIPE_FORMAT_R32G32B32A32_FLOAT,
      .src_stride = 4 * sizeof(float),
   };
   tp->elements = pipe->create_vertex_elements_state(pipe, 1, &element);

   static const float quad[4][4] = {
      { -1, -1, 0, 1 }, { 1, -1, 0, 1 }, { -1, 1, 0, 1 }, { 1, 1, 0, 1 },
   };
   tp->quad = pipe_buffer_create(pipe->screen, PIPE_BIND_VERTEX_BUFFER,
                                 PIPE_USAGE_IMMUTABLE, sizeof(quad));
   if (tp->quad)
      pipe_buffer_write(pipe, tp->quad, 0, sizeof(quad), quad);

   struct pipe_depth_stencil_alpha_state zsa = { 0 };
   tp->zsa = pipe->create_depth_stencil_alpha_state(pipe, &zsa);

   struct pipe_rasterizer_state rast = {
      .cull_face = PIPE_FACE_NONE,
      .half_pixel_center = true,
      .depth_clip_near = true,
      .depth_clip_far = true,
   };
   tp->rast = pipe->create_rasterizer_state(pipe, &rast);

   struct pipe_sampler_state sampler = {
      .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .min_img_filter = PIPE_TEX_FILTER_NEAREST,
      .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
      .min_mip_filter = PIPE_TEX_MIPFILTER_NONE,
   };
   tp->sampler = pipe->create_sampler_state(pipe, &sampler);

   if (!tp->vs || !tp->fs || !tp->elements || !tp->quad || !tp->zsa ||
       !tp->rast || !tp->sampler) {
      grate_twopass_destroy(context, tp);
      return NULL;
   }
   return tp;
}

void
grate_twopass_destroy(struct grate_context *context, struct grate_twopass *tp)
{
   struct pipe_context *pipe = &context->base;

   if (!tp)
      return;
   pipe_sampler_view_reference(&tp->view, NULL);
   pipe_resource_reference(&tp->target, NULL);
   pipe_resource_reference(&tp->quad, NULL);
   if (tp->sampler)
      pipe->delete_sampler_state(pipe, tp->sampler);
   if (tp->elements)
      pipe->delete_vertex_elements_state(pipe, tp->elements);
   if (tp->zsa)
      pipe->delete_depth_stencil_alpha_state(pipe, tp->zsa);
   if (tp->rast)
      pipe->delete_rasterizer_state(pipe, tp->rast);
   if (tp->vs)
      pipe->delete_vs_state(pipe, tp->vs);
   if (tp->fs)
      pipe->delete_fs_state(pipe, tp->fs);
   FREE(tp);
}

/* The first pass' colour buffer, of the framebuffer's size */
static bool
grate_twopass_target(struct grate_context *context, struct grate_twopass *tp,
                     unsigned width, unsigned height)
{
   struct pipe_context *pipe = &context->base;

   if (tp->target && tp->target->width0 == width && tp->target->height0 == height)
      return true;

   pipe_sampler_view_reference(&tp->view, NULL);
   pipe_resource_reference(&tp->target, NULL);

   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_B8G8R8A8_UNORM,
      .width0 = width,
      .height0 = height,
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   tp->target = pipe->screen->resource_create(pipe->screen, &templ);
   if (!tp->target)
      return false;

   struct pipe_sampler_view view = {
      .format = PIPE_FORMAT_B8G8R8A8_UNORM,
      .target = PIPE_TEXTURE_2D,
      .swizzle_r = PIPE_SWIZZLE_X,
      .swizzle_g = PIPE_SWIZZLE_Y,
      .swizzle_b = PIPE_SWIZZLE_Z,
      .swizzle_a = PIPE_SWIZZLE_W,
   };
   tp->view = pipe->create_sampler_view(pipe, tp->target, &view);
   return tp->view != NULL;
}

bool
grate_twopass_draw(struct grate_context *context, const struct pipe_draw_info *info,
                   unsigned drawid_offset, const struct pipe_draw_start_count_bias *draws,
                   unsigned num_draws)
{
   struct pipe_context *pipe = &context->base;
   const struct pipe_blend_state *blend = context->blend;
   const struct pipe_framebuffer_state *fb = &context->framebuffer.base;

   if (!blend || !blend->rt[0].blend_enable || fb->nr_cbufs != 1 ||
       !fb->cbufs[0].texture ||
       !grate_twopass_keeps_dst(blend->rt[0].rgb_func, blend->rt[0].rgb_dst_factor) ||
       !grate_twopass_keeps_dst(blend->rt[0].alpha_func, blend->rt[0].alpha_dst_factor))
      return false;

   if (!context->twopass)
      context->twopass = grate_twopass_init(context);
   struct grate_twopass *tp = context->twopass;
   if (!tp || !grate_twopass_target(context, tp, fb->width, fb->height))
      return false;

   /* only when the program fits without blending */
   context->blend = &tp->opaque;
   grate_fs_update(context);
   context->blend = blend;
   if (context->fshader->info.failed) {
      grate_fs_update(context);
      return false;
   }

   /* 1: the draw, unblended, into the cleared buffer */
   struct pipe_framebuffer_state saved = { 0 };
   util_copy_framebuffer_state(&saved, fb);
   struct pipe_framebuffer_state first = saved;
   first.cbufs[0] = (struct pipe_surface) {
      .format = PIPE_FORMAT_B8G8R8A8_UNORM,
      .texture = tp->target,
   };
   pipe->set_framebuffer_state(pipe, &first);
   static const union pipe_color_union transparent = { .f = { 0, 0, 0, 0 } };
   pipe->clear_render_target(pipe, &first.cbufs[0], &transparent, 0, 0,
                             fb->width, fb->height, false);
   context->blend = &tp->opaque;
   pipe->draw_vbo(pipe, info, drawid_offset, NULL, draws, num_draws);
   context->blend = blend;
   pipe->set_framebuffer_state(pipe, &saved);
   util_unreference_framebuffer_state(&saved);

   /* 2: that buffer over the framebuffer, blended */
   struct grate_vertex_shader_state *vshader = context->vshader;
   struct grate_fragment_shader_state *fshader = context->fshader;
   struct grate_vertex_state *elements = context->vs;
   struct grate_vertexbuf_state vbs = context->vbs;
   struct grate_zsa_state *zsa = context->zsa;
   struct grate_rasterizer_state *rast = context->rast;
   struct pipe_sampler_view *view0 = context->sampler_views[0];
   struct pipe_sampler_state *sampler0 = context->samplers[0];
   unsigned num_views = context->num_sampler_views;
   uint32_t viewport[ARRAY_SIZE(context->viewport)], guardband[ARRAY_SIZE(context->guardband)];
   bool y_invert = context->y_invert;
   memcpy(viewport, context->viewport, sizeof(viewport));
   memcpy(guardband, context->guardband, sizeof(guardband));

   context->vshader = tp->vs;
   context->fshader = tp->fs;
   context->vs = tp->elements;
   memset(&context->vbs, 0, sizeof(context->vbs));
   context->vbs.vb[0].buffer.resource = tp->quad;
   context->vbs.count = 1;
   context->vbs.enabled = 1;
   context->zsa = tp->zsa;
   context->rast = tp->rast;
   context->sampler_views[0] = tp->view;
   context->samplers[0] = tp->sampler;
   context->num_sampler_views = MAX2(num_views, 1);
   struct pipe_viewport_state vp = {
      .scale = { fb->width / 2.0f, fb->height / 2.0f, 0.5f },
      .translate = { fb->width / 2.0f, fb->height / 2.0f, 0.5f },
      .swizzle_x = PIPE_VIEWPORT_SWIZZLE_POSITIVE_X,
      .swizzle_y = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Y,
      .swizzle_z = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Z,
      .swizzle_w = PIPE_VIEWPORT_SWIZZLE_POSITIVE_W,
   };
   pipe->set_viewport_states(pipe, 0, 1, &vp);

   struct pipe_draw_info quad = { .mode = MESA_PRIM_TRIANGLE_STRIP, .instance_count = 1 };
   struct pipe_draw_start_count_bias range = { .start = 0, .count = 4 };
   pipe->draw_vbo(pipe, &quad, 0, NULL, &range, 1);

   context->vshader = vshader;
   context->fshader = fshader;
   context->vs = elements;
   context->vbs = vbs;
   context->zsa = zsa;
   context->rast = rast;
   context->sampler_views[0] = view0;
   context->samplers[0] = sampler0;
   context->num_sampler_views = num_views;
   memcpy(context->viewport, viewport, sizeof(viewport));
   memcpy(context->guardband, guardband, sizeof(guardband));
   context->y_invert = y_invert;
   return true;
}
