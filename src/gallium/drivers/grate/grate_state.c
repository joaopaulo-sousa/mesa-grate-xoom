#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "pipe/p_shader_tokens.h"
#include "util/format/u_format.h"
#include "util/u_bitcast.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_framebuffer.h"

#include "grate_common.h"
#include "grate_bo.h"
#include "grate_context.h"
#include "grate_program.h"
#include "grate_resource.h"
#include "grate_state.h"

#include "host1x01_hardware.h"

static void
grate_set_sample_mask(struct pipe_context *pcontext,
                      unsigned int sample_mask)
{
   grate_unimplemented();
}

static void
grate_set_constant_buffer(struct pipe_context *pcontext, enum mesa_shader_stage shader,
                          uint index, const struct pipe_constant_buffer *buffer)
{
   struct grate_context *context = grate_context(pcontext);

   assert(index == 0);
   assert(!buffer || buffer->user_buffer);

   util_copy_constant_buffer(&context->constant_buffer[shader], buffer);
}

static void
grate_set_render_target(struct grate_context *context, unsigned slot,
                        const struct grate_resource *res, unsigned level,
                        unsigned layer)
{
   uint32_t rt_params;

   /* a mip level: where grate_screen_resource_create put it */
   level = MIN2(level, GRATE_MAX_MIP_LEVELS - 1);
   uint32_t pitch = level ? res->level_pitch[level] : res->pitch;

   rt_params  = TGR3D_GLOBAL_SURFDESC_SURF_FORMAT(res->format);
   rt_params |= TGR3D_GLOBAL_SURFDESC_ARRAY_STRIDE(pitch);
   rt_params |= TGR3D_GLOBAL_SURFDESC_STRUCTURE(res->tiled);

   context->framebuffer.rt_params[slot] = rt_params;
   context->framebuffer.rt_bos[slot] = res->bo;
   context->framebuffer.rt_offsets[slot] = grate_resource_offset(res, level, layer);
   context->framebuffer.rt_present |= 1 << slot;
}

static void
grate_set_framebuffer_state(struct pipe_context *pcontext,
                            const struct pipe_framebuffer_state *framebuffer)
{
   struct grate_context *context = grate_context(pcontext);
   struct pipe_framebuffer_state *cso = &context->framebuffer.base;
   context->framebuffer.rt_present = 0;

   util_copy_framebuffer_state(cso, framebuffer);

   assert(framebuffer->nr_cbufs <= 1);
   if (framebuffer->nr_cbufs && framebuffer->cbufs[0].texture)
      grate_set_render_target(context, GRATE_RT_COLOR,
                              grate_resource(framebuffer->cbufs[0].texture),
                              framebuffer->cbufs[0].level, framebuffer->cbufs[0].first_layer);

   if (framebuffer->zsbuf.texture) {
      struct grate_resource *zs = grate_resource(framebuffer->zsbuf.texture);
      const struct util_format_description *desc =
         util_format_description(framebuffer->zsbuf.format);

      unsigned level = framebuffer->zsbuf.level, layer = framebuffer->zsbuf.first_layer;
      if (util_format_has_depth(desc))
         grate_set_render_target(context, GRATE_RT_DEPTH, zs, level, layer);
      if (util_format_has_stencil(desc))
         grate_set_render_target(context, GRATE_RT_STENCIL, zs->stencil ? zs->stencil : zs,
                                 level, layer);
   }

   /* prepare the scissor-registers for the non-scissor case */
}

static void
grate_set_polygon_stipple(struct pipe_context *pcontext,
                          const struct pipe_poly_stipple *stipple)
{
   grate_unimplemented();
}

static void
grate_set_scissor_states(struct pipe_context *pcontext,
                         unsigned start_slot,
                         unsigned num_scissors,
                         const struct pipe_scissor_state * scissors)
{
   assert(start_slot == 0 && num_scissors == 1);
   grate_context(pcontext)->scissor = scissors[0];
}

static void
grate_set_viewport_states(struct pipe_context *pcontext,
                          unsigned start_slot,
                          unsigned num_viewports,
                          const struct pipe_viewport_state *viewports)
{
   struct grate_context *context = grate_context(pcontext);
   /*
    * Window depth scaled by 1 - 2^-20, so that it stays below 1.0 (the
    * depth plane's last step) whatever the depth range, glDepthRange(0, 0)
    * and reversed ones too. (Upstream took 2^-21 off scale and translate,
    * which only does that for (0, 1): (0, 0) went below 0, clipped away.)
    */
   static const float zscale = 1.0f - 0x1p-20f;
   unsigned int hw_scale;

   if (context->drm->soc_id == DRM_TEGRA_SOC_T114)
      hw_scale = 0xFFFFFF;
   else
      hw_scale = 0xFFFFF;

   assert(num_viewports == 1);
   assert(start_slot == 0);

   context->depth_scale = viewports[0].scale[2];
   context->depth_translate = viewports[0].translate[2];

   context->viewport[0] = host1x_opcode_incr(REG_TGR3D_SU_VIEWPORT_X, 6);
   context->viewport[1] = u_bitcast_f2u(viewports[0].translate[0] * 16.0f);
   context->viewport[2] = u_bitcast_f2u(viewports[0].translate[1] * 16.0f);
   context->viewport[3] = u_bitcast_f2u(viewports[0].translate[2] * zscale);
   context->viewport[4] = u_bitcast_f2u(viewports[0].scale[0] * 16.0f);
   context->viewport[5] = u_bitcast_f2u(viewports[0].scale[1] * 16.0f);
   context->viewport[6] = u_bitcast_f2u(viewports[0].scale[2] * zscale);

   /* the range the depth unit keeps (Z_CLAMP_KILL): from the nearer to the
    * farther, whichever glDepthRange puts first (unscaled: the depth
    * written, z * zscale, stays within it) */
   float z0 = viewports[0].translate[2] - viewports[0].scale[2];
   float z1 = viewports[0].translate[2] + viewports[0].scale[2];
   uint32_t depth_near = MIN2(z0, z1) * hw_scale;
   uint32_t depth_far = MAX2(z0, z1) * hw_scale;
   /* an empty range (glDepthRange(z, z)) keeps nothing after rounding:
    * one step either side */
   if (depth_far - depth_near < 2) {
      depth_near = depth_near ? depth_near - 1 : 0;
      depth_far = MIN2(depth_far + 1, hw_scale);
   }
   context->viewport[7] = host1x_opcode_incr(REG_TGR3D_QR_Z_MIN, 2);
   context->viewport[8] = depth_near;
   context->viewport[9] = depth_far;

   assert(viewports[0].scale[0] >= 0.0f);
   float max_x = fabs(viewports[0].translate[0]);
   float max_y = fabs(viewports[0].translate[1]);
   float scale_x = viewports[0].scale[0];
   float scale_y = fabs(viewports[0].scale[1]);
   context->guardband[0] = host1x_opcode_incr(REG_TGR3D_SU_GUARDBAND_W, 3);
   context->guardband[1] = u_bitcast_f2u((3967 - max_x) / scale_x);
   context->guardband[2] = u_bitcast_f2u((3967 - max_y) / scale_y);
   /* the depth guard band at 1: the clipper clips to the near and far
    * planes itself (measured, dEQP clipping z_clip: 1.0 clips there, other
    * values do not); the depth unit stays off without a depth test */
   context->guardband[3] = u_bitcast_f2u(1.0f);


   /* GL viewports are whole pixels: round the edges */
   float x0 = viewports[0].translate[0] - scale_x, x1 = viewports[0].translate[0] + scale_x;
   float y0 = viewports[0].translate[1] - scale_y, y1 = viewports[0].translate[1] + scale_y;
   context->viewport_rect.minx = MAX2(lroundf(x0), 0);
   context->viewport_rect.maxx = MAX2(lroundf(x1), 0);
   context->viewport_rect.miny = MAX2(lroundf(y0), 0);
   context->viewport_rect.maxy = MAX2(lroundf(y1), 0);
}

static void
grate_set_vertex_buffers(struct pipe_context *pcontext,
                         unsigned count,
                         const struct pipe_vertex_buffer *buffer)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_vertexbuf_state *vbs = &context->vbs;

   util_set_vertex_buffers_mask(vbs->vb, &vbs->enabled, buffer, count);
   vbs->count = util_last_bit(vbs->enabled);
}


static void
grate_set_sampler_views(struct pipe_context *pctx, mesa_shader_stage shader,
                        unsigned start_slot, unsigned num_views,
                        unsigned unbind_num_trailing_slots,
                        struct pipe_sampler_view **views)
{
   struct grate_context *context = grate_context(pctx);
   unsigned i, end = start_slot + num_views + unbind_num_trailing_slots;

   if (shader != MESA_SHADER_FRAGMENT)
      return;

   for (i = start_slot; i < end && i < ARRAY_SIZE(context->sampler_views); i++) {
      struct pipe_sampler_view *view =
         views && i < start_slot + num_views ? views[i - start_slot] : NULL;
      pipe_sampler_view_reference(&context->sampler_views[i], view);
   }

   context->num_sampler_views = 0;
   for (i = 0; i < ARRAY_SIZE(context->sampler_views); i++)
      if (context->sampler_views[i])
         context->num_sampler_views = i + 1;
}

static void
grate_set_blend_color(struct pipe_context *pctx,
                      const struct pipe_blend_color *blend_color)
{
   grate_context(pctx)->blend_color = *blend_color;
}

static void
grate_set_stencil_ref(struct pipe_context *pctx,
                      const struct pipe_stencil_ref stencil_ref)
{
   grate_context(pctx)->stencil_ref = stencil_ref;
}

static void
grate_fp_state_bind(struct pipe_context *pctx, void *hwcso)
{
   grate_unimplemented();
}

static void
grate_vp_state_bind(struct pipe_context *pctx, void *hwcso)
{
   grate_unimplemented();
}

void
grate_context_state_init(struct pipe_context *pcontext)
{
   pcontext->set_blend_color = grate_set_blend_color;
   pcontext->set_stencil_ref = grate_set_stencil_ref;
   pcontext->bind_fs_state = grate_fp_state_bind;
   pcontext->bind_vs_state = grate_vp_state_bind;
   pcontext->set_sample_mask = grate_set_sample_mask;
   pcontext->set_constant_buffer = grate_set_constant_buffer;
   pcontext->set_framebuffer_state = grate_set_framebuffer_state;
   pcontext->set_polygon_stipple = grate_set_polygon_stipple;
   pcontext->set_scissor_states = grate_set_scissor_states;
   pcontext->set_viewport_states = grate_set_viewport_states;
   pcontext->set_sampler_views = grate_set_sampler_views;
   pcontext->set_vertex_buffers = grate_set_vertex_buffers;
}

static void *
grate_create_blend_state(struct pipe_context *pcontext,
                         const struct pipe_blend_state *template)
{
   struct pipe_blend_state *so = CALLOC_STRUCT(pipe_blend_state);
   if (!so)
      return NULL;

   *so = *template;

   return so;
}

static void
grate_bind_blend_state(struct pipe_context *pcontext, void *so)
{
   /* applied by the fragment program (grate_fs_update): Tegra20 has no
    * blend unit */
   grate_context(pcontext)->blend = so;
}

static void
grate_delete_blend_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_blend_init(struct pipe_context *pcontext)
{
   pcontext->create_blend_state = grate_create_blend_state;
   pcontext->bind_blend_state = grate_bind_blend_state;
   pcontext->delete_blend_state = grate_delete_blend_state;
}

static void *
grate_create_sampler_state(struct pipe_context *pcontext,
            const struct pipe_sampler_state *template)
{
   struct pipe_sampler_state *so = CALLOC_STRUCT(pipe_sampler_state);
   if (!so)
      return NULL;

   *so = *template;

   return so;
}

static void
grate_bind_sampler_states(struct pipe_context *pcontext,
                          enum mesa_shader_stage shader,
                          unsigned start_slot, unsigned num_samplers,
                          void **samplers)
{
   struct grate_context *context = grate_context(pcontext);

   if (shader != MESA_SHADER_FRAGMENT)
      return;

   for (unsigned i = 0; i < num_samplers; i++)
      if (start_slot + i < ARRAY_SIZE(context->samplers))
         context->samplers[start_slot + i] = samplers ? samplers[i] : NULL;
}

static void
grate_delete_sampler_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

static struct pipe_sampler_view *
grate_create_sampler_view(struct pipe_context *pcontext,
                          struct pipe_resource *resource,
                          const struct pipe_sampler_view *template)
{
   struct pipe_sampler_view *so = CALLOC_STRUCT(pipe_sampler_view);
   if (!so)
      return NULL;

   *so = *template;
   so->texture = NULL;
   pipe_resource_reference(&so->texture, resource);
   pipe_reference_init(&so->reference, 1);
   so->context = pcontext;

   return so;
}

static void
grate_sampler_view_destroy(struct pipe_context *pcontext,
                           struct pipe_sampler_view *pview)
{
   pipe_resource_reference(&pview->texture, NULL);
   FREE(pview);
}

void
grate_context_sampler_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_sampler_state = grate_create_sampler_state;
   pcontext->bind_sampler_states = grate_bind_sampler_states;
   pcontext->delete_sampler_state = grate_delete_sampler_state;
   pcontext->create_sampler_view = grate_create_sampler_view;
   pcontext->sampler_view_destroy = grate_sampler_view_destroy;
   pcontext->sampler_view_release = u_default_sampler_view_release;
}

static int
grate_cull_face(int cull_face, bool front_ccw)
{
   switch (cull_face) {
   case PIPE_FACE_NONE:
      return TGR3D_CULL_NONE;

   /* measured: CULL_NEG drops the counter-clockwise triangles (gratetest
    * GRATETEST_CULL, facetest.sh) */
   case PIPE_FACE_FRONT:
      return front_ccw ? TGR3D_CULL_NEG : TGR3D_CULL_POS;

   case PIPE_FACE_BACK:
      return front_ccw ? TGR3D_CULL_POS : TGR3D_CULL_NEG;

   case PIPE_FACE_FRONT_AND_BACK:
      return TGR3D_CULL_BOTH;

   default:
      UNREACHABLE("unknown cull_face");
   }
}

static void *
grate_create_rasterizer_state(struct pipe_context *pcontext,
                              const struct pipe_rasterizer_state *template)
{
   struct grate_rasterizer_state *so = CALLOC_STRUCT(grate_rasterizer_state);
   if (!so)
      return NULL;

   so->base = *template;

   so->draw_params = TGR3D_IDX_SET_PRIM_FLAT_VTX(!template->flatshade_first);

   /*
    * FRONT_FACE says which faces the fragment program's face register, the
    * culling and the FRONT stencil state take as front; it is the same for
    * a y-inverted viewport (measured: dEQP rasterization.culling,
    * builtin_variable.frontfacing and fragment_ops.depth_stencil on a
    * window surface, facetest.sh and stest.sh twoside on an FBO).
    */
   so->cull_face = TGR3D_SU_PARAM_FRONT_FACE(template->front_ccw);
   so->cull_face |= TGR3D_SU_PARAM_CULL(grate_cull_face(template->cull_face,
                                        template->front_ccw));

   return so;
}

static void
grate_bind_rasterizer_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->rast = so;
}

static void
grate_delete_rasterizer_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_rasterizer_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_rasterizer_state = grate_create_rasterizer_state;
   pcontext->bind_rasterizer_state = grate_bind_rasterizer_state;
   pcontext->delete_rasterizer_state = grate_delete_rasterizer_state;
}

static int
grate_compare_func(enum pipe_compare_func func)
{
   switch (func) {
   case PIPE_FUNC_NEVER: return TGR3D_FUNC_NEVER;
   case PIPE_FUNC_LESS: return TGR3D_FUNC_LESS;
   case PIPE_FUNC_EQUAL: return TGR3D_FUNC_EQUAL;
   case PIPE_FUNC_LEQUAL: return TGR3D_FUNC_LEQUAL;
   case PIPE_FUNC_GREATER: return TGR3D_FUNC_GREATER;
   case PIPE_FUNC_NOTEQUAL: return TGR3D_FUNC_NOTEQUAL;
   case PIPE_FUNC_GEQUAL: return TGR3D_FUNC_GEQUAL;
   case PIPE_FUNC_ALWAYS: return TGR3D_FUNC_ALWAYS;
   default: UNREACHABLE("unknown pipe_compare_func");
   }
}

/* libgrate's TGR3D_STENCIL_OP_* */
static uint32_t
grate_stencil_op(enum pipe_stencil_op op)
{
   switch (op) {
   case PIPE_STENCIL_OP_ZERO: return 0;
   case PIPE_STENCIL_OP_KEEP: return 1;
   case PIPE_STENCIL_OP_REPLACE: return 2;
   case PIPE_STENCIL_OP_INCR: return 3;
   case PIPE_STENCIL_OP_DECR: return 4;
   case PIPE_STENCIL_OP_INVERT: return 5;
   case PIPE_STENCIL_OP_INCR_WRAP: return 6;
   case PIPE_STENCIL_OP_DECR_WRAP: return 7;
   default: UNREACHABLE("unknown pipe_stencil_op");
   }
}

static void *
grate_create_zsa_state(struct pipe_context *pcontext,
                       const struct pipe_depth_stencil_alpha_state *template)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_zsa_state *so = CALLOC_STRUCT(grate_zsa_state);
   grate_trace();
   if (!so)
      return NULL;

   so->base = *template;

   uint32_t depth_test = 0;
   depth_test |= TGR3D_QR_Z_TEST_Z_FUNC(grate_compare_func(template->depth_func));
   depth_test |= TGR3D_QR_Z_TEST_Z_ENABLE(template->depth_enabled);
   depth_test |= TGR3D_QR_Z_TEST_QRAST_FB_WRITE(template->depth_writemask);
   depth_test |= TGR3D_QR_Z_TEST_Z_CLAMP(TGR3D_Z_CLAMP_KILL);

   /* the back face takes the front face's state unless two-sided */
   for (int i = 0; i < 2; i++) {
      const struct pipe_stencil_state *s =
         &template->stencil[i && template->stencil[1].enabled];
      so->s_test[i] = TGR3D_QR_S_TEST_FRONT_S_MASK(s->valuemask) |
                      TGR3D_QR_S_TEST_FRONT_S_FUNC(grate_compare_func(s->func));
      so->s_operation[i] =
         TGR3D_GLOBAL_S_OPERATION_FRONT_S_FAIL(grate_stencil_op(s->fail_op)) |
         TGR3D_GLOBAL_S_OPERATION_FRONT_Z_FAIL(grate_stencil_op(s->zfail_op)) |
         TGR3D_GLOBAL_S_OPERATION_FRONT_Z_PASS(grate_stencil_op(s->zpass_op)) |
         TGR3D_GLOBAL_S_OPERATION_FRONT_S_WR_MASK(s->writemask);
   }

   so->commands[0] = host1x_opcode_incr(REG_TGR3D_QR_Z_TEST, 1);
   so->commands[1] = depth_test;
   so->num_commands = 2;

   if (context->drm->soc_id == DRM_TEGRA_SOC_T114) {
      so->commands[2] = host1x_opcode_incr(0xe45, 1);
      so->commands[3] = depth_test;
      so->num_commands = 4;
   }

   return so;
}

static void
grate_bind_zsa_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->zsa = so;
}

static void
grate_delete_zsa_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_zsa_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_depth_stencil_alpha_state = grate_create_zsa_state;
   pcontext->bind_depth_stencil_alpha_state = grate_bind_zsa_state;
   pcontext->delete_depth_stencil_alpha_state = grate_delete_zsa_state;
}

/*
 * Note: this does not include the stride, which needs to be mixed in later
 **/
static uint32_t
attrib_mode(const struct pipe_vertex_element *e)
{
   const struct util_format_description *desc = util_format_description(e->src_format);
   const int c = util_format_get_first_non_void_channel(e->src_format);
   uint32_t type, format;

   assert(!desc->is_mixed);
   assert(c >= 0);

   switch (desc->channel[c].type) {
   case UTIL_FORMAT_TYPE_UNSIGNED:
   case UTIL_FORMAT_TYPE_SIGNED:
      switch (desc->channel[c].size) {
      case 8:
         type = TGR3D_ATTR_FMT_U8;
         break;

      case 16:
         type = TGR3D_ATTR_FMT_U16;
         break;

      case 32:
         type = TGR3D_ATTR_FMT_U32;
         break;

      default:
         UNREACHABLE("invalid channel-size");
      }

      if (desc->channel[c].type == UTIL_FORMAT_TYPE_SIGNED)
         type += 2;

      if (desc->channel[c].normalized)
         type += 1;

      break;

   case UTIL_FORMAT_TYPE_FIXED:
      assert(desc->channel[c].size == 32);
      type = TGR3D_ATTR_FMT_X32;
      break;

   case UTIL_FORMAT_TYPE_FLOAT:
      assert(desc->channel[c].size == 32); /* TODO: float16 ? */
      type = TGR3D_ATTR_FMT_F32;
      break;

   default:
      UNREACHABLE("invalid channel-type");
   }

   format  = TGR3D_IDX_ATTRIBUTE_MODE_ATTR_FMT(type);
   format |= TGR3D_IDX_ATTRIBUTE_MODE_ATTR_SIZE(desc->nr_channels);
   return format;
}

static void *
grate_create_vertex_state(struct pipe_context *pcontext, unsigned int count,
                          const struct pipe_vertex_element *elements)
{
   unsigned int i;
   uint16_t mask = 0;
   struct grate_vertex_state *vtx = CALLOC_STRUCT(grate_vertex_state);
   if (!vtx)
      return NULL;

   for (i = 0; i < count; ++i) {
      const struct pipe_vertex_element *src = elements + i;
      struct grate_vertex_element *dst = vtx->elements + i;
      dst->attrib = attrib_mode(src);
      dst->buffer_index = src->vertex_buffer_index;
      dst->offset = src->src_offset;
      dst->stride = src->src_stride;
      mask |= 1 << i;
   }

   vtx->num_elements = count;
   vtx->mask = mask;

   return vtx;
}

static void
grate_bind_vertex_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->vs = so;
}

static void
grate_delete_vertex_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

static void
emit_attribs(struct grate_context *context, uint32_t **ptrp)
{
   unsigned int i;
   struct grate_stream *stream = &context->gr3d->stream;

   assert(context->vs);

   for (i = 0; i < context->vs->num_elements; ++i) {
      const struct pipe_vertex_buffer *vb;
      const struct grate_vertex_element *e = context->vs->elements + i;
      const struct grate_resource *r;

      assert(e->buffer_index < context->vbs.count);
      vb = context->vbs.vb + e->buffer_index;
      assert(!vb->is_user_buffer);
      r = grate_resource(vb->buffer.resource);

      uint32_t attrib = e->attrib;
      assert(e->stride < 1 << 24);
      attrib |= TGR3D_IDX_ATTRIBUTE_MODE_ATTR_STRIDE(e->stride);

      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_IDX_ATTRIBUTE_BASE(i), 2));
      grate_stream_push_reloc(stream, ptrp, r->bo, vb->buffer_offset + e->offset, false);
      GRATE_PUSHBUF_WORD(*ptrp, attrib);
   }
}

/* DW_ST_ENABLE: surface 0 is the depth buffer (libgrate's
 * TGR3D_RT_ENABLE_DEPTH_BUFFER) */
#define GRATE_DW_ST_DEPTH_BUFFER 0x00010000

/* Whether the bound state tests stencil against a bound stencil buffer */
static bool
grate_stencil_test(const struct grate_context *context)
{
   return context->zsa && context->zsa->base.stencil[0].enabled &&
          (context->framebuffer.rt_present & (1 << GRATE_RT_STENCIL));
}

/* Whether the bound state tests depth against a bound depth buffer */
static bool
grate_depth_test(const struct grate_context *context)
{
   return context->zsa && context->zsa->base.depth_enabled &&
          (context->framebuffer.rt_present & (1 << GRATE_RT_DEPTH));
}

/*
 * The colour surface, and the depth and stencil ones while their tests are
 * on, as libgrate does: the units use surfaces 0 and 2 whenever enabled.
 */
static void
emit_render_targets(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   const struct grate_framebuffer_state *fb = &context->framebuffer;
   uint32_t enable = fb->rt_present & (1 << GRATE_RT_COLOR);

   if (grate_depth_test(context))
      enable |= 1 << GRATE_RT_DEPTH | GRATE_DW_ST_DEPTH_BUFFER;
   if (grate_stencil_test(context))
      enable |= 1 << GRATE_RT_STENCIL;

   u_foreach_bit(i, enable & 0xffff) {
      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_GLOBAL_SURFDESC(i), 1));
      GRATE_PUSHBUF_WORD(*ptrp, fb->rt_params[i]);
      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_GLOBAL_SURFADDR(i), 1));
      grate_stream_push_reloc(stream, ptrp, fb->rt_bos[i], fb->rt_offsets[i], true);
   }

   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_DW_ST_ENABLE, 1));
   GRATE_PUSHBUF_WORD(*ptrp, enable);
}

/*
 * The scissor rectangle is in the same coordinates as the viewport output
 * (framebuffer rows from the first in memory), as the hardware rasterizes;
 * max is exclusive. It is always the viewport, within the scissor while the
 * scissor test is on: the hardware clips primitives to the guard band only,
 * so parts outside a viewport smaller than the framebuffer would be drawn
 * (dEQP clipping).
 */
static void
emit_scissor(struct grate_context *context, uint32_t **ptrp)
{
   const struct pipe_framebuffer_state *fb = &context->framebuffer.base;
   struct pipe_scissor_state r = context->viewport_rect;

   if (context->rast && context->rast->base.scissor) {
      const struct pipe_scissor_state *s = &context->scissor;
      r.minx = MAX2(r.minx, s->minx);
      r.miny = MAX2(r.miny, s->miny);
      r.maxx = MIN2(r.maxx, s->maxx);
      r.maxy = MIN2(r.maxy, s->maxy);
   }

   unsigned maxx = MIN2(r.maxx, fb->width), maxy = MIN2(r.maxy, fb->height);
   unsigned minx = MIN2(r.minx, maxx), miny = MIN2(r.miny, maxy);

   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_SU_SCISSOR_X, 2));
   GRATE_PUSHBUF_WORD(*ptrp, TGR3D_SU_SCISSOR_X_MIN(minx) |
                             TGR3D_SU_SCISSOR_X_MAX(maxx));
   GRATE_PUSHBUF_WORD(*ptrp, TGR3D_SU_SCISSOR_Y_MIN(miny) |
                             TGR3D_SU_SCISSOR_Y_MAX(maxy));
}

/* Wide lines: the setup unit takes half the width (in pixels) */
static void
emit_line_width(struct grate_context *context, uint32_t **ptrp)
{
   float width = context->rast ? context->rast->base.line_width : 1.0f;

   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_SU_LINE_WIDTH_2, 1));
   GRATE_PUSHBUF_WORD(*ptrp, u_bitcast_f2u(MAX2(width, 1.0f) * 0.5f));
}

/* The VS export with that semantic, numbered among the enabled outputs (the
 * hardware squeezes the export mask: exports 2, 7, 9 are seen as 0, 1, 2),
 * -1 if none */
static int
grate_vs_export(const struct grate_context *context, uint16_t semantic)
{
   for (int o = 0; context->vshader && o < 16; o++)
      if (context->vshader->output_semantic[o] == semantic)
         return util_bitcount(context->vshader->output_mask & BITFIELD_MASK(o));
   return -1;
}

/*
 * Point size: gl_PointSize from its VS export (POINT_SIZE_MODE_PSIZE, the
 * export and component as the linker numbers them; the full width), else
 * the fixed half width. The sprite coordinates go from MIN to MAX across
 * the point, T down the framebuffer rows (measured): the upper left origin,
 * swapped for the lower left one.
 */
static void
emit_point_params(struct grate_context *context, uint32_t **ptrp)
{
   const struct pipe_rasterizer_state *rast = context->rast ? &context->rast->base : NULL;
   int psize = grate_vs_export(context, GRATE_SEMANTIC(TGSI_SEMANTIC_PSIZE, 0));
   float size = rast ? rast->point_size : 1.0f;
   bool lower_left = rast && rast->sprite_coord_mode == PIPE_SPRITE_COORD_LOWER_LEFT;
   uint32_t param = TGR3D_SU_POINT_PARAM_CLIP_MODE(TGR3D_CLIP_MODE_GUARDBAND);

   if (psize >= 0 && (!rast || rast->point_size_per_vertex))
      param |= TGR3D_SU_POINT_PARAM_VTX_OFFS(psize) |
               TGR3D_SU_POINT_PARAM_ATTR_COMP(TGR3D_ATTR_COMP_X) |
               TGR3D_SU_POINT_PARAM_POINT_SIZE_MODE(TGR3D_POINT_SIZE_MODE_PSIZE);
   else
      param |= TGR3D_SU_POINT_PARAM_POINT_SIZE_MODE(TGR3D_POINT_SIZE_MODE_FIXED);

   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_SU_POINT_PARAM, 6));
   GRATE_PUSHBUF_WORD(*ptrp, param);
   GRATE_PUSHBUF_WORD(*ptrp, u_bitcast_f2u(MAX2(size, 1.0f) * 0.5f));
   GRATE_PUSHBUF_WORD(*ptrp, u_bitcast_f2u(1.0f)); /* MAX_S */
   GRATE_PUSHBUF_WORD(*ptrp, u_bitcast_f2u(lower_left ? 0.0f : 1.0f)); /* MAX_T */
   GRATE_PUSHBUF_WORD(*ptrp, u_bitcast_f2u(0.0f)); /* MIN_S */
   GRATE_PUSHBUF_WORD(*ptrp, u_bitcast_f2u(lower_left ? 1.0f : 0.0f)); /* MIN_T */
}

/*
 * Polygon offset (measured, gpu/dev/zbiastest.sh): the setup unit adds
 * ZFACTOR (a float) times max(|dz/dx|, |dz/dy|) in window depth per pixel,
 * plus ZBIAS, a signed 22-bit count of 2^-20, the rasterizer's depth step.
 * The minimum resolvable difference r of the units is the depth plane's
 * step, the coarsest one (2^-14, depth below 0.5) for Z16NL. Polygons only,
 * not lines and points.
 */
static void
emit_depth_offset(struct grate_context *context, uint32_t **ptrp, enum mesa_prim mode)
{
   const struct pipe_rasterizer_state *rast = context->rast ? &context->rast->base : NULL;
   const struct pipe_surface *zs = &context->framebuffer.base.zsbuf;
   uint32_t bias = 0;
   float factor = 0.0f;

   if (rast && rast->offset_tri && zs->texture &&
       u_reduced_prim(mode) == MESA_PRIM_TRIANGLES) {
      int r = grate_resource(zs->texture)->format == TGR3D_SURF_FORMAT_Z16NL ? 64 : 16;
      long units = lroundf(rast->offset_units * r);
      bias = CLAMP(units, -(1 << 21), (1 << 21) - 1) & TGR3D_SU_ZBIAS_VAL__MASK;
      factor = rast->offset_scale;
   }

   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_SU_ZBIAS, 2));
   GRATE_PUSHBUF_WORD(*ptrp, bias);
   GRATE_PUSHBUF_WORD(*ptrp, u_bitcast_f2u(factor));
}

static void
emit_viewport(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, ptrp, context->viewport, 10, 0);
}

static void
emit_guardband(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, ptrp, context->guardband, 4, 0);
}

static void
emit_zsa_state(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   uint32_t commands[ARRAY_SIZE(context->zsa->commands)];
   int n = context->zsa->num_commands;

   memcpy(commands, context->zsa->commands, n * sizeof(commands[0]));
   /* without the depth test (or a depth buffer) the unit stays off */
   if (!grate_depth_test(context))
      for (int i = 1; i < n; i += 2)
         commands[i] &= ~(TGR3D_QR_Z_TEST_Z_ENABLE__MASK |
                          TGR3D_QR_Z_TEST_QRAST_FB_WRITE__MASK);
   grate_stream_push_words(stream, ptrp, commands, n, 0);

   if (grate_stencil_test(context)) {
      const struct grate_zsa_state *zsa = context->zsa;
      /*
       * The stencil buffer is written by the rasterizer, before the
       * fragment program runs. libgrate has a program that discards
       * write it after the test instead (DW "store stencil", late test in
       * QR_OUTPUT_TO_SHADER), which the compiler does not emit: the
       * fragments it discards still update the stencil buffer.
       */
      bool discards = context->fshader && context->fshader->discards;
      static bool warned;
      if (discards && !warned) {
         warned = true;
         mesa_logw("grate: stencil written before discard\n");
      }

      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_QR_S_TEST_FRONT, 3));
      GRATE_PUSHBUF_WORD(*ptrp, zsa->s_test[0]);
      GRATE_PUSHBUF_WORD(*ptrp, zsa->s_test[1]);
      GRATE_PUSHBUF_WORD(*ptrp, TGR3D_QR_S_CTRL_S_ENABLE(TGR3D_STATE_ENABLED) |
                                TGR3D_QR_S_CTRL_S_SURF_PTR(GRATE_RT_STENCIL) |
                                TGR3D_QR_S_CTRL_QRAST_FB_WRITE(TGR3D_STATE_ENABLED));
      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_GLOBAL_S_OPERATION_FRONT, 2));
      GRATE_PUSHBUF_WORD(*ptrp, zsa->s_operation[0] |
                                TGR3D_GLOBAL_S_OPERATION_FRONT_S_REF(context->stencil_ref.ref_value[0]));
      GRATE_PUSHBUF_WORD(*ptrp, zsa->s_operation[1] |
                                TGR3D_GLOBAL_S_OPERATION_BACK_S_REF(context->stencil_ref.ref_value[1]));
   }
}

static void
emit_vs_uniforms(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   struct pipe_constant_buffer *constbuf = &context->constant_buffer[MESA_SHADER_VERTEX];
   const struct grate_vertex_shader_state *vs = context->vshader;
   uint32_t consts[GRATE_VP_NUM_CONSTS * 4] = { 0 };
   int len = 0;

   if (constbuf->user_buffer != NULL) {
      assert(constbuf->buffer_size % sizeof(uint32_t) == 0);
      len = constbuf->buffer_size / 4;
      assert(len <= GRATE_VP_NUM_CONSTS * 4);
      memcpy(consts, constbuf->user_buffer, len * 4);
   }

   /* immediates are allocated from the top, see tgsi_src_to_vp() */
   for (int i = 0; i < vs->num_immediates; i++) {
      int slot = GRATE_VP_NUM_CONSTS - 1 - i;
      assert(slot * 4 >= len);
      memcpy(&consts[slot * 4], vs->immediates[i], sizeof(vs->immediates[i]));
   }

   /* the fragment program's uniforms its moved coordinates read */
   if (vs->fs_const_base >= 0 && context->fshader) {
      uint32_t fs[GRATE_FP_NUM_CONSTS];
      grate_fs_consts(context, fs);
      for (int k = 0; k < GRATE_FP_NUM_CONSTS; k++) {
         union { float f; uint32_t u; } v = { .f = fp20_to_float(fs[k]) };
         assert((vs->fs_const_base + k) * 4 >= len);
         consts[(vs->fs_const_base + k) * 4] = v.u;
      }
   }

   if (len == 0 && vs->num_immediates == 0)
      return;

   /* upload the whole block from offset 0, as the blob does */
   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_imm(REG_TGR3D_VPE_CONST_OFFSET, 0));
   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_nonincr(REG_TGR3D_VPE_CONST_DATA,
                                                   GRATE_VP_NUM_CONSTS * 4));
   grate_stream_push_words(stream, ptrp, consts, GRATE_VP_NUM_CONSTS * 4, 0);
}

static void
emit_shader(struct grate_stream *stream, uint32_t **ptrp, struct grate_shader_blob *blob)
{
   grate_stream_push_words(stream, ptrp, blob->commands, blob->num_commands, 0);
}

/*
 * The fragment program. One PSEQ pass: its blob has it all (the command
 * 0x20006000 | EXECs in EVEN(0)). Several (grate_fp_chunks): the PSEQ
 * fetches each chunk from memory and runs it, as NVIDIA's driver does
 * (libGLESv2_tegra.so 0x304c4; asmtri exp-pseq2/3):
 *   0x10000000 | words, address   fetch and write the chunk's registers
 *   0x20002000 | EXECs, 0          run them, more passes follow
 *   0x20004000 | EXECs, 0          ... the last pass
 *   0x40000000 | (-n & 31), 0      back to command 0 for the next quads
 *                                  (a jump n commands back, libcgdrv.so 0xdce50)
 * with the spill surface for the quads between passes (0x13d0 bytes in
 * NVIDIA's driver, 16 per quad ID).
 */
static void
emit_fs_passes(struct grate_context *context, uint32_t **ptrp,
               struct grate_shader_blob *blob)
{
   struct grate_stream *stream = &context->gr3d->stream;

   if (blob->num_chunks < 2 || !blob->chunk_bo) {
      if (context->pseq_cmds_set) {
         /* what a program of several passes left past command 0 */
         GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_PSEQ_COMMAND_ODD(0), 31));
         for (int i = 0; i < 31; i++)
            GRATE_PUSHBUF_WORD(*ptrp, 0);
         context->pseq_cmds_set = false;
      }
      emit_shader(stream, ptrp, blob);
      return;
   }

   emit_shader(stream, ptrp, blob);
   if (!context->spill_bo)
      context->spill_bo = grate_bo_alloc(context->drm, 0x2000, 0);
   if (context->spill_bo) {
      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_GLOBAL_SPILLSURFADDR, 1));
      grate_stream_push_reloc(stream, ptrp, context->spill_bo, 0, true);
   }
   int n = blob->num_chunks, cmds = 2 * n + 1;
   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_PSEQ_COMMAND_EVEN(0), 2 * cmds));
   for (int k = 0; k < n; k++) {
      GRATE_PUSHBUF_WORD(*ptrp, 0x10000000 | blob->chunk_len[k]);
      grate_stream_push_reloc(stream, ptrp, blob->chunk_bo, blob->chunk_off[k] * 4, false);
      GRATE_PUSHBUF_WORD(*ptrp, (k + 1 < n ? 0x20002000 : 0x20004000) | blob->chunk_execs[k]);
      GRATE_PUSHBUF_WORD(*ptrp, 0);
   }
   GRATE_PUSHBUF_WORD(*ptrp, 0x40000000 | (-(2 * n) & 31));
   GRATE_PUSHBUF_WORD(*ptrp, 0);
   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_imm(REG_TGR3D_PSEQ_PC, 0));
   context->pseq_cmds_set = true;
}

static void
emit_program(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   uint32_t cull_face_linker_setup;

   emit_shader(stream, ptrp, &context->vshader->blob);
   emit_fs_passes(context, ptrp, &context->fshader->blob);

   /* fragment shader constants: ALU uniform registers 32.. */
   const struct grate_fp_info *fpinfo = context->fshader->info;
   if (fpinfo->reads_uniforms && !BITSET_IS_EMPTY(fpinfo->used_mask)) {
      uint32_t consts[GRATE_FP_NUM_CONSTS];
      grate_fs_consts(context, consts);

      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_ALU_GLOBALS(0),
                                                   GRATE_FP_NUM_CONSTS));
      grate_stream_push_words(stream, ptrp, consts, GRATE_FP_NUM_CONSTS, 0);
   }

   cull_face_linker_setup = TGR3D_SU_PARAM_SUBPIX_XOFF(0x38) |
                            TGR3D_SU_PARAM_SUBPIX_YOFF(0x38) |
                            TGR3D_SU_PARAM_TRANSPOSE_XY(TGR3D_STATE_DISABLED) |
                            TGR3D_SU_PARAM_CLIP_ENABLE(TGR3D_STATE_ENABLED);

   /* depends on cull-face */
   cull_face_linker_setup |= context->rast->cull_face;

   /* depends on linking */
   const struct grate_fp_info *info = context->fshader->info;
   cull_face_linker_setup |= TGR3D_SU_PARAM_LAST_INST(0 < info->num_links ?
                                                      (info->num_links - 1) : 0);

   uint32_t linker_insts[3 + info->num_links * 2];
   linker_insts[0] = host1x_opcode_incr(REG_TGR3D_SU_PARAM, 1);
   linker_insts[1] = cull_face_linker_setup;
   linker_insts[2] = host1x_opcode_incr(REG_TGR3D_SU_INST_EVEN(0), info->num_links * 2);

   for (int i = 0; i < info->num_links; ++i) {
      /*
       * Source: the VS export with the same semantic, export 1 if there is
       * none (the point coordinate reads any: the position, export 0).
       * Bits 9..14: the TRAM row written, which the fragment MFU reads.
       */
      uint32_t src = info->links[i].point ? 0 : 1 << 3;
      int o = grate_vs_export(context, info->links[i].semantic);
      if (o >= 0)
         src = o << 3;
      src |= (uint32_t)info->links[i].tram_row << 9 | info->links[i].point;
      linker_insts[3 + i * 2] = src;
      linker_insts[3 + i * 2 + 1] = info->links[i].dst;
      /* flat shading: no interpolation of the colour */
      if (context->rast->base.flatshade && info->links[i].input == info->color_input)
         linker_insts[3 + i * 2 + 1] |= 0xf << 16;
   }

   grate_stream_push_words(stream, ptrp, linker_insts, ARRAY_SIZE(linker_insts), 0);
}

static bool
grate_wrap_clamps(unsigned wrap)
{
   return wrap == PIPE_TEX_WRAP_CLAMP || wrap == PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
          wrap == PIPE_TEX_WRAP_CLAMP_TO_BORDER;
}

/*
 * Texture descriptors of the samplers the fragment shader reads: linear. The TEX unit derives the pitch from the width (rows aligned
 * to 64 bytes, see grate_screen_resource_create).
 */
static void
emit_textures(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   uint32_t mask = context->fshader ? context->fshader->info->sampler_mask : 0;

   u_foreach_bit(i, mask) {
      struct pipe_sampler_view *view = context->sampler_views[i];
      const struct pipe_sampler_state *sampler = context->samplers[i];
      if (!view || !view->texture)
         continue;

      struct grate_resource *res = grate_resource(view->texture);
      /* other filters for minified and magnified: a power-of-two copy,
       * brought up to date before the draw (grate_draw_vbo) */
      if (grate_npot_shadow(view, sampler) && res->shadow)
         res = res->shadow;
      unsigned width = res->b.width0, height = res->b.height0;
      uint32_t lo, hi;

      /* Z16: no depth format in the TEX unit, the bytes as L8A8
       * (grate_fp_key.depth_tex) */
      unsigned fmt = res->format == TGR3D_SURF_FORMAT_Z16 ? TGR3D_SURF_FORMAT_L8A8
                                                          : res->format;
      lo = TGR3D_TEX_TEXDESC_LO_SURF_FORMAT(fmt) |
           TGR3D_TEX_TEXDESC_LO_LAYOUT(TGR3D_LAYOUT_LINEAR);
      if (view->target == PIPE_TEXTURE_CUBE)
         lo |= TGR3D_TEX_TEXDESC_LO_CUBEMAP;
      /* compressed (DXT, ETC1): libgrate's COMPRESSED bit */
      if (util_format_is_compressed(view->format))
         lo |= TGR3D_TEX_TEXDESC_LO_LAYOUT(TGR3D_LAYOUT_SWIZZLED);
      if (sampler) {
         if (sampler->mag_img_filter == PIPE_TEX_FILTER_LINEAR)
            lo |= TGR3D_TEX_TEXDESC_LO_LERP_MAG;
         if (sampler->min_img_filter == PIPE_TEX_FILTER_LINEAR)
            lo |= TGR3D_TEX_TEXDESC_LO_LERP_MIN;
         if (grate_wrap_clamps(sampler->wrap_s) || grate_npot_repeat(view, sampler, 0))
            lo |= TGR3D_TEX_TEXDESC_LO_CLAMP_S(TGR3D_CLAMP_CLAMP);
         if (grate_wrap_clamps(sampler->wrap_t) || grate_npot_repeat(view, sampler, 1))
            lo |= TGR3D_TEX_TEXDESC_LO_CLAMP_T(TGR3D_CLAMP_CLAMP);
         if (sampler->wrap_s == PIPE_TEX_WRAP_MIRROR_REPEAT)
            lo |= TGR3D_TEX_TEXDESC_LO_MIRROR_S(TGR3D_STATE_ENABLED);
         if (sampler->wrap_t == PIPE_TEX_WRAP_MIRROR_REPEAT)
            lo |= TGR3D_TEX_TEXDESC_LO_MIRROR_T(TGR3D_STATE_ENABLED);
      }

      /*
       * Mipmaps for power-of-two textures, as libgrate: the levels follow
       * each other in the BO, rows aligned to 16 bytes (as
       * grate_screen_resource_create lays them out), LOD_MAX in 4.4 fixed
       * point. Otherwise the base level only.
       */
      bool pot = util_is_power_of_two_nonzero(width) &&
                 util_is_power_of_two_nonzero(height);
      unsigned levels = res->b.last_level;
      bool mip = pot && levels && view->u.tex.first_level == 0 && sampler &&
                 sampler->min_mip_filter != PIPE_TEX_MIPFILTER_NONE;
      if (mip) {
         levels = MIN2(levels, view->u.tex.last_level);
         hi = TGR3D_TEX_TEXDESC_HI_LOD_MAX(levels << 4);
         if (sampler->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR)
            lo |= TGR3D_TEX_TEXDESC_LO_LERP_MIP;
      } else if (pot) {
         /*
          * Measured on cube maps: with BASE_LEVEL_ONLY the TEX unit steps
          * from face to face by the base level alone (the faces are laid out
          * with their whole mip chain) and always filters as magnified. With
          * LOD_MAX = 1/16 it reads level 0 and picks the min or mag filter.
          * LOD_MAX = 0 steps right but always magnifies too. 2D textures
          * the same (dEQP texture.filtering.2d nearest_linear/linear_nearest).
          */
         hi = TGR3D_TEX_TEXDESC_HI_LOD_MAX(1);
      } else {
         hi = TGR3D_TEX_TEXDESC_HI_BASE_LEVEL_ONLY(TGR3D_STATE_ENABLED);
      }
      if (pot)
         hi |= TGR3D_TEX_TEXDESC_HI_LOG2_WIDTH(util_logbase2(width)) |
               TGR3D_TEX_TEXDESC_HI_LOG2_HEIGHT(util_logbase2(height));
      else
         hi |= TGR3D_TEX_TEXDESC_HI_NON_POWER_OF_TWO(TGR3D_STATE_ENABLED) |
               TGR3D_TEX_TEXDESC_HI_WIDTH(width) |
               TGR3D_TEX_TEXDESC_HI_HEIGHT(height);
      /* LOD bias, s4.4 (measured: 0x10 = one level): the sampler's plus
       * the shader's constant TXB bias */
      const struct grate_fp_info *fpinfo = context->fshader->info;
      int bias = fpinfo->lod_bias[i] +
                 (sampler ? (int)lroundf(sampler->lod_bias * 16.0f) : 0);
      if (fpinfo->lod_bias_user_mask & (1u << i)) {
         const struct pipe_constant_buffer *cb =
            &context->constant_buffer[MESA_SHADER_FRAGMENT];
         unsigned u = fpinfo->lod_bias_user[i];
         float v = cb->user_buffer && u < cb->buffer_size / sizeof(float) ?
                   ((const float *)cb->user_buffer)[u] : 0.0f;
         v = CLAMP(v, -16.0f, 16.0f) * 16.0f;
         bias += (fpinfo->lod_bias_user_neg & (1u << i)) ? -(int)lroundf(v) :
                                                           (int)lroundf(v);
      }
      lo |= TGR3D_TEX_TEXDESC_LO_LOD_BIAS(CLAMP(bias, -256, 255) & 0x1ff);

      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_TEX_TEXADDR(i), 1));
      grate_stream_push_reloc(stream, ptrp, res->bo, 0, false);
      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_TEX_TEXDESC_LO(i), 2));
      GRATE_PUSHBUF_WORD(*ptrp, lo);
      GRATE_PUSHBUF_WORD(*ptrp, hi);
   }
}

void
grate_emit_state(struct grate_context *context, uint32_t **ptrp, enum mesa_prim mode)
{
   emit_render_targets(context, ptrp);
   emit_viewport(context, ptrp);
   emit_guardband(context, ptrp);
   emit_scissor(context, ptrp);
   emit_line_width(context, ptrp);
   emit_point_params(context, ptrp);
   emit_depth_offset(context, ptrp, mode);
   emit_zsa_state(context, ptrp);
   emit_attribs(context, ptrp);
   emit_vs_uniforms(context, ptrp);
   emit_program(context, ptrp);
   emit_textures(context, ptrp);
}

void
grate_context_vbo_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_vertex_elements_state = grate_create_vertex_state;
   pcontext->bind_vertex_elements_state = grate_bind_vertex_state;
   pcontext->delete_vertex_elements_state = grate_delete_vertex_state;
}
