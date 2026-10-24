#include <stdio.h>
#include <string.h>

#include "util/u_dynarray.h"
#include "util/u_memory.h"

#include "nir/nir_to_tgsi.h"

#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_scan.h"
#include "tgsi/tgsi_transform.h"

#include "grate_device.h"
#include "host1x01_hardware.h"
#include "grate_common.h"
#include "grate_context.h"
#include "grate_screen.h"
#include "grate_program.h"
#include "grate_compiler.h"
#include "fp/fpir.h"
#include "vp/vpir.h"

struct grate_vs_tgsi_transform_context {
   struct tgsi_transform_context base;
   unsigned next_temp;
};

static void
grate_vs_tgsi_transform_instruction(struct tgsi_transform_context *ctx,
                                    struct tgsi_full_instruction *inst)
{
   struct grate_vs_tgsi_transform_context *gctx =
      (struct grate_vs_tgsi_transform_context *)ctx;

   int const_index = -1;
   for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
      if (inst->Src[i].Register.File != TGSI_FILE_CONSTANT)
         continue;

      if (const_index < 0)
         const_index = inst->Src[i].Register.Index;
      else if (inst->Src[i].Register.Index != const_index) {
         /* We already have a constant-reads in this instruction,
          * add extra instructions in order to read more
          */
         unsigned temp = gctx->next_temp++;
         tgsi_transform_temp_decl(ctx, temp);
         /* insert MOV to temp */
         tgsi_transform_op1_inst(ctx, TGSI_OPCODE_MOV,
                                 /* dst reg */
                                 TGSI_FILE_TEMPORARY, temp,
                                 TGSI_WRITEMASK_XYZW,
                                 /* src reg */
                                 TGSI_FILE_CONSTANT,
                                 inst->Src[i].Register.Index);

         /* rewrite instruction */
         tgsi_transform_src_reg_xyzw(&inst->Src[i], TGSI_FILE_TEMPORARY, temp);
         inst->Src[i].Register.File = TGSI_FILE_TEMPORARY;
         inst->Src[i].Register.Index = temp;
      }
   }

   ctx->emit_instruction(ctx, inst);
}

static struct tgsi_token *
grate_vs_tgsi_transform(const struct tgsi_token *tokens_in)
{
   grate_trace();
   const unsigned new_len = tgsi_num_tokens(tokens_in) + 100;

   struct tgsi_shader_info info;
   tgsi_scan_shader(tokens_in, &info);

   struct grate_vs_tgsi_transform_context ctx = {};
   ctx.base.transform_instruction = grate_vs_tgsi_transform_instruction;
   ctx.next_temp = info.file_max[TGSI_FILE_TEMPORARY] + 1;

   return tgsi_transform_shader(tokens_in, new_len, &ctx.base);
}


/* Compile the vertex program, computing "moved" too if not NULL */
static void
grate_vs_compile(const struct tgsi_token *tokens, const struct grate_moved *moved,
                 struct grate_vs_variant *v)
{
   struct tgsi_parse_context parser;
   unsigned ok = tgsi_parse_init(&parser, tokens);
   assert(ok == TGSI_PARSE_OK);

   struct grate_vp_shader vp;
   memset(v, 0, sizeof(*v));
   if (moved)
      v->moved = *moved;
   v->ok = grate_tgsi_to_vp(&vp, &parser, moved);

   memcpy(v->immediates, vp.immediates, sizeof(vp.immediates));
   v->num_immediates = vp.num_immediates;
   v->fs_const_base = vp.fs_const_base;

   int num_instructions = v->ok ? list_length(&vp.instructions) : 0;
   int num_commands = 2 + num_instructions * 4;
   uint32_t *commands = MALLOC(num_commands * sizeof(uint32_t));

   commands[0] = host1x_opcode_imm(REG_TGR3D_VPE_INST_OFFSET, 0);
   commands[1] = host1x_opcode_nonincr(REG_TGR3D_VPE_INST_DATA,
                                       num_instructions * 4);

   struct vp_instr *last = list_last_entry(&vp.instructions, struct vp_instr, link);
   int offset = 2;
   list_for_each_entry_safe(struct vp_instr, instr, &vp.instructions, link) {
      if (v->ok) {
         grate_vp_pack(commands + offset, instr, instr == last);
         offset += 4;
      }
      FREE(instr);
   }

   v->blob.commands = commands;
   v->blob.num_commands = num_commands;
   v->output_mask = vp.output_mask;
   memcpy(v->output_semantic, vp.output_semantic, sizeof(v->output_semantic));
}

static void
grate_vs_use(struct grate_vertex_shader_state *so, const struct grate_vs_variant *v)
{
   so->blob = v->blob;
   so->output_mask = v->output_mask;
   memcpy(so->output_semantic, v->output_semantic, sizeof(so->output_semantic));
   memcpy(so->immediates, v->immediates, sizeof(so->immediates));
   so->num_immediates = v->num_immediates;
   so->fs_const_base = v->fs_const_base;
}

/* The variant computing "moved" (NULL: the plain one), compiled if needed;
 * NULL if they do not fit */
static struct grate_vs_variant *
grate_vs_variant(struct grate_vertex_shader_state *so, const struct grate_moved *moved)
{
   static const struct grate_moved none;
   unsigned i;

   if (!moved)
      moved = &none;
   for (i = 0; i < so->num_variants; i++)
      if (!memcmp(&so->variants[i].moved, moved, sizeof(*moved)))
         return so->variants[i].ok ? &so->variants[i] : NULL;

   if (so->num_variants < GRATE_VS_MAX_VARIANTS) {
      i = so->num_variants++;
   } else {
      /* replace one, never the plain variant 0 */
      i = 1 + so->next_evict++ % (GRATE_VS_MAX_VARIANTS - 1);
      FREE(so->variants[i].blob.commands);
   }
   grate_vs_compile(so->tokens, moved == &none ? NULL : moved, &so->variants[i]);
   so->variants[i].moved = *moved;
   return so->variants[i].ok ? &so->variants[i] : NULL;
}

static void *
grate_create_vs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   grate_trace();
   struct grate_vertex_shader_state *so =
      CALLOC_STRUCT(grate_vertex_shader_state);

   if (!so)
      return NULL;

   so->base = *template;

   if (template->type == PIPE_SHADER_IR_NIR) {
      so->base.tokens = nir_to_tgsi(template->ir.nir,
                                    pcontext->screen);
      so->base.type = PIPE_SHADER_IR_TGSI;
   }


   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "DEBUG: TGSI:\n");
      tgsi_dump(so->base.tokens, 0);
      fprintf(stderr, "\n");
   }

   so->tokens = grate_vs_tgsi_transform(so->base.tokens);
   if (!so->tokens) {
      FREE(so);
      return NULL;
   }

   /* the plain variant */
   struct grate_vs_variant *v = grate_vs_variant(so, NULL);
   assert(v);
   grate_vs_use(so, v);
   return so;
}

static void
grate_bind_vs_state(struct pipe_context *pcontext, void *so)
{
   grate_trace();
   grate_context(pcontext)->vshader = so;
}

static void
grate_delete_vs_state(struct pipe_context *pcontext, void *hwcso)
{
   struct grate_vertex_shader_state *so = hwcso;

   grate_trace();
   for (unsigned i = 0; i < so->num_variants; i++)
      FREE(so->variants[i].blob.commands);
   FREE(so->tokens);
   FREE(so);
}

/* Compile a fragment program variant into its command stream */
static void
grate_fs_compile(enum drm_tegra_soc_id soc_id, const struct tgsi_token *tokens,
                 const struct grate_fp_key *key, struct grate_fs_variant *v)
{
   struct tgsi_parse_context parser;
   unsigned ok = tgsi_parse_init(&parser, tokens);
   assert(ok == TGSI_PARSE_OK);

   struct grate_fp_shader fp;
   grate_tgsi_to_fp(&fp, &parser, key);

   /* say which shader draws magenta, to have something to work from (not
    * the generic version of one that gets specialized at draw time) */
   if (fp.info.failed && !fp.info.num_flow && (!fp.info.num_spec || (key && key->spec))) {
      fprintf(stderr, "GRATE FRAG: failed shader (blend %d: rgb %d %d %d, "
              "alpha %d %d %d, colormask 0x%x):\n",
              key ? key->blend : 0, key ? key->rgb_func : 0,
              key ? key->rgb_src : 0, key ? key->rgb_dst : 0,
              key ? key->alpha_func : 0, key ? key->alpha_src : 0,
              key ? key->alpha_dst : 0, key ? key->colormask : 0xf);
      tgsi_dump_to_file(tokens, 0, stderr);
   }

   struct util_dynarray buf;
   util_dynarray_init(&buf, NULL);

#define PUSH(x) util_dynarray_append_typed(&buf, uint32_t, (x))
   /* one pixel packet row (ALU buffer size 1); MAX_QID matches PSEQ_CTL
    * MAX_OUT = 0x12c as in the grate tests: (0x12c - 1) / (rows * 4) */
   PUSH(host1x_opcode_incr(REG_TGR3D_GLOBAL_PIX_ATTR, 1));
   PUSH(TGR3D_GLOBAL_PIX_ATTR_MAX_QID((0x12c - 1) / (fp.info.num_rows * 4)) |
        TGR3D_GLOBAL_PIX_ATTR_NUM_ROWS(fp.info.num_rows - 1));

   PUSH(host1x_opcode_imm(REG_TGR3D_PSEQ_QUAD_ID, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_GLOBAL_INST_OFFSET, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_AT_INST_OFFSET, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_ALU_INST_OFFSET, 0));

   int num_fp_instrs = list_length(&fp.fp_instructions);
   assert(num_fp_instrs < 64);

   PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_COMMAND_EVEN(0), 1));
   // TODO: document/convert these values
   PUSH(0x20006000 | num_fp_instrs);

   /*
    * FP_PSEQ_DW_CFG: PSEQ_TO_DW_EXEC_NB (bits 6..12) = how many instruction
    * batches run before the DW stage stores the pixel. Every batch we emit
    * has an ALU packet, so the store has to wait for all of them (a value
    * of 1 stores the pixel right after the first batch).
    */
   PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_DWR_IF_STATE, 1));
   PUSH(num_fp_instrs << 6);

   if (soc_id == DRM_TEGRA_SOC_T114) {
      // TODO: document/convert these values
      /* XXX: maybe not needed */
      PUSH(host1x_opcode_incr(0x547, 0x0002));
      PUSH(0xc0000000);
      PUSH(0x00000000);
   }

   PUSH(host1x_opcode_imm(REG_TGR3D_PSEQ_FLUSH, 0));

   PUSH(host1x_opcode_nonincr(REG_TGR3D_PSEQ_INST_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(instr->pseq);

   PUSH(host1x_opcode_nonincr(REG_TGR3D_AT_REMAP_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(grate_fp_pack_sched(&instr->mfu_sched));

   int num_mfu_instrs = list_length(&fp.mfu_instructions);
   assert(num_mfu_instrs < 64); // TODO: not sure if this is really correct

   PUSH(host1x_opcode_nonincr(REG_TGR3D_AT_INST_DATA_LO, num_mfu_instrs * 2));
   list_for_each_entry(struct fp_mfu_instr, instr, &fp.mfu_instructions, link) {
      uint32_t words[2];
      grate_fp_pack_mfu(words, instr);
      PUSH(words[0]);
      PUSH(words[1]);
   }

   PUSH(host1x_opcode_nonincr(REG_TGR3D_TEX_INST_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(instr->tex);

   PUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_REMAP_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link) {
      if (soc_id == DRM_TEGRA_SOC_T114)
         PUSH(grate_fp_pack_alu_sched_t114(&instr->alu_sched));
      else
         PUSH(grate_fp_pack_sched(&instr->alu_sched));
   }

   int num_alu_instrs = list_length(&fp.alu_instructions);
   PUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_INST_DATA, num_alu_instrs * 4 * 2));
   list_for_each_entry(struct fp_alu_instr_packet, instr, &fp.alu_instructions, link) {
      for (int i = 0; i < 4; ++i) {
         uint32_t words[2];
         grate_fp_pack_alu(words, instr->slots + i);
         PUSH(words[0]);
         PUSH(words[1]);
      }
   }

   PUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_P2CX_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(0x00000000);

   PUSH(host1x_opcode_nonincr(REG_TGR3D_DW_INST_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(grate_fp_pack_dw(&instr->dw));

   uint32_t tram_setup = 0;
   tram_setup |= TGR3D_GLOBAL_TRI_ATTR_NUM_TRIS(64 / fp.info.max_tram_row);
   tram_setup |= TGR3D_GLOBAL_TRI_ATTR_TRI_ROWS(fp.info.max_tram_row);

   PUSH(host1x_opcode_incr(REG_TGR3D_GLOBAL_TRI_ATTR, 1));
   PUSH(tram_setup);

#undef PUSH
   util_dynarray_trim(&buf);

   v->blob.num_commands = buf.size / sizeof(uint32_t);
   v->blob.commands = buf.data;
   v->info = fp.info;
   v->key = *key;

   /* machine code for gpu/tools/fpdis */
   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "FPBLOB %u", v->blob.num_commands);
      for (unsigned i = 0; i < v->blob.num_commands; i++)
         fprintf(stderr, " %08x", v->blob.commands[i]);
      fprintf(stderr, "\n");
      for (int i = 0; i < GRATE_FP_NUM_CONSTS; i++)
         if (BITSET_TEST(fp.info.imm_mask, i))
            fprintf(stderr, "FPCONST %d %05x\n", i, fp.info.consts[i]);
   }
}

static const struct grate_fp_key grate_fp_key_none = { .colormask = 0xf };

static void *
grate_create_fs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   grate_trace();
   struct grate_context *context = grate_context(pcontext);
   struct grate_fragment_shader_state *so = CALLOC_STRUCT(grate_fragment_shader_state);

   if (!so)
      return NULL;

   so->base = *template;

   /* variants are compiled later, from our own copy of the tokens */
   if (template->type == PIPE_SHADER_IR_NIR) {
      so->base.tokens = nir_to_tgsi(template->ir.nir, pcontext->screen);
      so->base.type = PIPE_SHADER_IR_TGSI;
   } else
      so->base.tokens = tgsi_dup_tokens(template->tokens);

   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "DEBUG: TGSI:\n");
      tgsi_dump(so->base.tokens, 0);
      fprintf(stderr, "\n");
   }

   struct tgsi_shader_info scan;
   tgsi_scan_shader(so->base.tokens, &scan);
   so->discards = scan.uses_kill;

   /* the opaque variant, used without blending */
   grate_fs_compile(context->drm->soc_id, so->base.tokens, &grate_fp_key_none,
                    &so->variants[0]);
   so->num_variants = 1;
   so->blob = so->variants[0].blob;
   so->info = so->variants[0].info;
   return so;
}

/* Variant key of the bound blend state (render target 0) */
static struct grate_fp_key
grate_fs_key(struct grate_context *context)
{
   const struct pipe_blend_state *blend = context->blend;
   struct grate_fp_key key;

   memset(&key, 0, sizeof(key));
   key.colormask = 0xf;

   if (!blend)
      return key;

   const struct pipe_rt_blend_state *rt = &blend->rt[0];
   key.colormask = rt->colormask & 0xf;
   if (rt->blend_enable) {
      key.blend = 1;
      key.rgb_func = rt->rgb_func;
      key.rgb_src = rt->rgb_src_factor;
      key.rgb_dst = rt->rgb_dst_factor;
      key.alpha_func = rt->alpha_func;
      key.alpha_src = rt->alpha_src_factor;
      key.alpha_dst = rt->alpha_dst_factor;
   }
   return key;
}

static struct grate_fs_variant *
grate_fs_variant(struct grate_context *context,
                 struct grate_fragment_shader_state *so,
                 const struct grate_fp_key *key)
{
   unsigned i;

   for (i = 0; i < so->num_variants; i++)
      if (!memcmp(&so->variants[i].key, key, sizeof(*key)))
         return &so->variants[i];

   if (so->num_variants < GRATE_FS_MAX_VARIANTS) {
      i = so->num_variants++;
   } else {
      /* replace one, never the opaque variant 0 */
      i = 1 + so->next_evict++ % (GRATE_FS_MAX_VARIANTS - 1);
      FREE(so->variants[i].blob.commands);
   }
   grate_fs_compile(context->drm->soc_id, so->base.tokens, key,
                    &so->variants[i]);
   return &so->variants[i];
}

/*
 * The variant for the bound blend state and, when the program branches on
 * uniforms (grate_fp_spec), for the values they have at this draw
 */
static struct grate_fs_variant *
grate_fs_select(struct grate_context *context, struct grate_fragment_shader_state *so,
                struct grate_fp_key key, const float *user, unsigned user_len)
{
   struct grate_fs_variant *v = grate_fs_variant(context, so, &key);
   if (v->info.num_flow) {
      /* loops counted by uniforms: the variant for their values */
      key.num_flow = v->info.num_flow;
      for (int i = 0; i < v->info.num_flow; i++) {
         unsigned u = v->info.flow_user[i];
         key.flow_user[i] = u;
         key.flow[i] = u < user_len ? user[u] : 0.0f;
      }
      v = grate_fs_variant(context, so, &key);
   }
   if (v->info.num_spec) {
      key.spec = 1;
      key.spec_bits = grate_fp_eval_spec(&v->info, user, user_len);
      v = grate_fs_variant(context, so, &key);
   }
   return v;
}

void
grate_fs_update(struct grate_context *context)
{
   struct grate_fragment_shader_state *so = context->fshader;
   struct grate_vertex_shader_state *vs = context->vshader;
   struct grate_fp_key key = grate_fs_key(context);

   if (!so)
      return;

   const struct pipe_constant_buffer *cb =
      &context->constant_buffer[MESA_SHADER_FRAGMENT];
   const float *user = cb->user_buffer;
   unsigned user_len = user ? cb->buffer_size / sizeof(float) : 0;

   struct grate_fs_variant *v = grate_fs_select(context, so, key, user, user_len);
   struct grate_vs_variant *vv = vs ? grate_vs_variant(vs, NULL) : NULL;

   /* texture coordinates the vertex shader can compute: when both fit */
   if (vv && v->info.can_move) {
      key.move = 1;
      struct grate_fs_variant *mv = grate_fs_select(context, so, key, user, user_len);
      struct grate_vs_variant *mvv = !mv->info.failed && mv->info.moved.num_outs ?
                                     grate_vs_variant(vs, &mv->info.moved) : NULL;
      if (mvv) {
         v = mv;
         vv = mvv;
      }
   }

   so->blob = v->blob;
   so->info = v->info;
   if (vv)
      grate_vs_use(vs, vv);
}

void
grate_fs_consts(struct grate_context *context, uint32_t *consts)
{
   const struct grate_fp_info *fpinfo = &context->fshader->info;
   const struct pipe_constant_buffer *cb =
      &context->constant_buffer[MESA_SHADER_FRAGMENT];
   const float *user = cb->user_buffer;
   unsigned user_len = user ? cb->buffer_size / sizeof(float) : 0;

   for (int i = 0; i < GRATE_FP_NUM_CONSTS; i++) {
      unsigned u = fpinfo->user_index[i];
      if (BITSET_TEST(fpinfo->imm_mask, i))
         consts[i] = fpinfo->consts[i];
      else if (BITSET_TEST(fpinfo->user_mask, i) && u < user_len)
         consts[i] = grate_float_to_fp20(user[u]);
      else
         consts[i] = 0;
   }
   grate_fp_eval_pre(fpinfo, user, user_len, consts);
}

static void
grate_bind_fs_state(struct pipe_context *pcontext, void *so)
{
   grate_trace();
   grate_context(pcontext)->fshader = so;
}

static void
grate_delete_fs_state(struct pipe_context *pcontext, void *hwcso)
{
   struct grate_fragment_shader_state *so = hwcso;

   grate_trace();
   for (unsigned i = 0; i < so->num_variants; i++)
      FREE(so->variants[i].blob.commands);
   FREE((void *)so->base.tokens);
   FREE(so);
}

void
grate_context_program_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_vs_state = grate_create_vs_state;
   pcontext->bind_vs_state = grate_bind_vs_state;
   pcontext->delete_vs_state = grate_delete_vs_state;

   pcontext->create_fs_state = grate_create_fs_state;
   pcontext->bind_fs_state = grate_bind_fs_state;
   pcontext->delete_fs_state = grate_delete_fs_state;
}
