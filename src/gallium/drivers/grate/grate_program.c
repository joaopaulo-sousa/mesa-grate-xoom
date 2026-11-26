#include <stdio.h>
#include <string.h>

#include "util/blob.h"
#include "util/disk_cache.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/u_memory.h"

#include "compiler/nir/nir.h"
#include "nir/nir_to_tgsi.h"

#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_scan.h"
#include "tgsi/tgsi_transform.h"

#include "grate_device.h"
#include "host1x01_hardware.h"
#include "grate_common.h"
#include "grate_bo.h"
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
      NIR_PASS(_, template->ir.nir, nir_lower_point_size, 1.0f, GRATE_MAX_POINT_SIZE);
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

static void
grate_fs_free(struct grate_fp_shader *fp)
{
   list_for_each_entry_safe(struct fp_instr, i, &fp->fp_instructions, link)
      FREE(i);
   list_for_each_entry_safe(struct fp_alu_instr_packet, i, &fp->alu_instructions, link)
      FREE(i);
   list_for_each_entry_safe(struct fp_mfu_instr, i, &fp->mfu_instructions, link)
      FREE(i);
   grate_fp_info_fini(&fp->info);
}

static void
grate_fs_translate(struct grate_fp_shader *fp, const struct tgsi_token *tokens,
                   const struct grate_fp_key *key, const struct grate_fp_opts *opts)
{
   struct tgsi_parse_context parser;
   unsigned ok = tgsi_parse_init(&parser, tokens);
   assert(ok == TGSI_PARSE_OK);
   (void)ok;
   grate_tgsi_to_fp(fp, &parser, key, opts);
   tgsi_parse_free(&parser);
}

/* Compile a fragment program variant into its command stream */
/*
 * The fragment program's code for EXECs e0 up to e1: PSEQ, MFU (AT), TEX,
 * ALU and DW instructions, the MFU and ALU addresses taken from the first
 * of those EXECs that has any (a PSEQ pass loads its own code at 0)
 */
static void
grate_fp_push_code(struct util_dynarray *buf, struct fp_instr **ins, int e0, int e1,
                   struct fp_mfu_instr **mfu, struct fp_alu_instr_packet **alu,
                   enum drm_tegra_soc_id soc_id)
{
#define PUSHB(x) util_dynarray_append_typed(buf, uint32_t, (x))
   int n = e1 - e0, mfu0 = -1, mfu1 = 0, alu0 = -1, alu1 = 0;
   for (int e = e0; e < e1; e++) {
      struct fp_sched *m = &ins[e]->mfu_sched, *a = &ins[e]->alu_sched;
      if (m->num_instructions) {
         if (mfu0 < 0)
            mfu0 = m->address;
         mfu1 = m->address + m->num_instructions;
      }
      if (a->num_instructions) {
         if (alu0 < 0)
            alu0 = a->address;
         alu1 = a->address + a->num_instructions;
      }
   }
   mfu0 = MAX2(mfu0, 0);
   alu0 = MAX2(alu0, 0);
   assert(n <= 64 && mfu1 - mfu0 <= 64 && alu1 - alu0 <= 64);

   PUSHB(host1x_opcode_nonincr(REG_TGR3D_PSEQ_INST_DATA, n));
   for (int e = e0; e < e1; e++)
      PUSHB(ins[e]->pseq);

   PUSHB(host1x_opcode_nonincr(REG_TGR3D_AT_REMAP_DATA, n));
   for (int e = e0; e < e1; e++) {
      struct fp_sched m = ins[e]->mfu_sched;
      if (m.num_instructions)
         m.address -= mfu0;
      PUSHB(grate_fp_pack_sched(&m));
   }

   PUSHB(host1x_opcode_nonincr(REG_TGR3D_AT_INST_DATA_LO, (mfu1 - mfu0) * 2));
   for (int i = mfu0; i < mfu1; i++) {
      uint32_t words[2];
      grate_fp_pack_mfu(words, mfu[i]);
      PUSHB(words[0]);
      PUSHB(words[1]);
   }

   PUSHB(host1x_opcode_nonincr(REG_TGR3D_TEX_INST_DATA, n));
   for (int e = e0; e < e1; e++)
      PUSHB(ins[e]->tex);

   PUSHB(host1x_opcode_nonincr(REG_TGR3D_ALU_REMAP_DATA, n));
   for (int e = e0; e < e1; e++) {
      struct fp_sched a = ins[e]->alu_sched;
      if (a.num_instructions)
         a.address -= alu0;
      if (soc_id == DRM_TEGRA_SOC_T114)
         PUSHB(grate_fp_pack_alu_sched_t114(&a));
      else
         PUSHB(grate_fp_pack_sched(&a));
   }

   PUSHB(host1x_opcode_nonincr(REG_TGR3D_ALU_INST_DATA, (alu1 - alu0) * 4 * 2));
   for (int p = alu0; p < alu1; p++) {
      struct fp_alu_instr_packet *instr = alu[p];
      for (int i = 0; i < 4; ++i) {
         uint32_t words[2];
         if (i == 3 && instr->num_imm) {
            /* imm0 at bits 4..23 of the first word, imm1 across bits 24..31
             * and the next word's 0..11, imm2 at 12..31 (libgrate's layout) */
            const uint32_t *imm = instr->imm;
            words[0] = imm[0] << 4 | (imm[1] & 0xff) << 24;
            words[1] = imm[1] >> 8 | imm[2] << 12;
         } else {
            grate_fp_pack_alu(words, instr->slots + i);
         }
         PUSHB(words[0]);
         PUSHB(words[1]);
      }
   }

   PUSHB(host1x_opcode_nonincr(REG_TGR3D_ALU_P2CX_DATA, n));
   for (int e = e0; e < e1; e++)
      PUSHB(0x00000000);

   PUSHB(host1x_opcode_nonincr(REG_TGR3D_DW_INST_DATA, n));
   for (int e = e0; e < e1; e++)
      PUSHB(grate_fp_pack_dw(&ins[e]->dw));
#undef PUSHB
}

/*
 * A program of several PSEQ passes (info.chunk_exec): what the PSEQ fetches
 * for each (blob.chunk_words), as the NVIDIA compiler lays out its chunks
 * (libcgdrv.so 0xdcc80): the upload counters cleared, then the pass's
 * code. grate_state.c emit_program points the PSEQ commands at them.
 */
static void
grate_fp_chunks(struct grate_shader_blob *blob, const struct grate_fp_info *info,
                struct fp_instr **ins, struct fp_mfu_instr **mfu,
                struct fp_alu_instr_packet **alu, enum drm_tegra_soc_id soc_id)
{
   struct util_dynarray buf;
   util_dynarray_init(&buf, NULL);
   blob->num_chunks = info->num_chunks;
   for (int k = 0; k < info->num_chunks; k++) {
      int e0 = info->chunk_exec[k], e1 = info->chunk_exec[k + 1];
      blob->chunk_off[k] = buf.size / sizeof(uint32_t);
      blob->chunk_execs[k] = e1 - e0;
      util_dynarray_append_typed(&buf, uint32_t, host1x_opcode_incr(REG_TGR3D_GLOBAL_INST_OFFSET, 1));
      util_dynarray_append_typed(&buf, uint32_t, 0);
      util_dynarray_append_typed(&buf, uint32_t, host1x_opcode_incr(REG_TGR3D_AT_INST_OFFSET, 1));
      util_dynarray_append_typed(&buf, uint32_t, 0);
      util_dynarray_append_typed(&buf, uint32_t, host1x_opcode_incr(REG_TGR3D_ALU_INST_OFFSET, 1));
      util_dynarray_append_typed(&buf, uint32_t, 0);
      util_dynarray_append_typed(&buf, uint32_t, host1x_opcode_incr(REG_TGR3D_PSEQ_FLUSH, 1));
      util_dynarray_append_typed(&buf, uint32_t, 0);
      grate_fp_push_code(&buf, ins, e0, e1, mfu, alu, soc_id);
      blob->chunk_len[k] = buf.size / sizeof(uint32_t) - blob->chunk_off[k];
   }
   util_dynarray_trim(&buf);
   blob->chunk_words = buf.data;
   blob->num_chunk_words = buf.size / sizeof(uint32_t);
}

/* Compile with alt_opts too and keep that version if it takes fewer EXECs,
 * or as many on fewer rows: what the frame time follows (glmark2 blur on the
 * Xoom: 33 EXECs 8 fps, 24 EXECs on more rows 12 fps; 21 EXECs on 3 rows 12
 * fps, 22 on 2 rows 10 fps) */
static void
grate_fs_try(struct grate_fp_shader *fp, struct grate_fp_opts *opts,
             const struct tgsi_token *tokens, const struct grate_fp_key *key,
             const struct grate_fp_opts *alt_opts)
{
   struct grate_fp_shader alt;
   grate_fs_translate(&alt, tokens, key, alt_opts);
   if (!alt.info.failed &&
       (alt.info.num_execs < fp->info.num_execs ||
        (alt.info.num_execs == fp->info.num_execs && alt.info.num_rows < fp->info.num_rows))) {
      grate_fs_free(fp);
      *fp = alt;
      /* the lists' heads moved with the struct */
      list_replace(&alt.fp_instructions, &fp->fp_instructions);
      list_replace(&alt.alu_instructions, &fp->alu_instructions);
      list_replace(&alt.mfu_instructions, &fp->mfu_instructions);
      *opts = *alt_opts;
   } else {
      grate_fs_free(&alt);
   }
}

/* quick: without the slow last tries (seq_fallbacks, seq_chunks), for a
 * variant only worth having when it fits as easily as the one it would
 * replace (texture coordinates moved to the vertex shader) */
static void
grate_fs_compile(enum drm_tegra_soc_id soc_id, const struct tgsi_token *tokens,
                 const struct grate_fp_key *key, struct grate_fs_variant *v,
                 bool quick)
{
   struct grate_fp_shader fp;
   struct grate_fp_opts opts = {
      .max_pre = GRATE_FP_MAX_PRE,
      /* TEST: passes forced (sched_seq.c seq_chunk_execs) */
      .seq_chunks = getenv("GRATE_FP_CHUNK_EXECS") != NULL,
   };
   grate_fs_translate(&fp, tokens, key, &opts);

   /*
    * Compiled again when it fails:
    * - for the uniform registers, with half the preshader entries it took:
    *   each result read takes a register, the work left in the program
    *   may read fewer;
    * - TRUNC of a varying is compiled as fract(x) so that a multi-row
    *   program can run the SFU on the interpolated register; a sequential
    *   program has fewer registers for its extra values and may only fit
    *   with |x|;
    * - a dot product as a chain of MADs: more packets, but fewer registers
    *   at a time.
    */
   while (fp.info.failed) {
      if (fp.info.out_of_uniforms && fp.info.num_pre)
         opts.max_pre = fp.info.num_pre / 2;
      else if (fp.info.varying_trunc && !opts.trunc_abs)
         opts.trunc_abs = true;
      else if (fp.info.dots && !opts.split_dots)
         opts.split_dots = true;
      else if (fp.info.seq_too_big && !opts.seq_fallbacks && !quick &&
               /* not for a generic version specialized at draw time */
               fp.info.num_flow == (key ? key->num_flow : 0) &&
               (!fp.info.num_spec || (key && key->spec)))
         opts.seq_fallbacks = true;   /* slow */
      else if (fp.info.seq_too_big && !opts.seq_chunks && !quick &&
               fp.info.num_flow == (key ? key->num_flow : 0) &&
               (!fp.info.num_spec || (key && key->spec)))
         opts.seq_chunks = true;      /* several PSEQ passes: last */
      else
         break;
      grate_fs_free(&fp);
      grate_fs_translate(&fp, tokens, key, &opts);
   }

   /* a sequential program that fits on more rows with fract(x) may fit on
    * fewer with |x|: fewer rows, more pixels in flight */
   if (!fp.info.failed && fp.info.sequential && fp.info.num_rows > 1 &&
       fp.info.varying_trunc && !opts.trunc_abs) {
      struct grate_fp_shader alt;
      struct grate_fp_opts alt_opts = opts;
      alt_opts.trunc_abs = true;
      grate_fs_translate(&alt, tokens, key, &alt_opts);
      if (!alt.info.failed && alt.info.num_rows < fp.info.num_rows) {
         grate_fs_free(&fp);
         fp = alt;
         /* the lists' heads moved with the struct */
         list_replace(&alt.fp_instructions, &fp.fp_instructions);
         list_replace(&alt.alu_instructions, &fp.alu_instructions);
         list_replace(&alt.mfu_instructions, &fp.mfu_instructions);
         opts = alt_opts;
      } else {
         grate_fs_free(&alt);
      }
   }

   /*
    * Interpolating a shared varying coordinate again for each fetch holds
    * no register before its TEX (glmark2 bump: 3 rows, 27 EXECs -> 2, 22),
    * but the schedule may also come out longer than with copies (blur:
    * 24 -> 40 EXECs): both (grate_fs_try)
    */
   if (!fp.info.failed && fp.info.input_aliases && !opts.no_alias) {
      struct grate_fp_opts alt_opts = opts;
      alt_opts.no_alias = true;
      grate_fs_try(&fp, &opts, tokens, key, &alt_opts);
   }

   /*
    * Critical path first comes out shorter for some programs, longer for
    * others (glmark2 shading cel 28 -> 21 EXECs, 30 -> 32 fps; blur 21 on
    * 3 rows -> 22 on 2): both (grate_fs_try), when the schedule is more than
    * a round of its rows longer than the work needs (info.seq_bound; the
    * fphost corpus: 8 programs shorter, none longer, 1150 EXECs instead of
    * 1187)
    */
   if (!fp.info.failed && fp.info.sequential && !quick && !opts.seq_prio &&
       fp.info.num_execs > fp.info.seq_bound + fp.info.num_rows) {
      struct grate_fp_opts alt_opts = opts;
      alt_opts.seq_prio = true;
      grate_fs_try(&fp, &opts, tokens, key, &alt_opts);
   }

   /* say which shader draws magenta, to have something to work from (not
    * the generic version of one that gets specialized at draw time) */
   if (fp.info.failed && fp.info.num_flow == (key ? key->num_flow : 0) &&
       (!fp.info.num_spec || (key && key->spec))) {
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
   int num_fp_instrs = list_length(&fp.fp_instructions);
   int num_mfu_instrs = list_length(&fp.mfu_instructions);
   int num_alu_instrs = list_length(&fp.alu_instructions);
   struct fp_instr **ins = MALLOC(sizeof(*ins) * MAX2(num_fp_instrs, 1));
   struct fp_mfu_instr **mfu = MALLOC(sizeof(*mfu) * MAX2(num_mfu_instrs, 1));
   struct fp_alu_instr_packet **alu = MALLOC(sizeof(*alu) * MAX2(num_alu_instrs, 1));
   int n = 0;
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      ins[n++] = instr;
   n = 0;
   list_for_each_entry(struct fp_mfu_instr, instr, &fp.mfu_instructions, link)
      mfu[n++] = instr;
   n = 0;
   list_for_each_entry(struct fp_alu_instr_packet, instr, &fp.alu_instructions, link)
      alu[n++] = instr;

   int num_chunks = fp.info.failed ? 1 : MAX2(fp.info.num_chunks, 1);
   if (num_chunks == 1) {
      PUSH(host1x_opcode_imm(REG_TGR3D_GLOBAL_INST_OFFSET, 0));
      PUSH(host1x_opcode_imm(REG_TGR3D_AT_INST_OFFSET, 0));
      PUSH(host1x_opcode_imm(REG_TGR3D_ALU_INST_OFFSET, 0));

      assert(num_fp_instrs <= 64);

      PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_COMMAND_EVEN(0), 1));
      /* run the program's EXECs as the only pass (see grate_fp_chunks) */
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
      grate_fp_push_code(&buf, ins, 0, num_fp_instrs, mfu, alu, soc_id);
   } else {
      /* several PSEQ passes: the store waits for the last one's EXECs */
      int last = fp.info.chunk_exec[num_chunks] - fp.info.chunk_exec[num_chunks - 1];
      PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_DWR_IF_STATE, 1));
      PUSH(last << 6);
      grate_fp_chunks(&v->blob, &fp.info, ins, mfu, alu, soc_id);
   }
   FREE(ins);
   FREE(mfu);
   FREE(alu);

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

/*
 * Fragment program variants in the disk cache (grate_screen.disk_cache):
 * the compile takes 1-2 s for big programs on the Xoom, the second run of
 * an application reads them back. Keyed by the TGSI, the variant key, the
 * SoC and the compiler's test variables (GRATE_FP_*: experiments must not
 * read each other's programs); the build id names the cache itself.
 */
#define GRATE_FS_CACHE_MAGIC 0x31534647   /* "GFS1" */

static const char *const grate_fs_cache_env[] = {
   "GRATE_FP_ALL_PACKETS", "GRATE_FP_CHUNK_EXECS", "GRATE_FP_CR", "GRATE_FP_FX10",
   "GRATE_FP_INPLACE", "GRATE_FP_NO_EMBED", "GRATE_FP_NO_MINUS_ONE", "GRATE_FP_ROWS",
   "GRATE_FP_SEQ", "GRATE_FP_SEQ_ROWS", "GRATE_FP_SUM_ORDER",
};

static void
grate_fs_cache_key(struct disk_cache *cache, enum drm_tegra_soc_id soc_id,
                   const struct tgsi_token *tokens, const struct grate_fp_key *key,
                   bool quick, cache_key ck)
{
   struct blob b;
   blob_init(&b);
   blob_write_uint32(&b, GRATE_FS_CACHE_MAGIC);
   blob_write_uint32(&b, soc_id);
   blob_write_uint8(&b, quick);
   blob_write_bytes(&b, key, sizeof(*key));
   blob_write_bytes(&b, tokens, tgsi_num_tokens(tokens) * sizeof(struct tgsi_token));
   for (unsigned i = 0; i < ARRAY_SIZE(grate_fs_cache_env); i++) {
      const char *v = getenv(grate_fs_cache_env[i]);
      blob_write_string(&b, v ? v : "");
   }
   disk_cache_compute_key(cache, b.data, b.size, ck);
   blob_finish(&b);
}

static void
grate_fs_cache_store(struct disk_cache *cache, const cache_key ck,
                     const struct grate_fs_variant *v)
{
   const struct grate_shader_blob *s = &v->blob;
   struct grate_fp_info info = v->info;
   struct blob b;

   info.pre = NULL;
   blob_init(&b);
   blob_write_uint32(&b, GRATE_FS_CACHE_MAGIC);
   blob_write_bytes(&b, &info, sizeof(info));
   blob_write_bytes(&b, v->info.pre, v->info.num_pre * sizeof(v->info.pre[0]));
   blob_write_uint32(&b, s->num_commands);
   blob_write_bytes(&b, s->commands, s->num_commands * sizeof(uint32_t));
   blob_write_uint32(&b, s->num_chunks);
   blob_write_uint32(&b, s->num_chunk_words);
   blob_write_bytes(&b, s->chunk_words, s->num_chunk_words * sizeof(uint32_t));
   blob_write_bytes(&b, s->chunk_off, sizeof(s->chunk_off));
   blob_write_bytes(&b, s->chunk_len, sizeof(s->chunk_len));
   blob_write_bytes(&b, s->chunk_execs, sizeof(s->chunk_execs));
   if (!b.out_of_memory)
      disk_cache_put(cache, ck, b.data, b.size, NULL);
   blob_finish(&b);
}

static bool
grate_fs_cache_load(struct disk_cache *cache, const cache_key ck,
                    const struct grate_fp_key *key, struct grate_fs_variant *v)
{
   size_t size;
   void *data = disk_cache_get(cache, ck, &size);
   if (!data)
      return false;

   struct grate_shader_blob *s = &v->blob;
   struct blob_reader r;
   memset(v, 0, sizeof(*v));
   blob_reader_init(&r, data, size);
   bool ok = blob_read_uint32(&r) == GRATE_FS_CACHE_MAGIC;
   if (ok)
      blob_copy_bytes(&r, &v->info, sizeof(v->info));
   v->info.pre = NULL;
   if (ok && v->info.num_pre > 0 && v->info.num_pre <= GRATE_FP_MAX_PRE) {
      v->info.pre = MALLOC(v->info.num_pre * sizeof(v->info.pre[0]));
      blob_copy_bytes(&r, v->info.pre, v->info.num_pre * sizeof(v->info.pre[0]));
   } else if (v->info.num_pre) {
      ok = false;
   }
   if (ok) {
      s->num_commands = blob_read_uint32(&r);
      s->commands = MALLOC(MAX2(s->num_commands, 1) * sizeof(uint32_t));
      blob_copy_bytes(&r, s->commands, s->num_commands * sizeof(uint32_t));
      s->num_chunks = blob_read_uint32(&r);
      s->num_chunk_words = blob_read_uint32(&r);
      if (s->num_chunk_words) {
         s->chunk_words = MALLOC(s->num_chunk_words * sizeof(uint32_t));
         blob_copy_bytes(&r, s->chunk_words, s->num_chunk_words * sizeof(uint32_t));
      }
      blob_copy_bytes(&r, s->chunk_off, sizeof(s->chunk_off));
      blob_copy_bytes(&r, s->chunk_len, sizeof(s->chunk_len));
      blob_copy_bytes(&r, s->chunk_execs, sizeof(s->chunk_execs));
      ok = !r.overrun && r.current == r.end;
   }
   free(data);
   if (!ok) {
      FREE(v->info.pre);
      FREE(s->commands);
      FREE(s->chunk_words);
      memset(v, 0, sizeof(*v));
      return false;
   }
   v->key = *key;
   return true;
}

/* grate_fs_compile, through the disk cache (not with GRATE_DEBUG=tgsi,
 * whose dumps want the compile) */
static void
grate_fs_compile_cached(struct grate_context *context, const struct tgsi_token *tokens,
                        const struct grate_fp_key *key, struct grate_fs_variant *v,
                        bool quick)
{
   struct disk_cache *cache = grate_screen(context->base.screen)->disk_cache;
   enum drm_tegra_soc_id soc_id = context->drm->soc_id;
   cache_key ck;

   if (grate_debug & GRATE_DEBUG_TGSI)
      cache = NULL;
   if (cache) {
      grate_fs_cache_key(cache, soc_id, tokens, key, quick, ck);
      if (grate_fs_cache_load(cache, ck, key, v))
         return;
   }
   grate_fs_compile(soc_id, tokens, key, v, quick);
   if (cache)
      grate_fs_cache_store(cache, ck, v);
}

static void grate_fs_upload_chunks(struct grate_context *context,
                                   struct grate_shader_blob *blob);

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
   so->reads_z = scan.reads_z;

   /* the opaque variant, used without blending */
   grate_fs_compile_cached(context, so->base.tokens, &grate_fp_key_none,
                           &so->variants[0], false);
   grate_fs_upload_chunks(context, &so->variants[0].blob);
   so->num_variants = 1;
   so->blob = so->variants[0].blob;
   so->info = &so->variants[0].info;
   return so;
}

/* Variant key of the bound blend state (render target 0) */
/*
 * Whether sampler view + state repeat a non-power-of-two texture along s
 * (axis 0) or t: measured on Tegra20 (gpu/tools/xrtest/xsmalltile), such a
 * texture does not wrap right whatever axis is non-power-of-two, so the
 * program wraps the coordinate itself and the TEX unit clamps.
 */
bool
grate_npot_repeat(const struct pipe_sampler_view *view,
                  const struct pipe_sampler_state *sampler, int axis)
{
   if (!view || !view->texture || !sampler)
      return false;
   unsigned wrap = axis ? sampler->wrap_t : sampler->wrap_s;
   /* the size of the level the view starts at (emit_textures) */
   unsigned first = view->target == PIPE_TEXTURE_CUBE ? 0 : view->u.tex.first_level;
   return wrap == PIPE_TEX_WRAP_REPEAT &&
          !(util_is_power_of_two_nonzero(u_minify(view->texture->width0, first)) &&
            util_is_power_of_two_nonzero(u_minify(view->texture->height0, first)));
}

/*
 * Whether a sampler view + state is read from a power-of-two copy
 * (grate_resource_npot_shadow): a 2D non-power-of-two texture minified and
 * magnified with different filters, which the TEX unit picks only for
 * power-of-two ones. Not mirrored (that would turn at the copy's edge) nor
 * compressed (the 2D engine copies bytes).
 */
bool
grate_npot_shadow(const struct pipe_sampler_view *view,
                  const struct pipe_sampler_state *sampler)
{
   if (!view || !view->texture || !sampler || view->target != PIPE_TEXTURE_2D)
      return false;
   const struct pipe_resource *t = view->texture;
   return !(util_is_power_of_two_nonzero(t->width0) &&
            util_is_power_of_two_nonzero(t->height0)) &&
          sampler->min_img_filter != sampler->mag_img_filter &&
          view->u.tex.first_level == 0 &&
          sampler->wrap_s != PIPE_TEX_WRAP_MIRROR_REPEAT &&
          sampler->wrap_t != PIPE_TEX_WRAP_MIRROR_REPEAT &&
          !util_format_is_compressed(view->format) &&
          !util_format_is_depth_or_stencil(view->format);
}

static struct grate_fp_key
grate_fs_key(struct grate_context *context)
{
   const struct pipe_blend_state *blend = context->blend;
   struct grate_fp_key key;

   memset(&key, 0, sizeof(key));
   key.colormask = 0xf;

   if (context->fshader && context->fshader->reads_z) {
      key.depth[0] = context->depth_scale;
      key.depth[1] = context->depth_translate;
   }

   for (unsigned i = 0; i < GRATE_FP_MAX_SWIZZLED; i++) {
      const struct pipe_sampler_view *view = context->sampler_views[i];
      if (!view)
         continue;
      /* the view's swizzle, then where the TEX unit puts the channels */
      unsigned sw[4] = { view->swizzle_r, view->swizzle_g, view->swizzle_b,
                         view->swizzle_a };
      unsigned bits = 0;
      /* the X of RGBX formats: the TEX unit reads the byte as alpha, the
       * format means one (the GL frontend swizzles it away; u_blitter, e.g.
       * glReadPixels of an RGB framebuffer, does not) */
      bool x8 = view->format == PIPE_FORMAT_R8G8B8X8_UNORM ||
                view->format == PIPE_FORMAT_B8G8R8X8_UNORM;
      for (int c = 0; c < 4; c++) {
         if (x8 && sw[c] == PIPE_SWIZZLE_W)
            sw[c] = PIPE_SWIZZLE_1;
         if (sw[c] <= PIPE_SWIZZLE_W)
            sw[c] = grate_sampler_channel(view->format, sw[c]);
         bits |= sw[c] << (3 * c);
      }
      key.tex_swizzle[i] = bits ^ GRATE_SWIZZLE_IDENTITY;

      if (grate_npot_repeat(view, context->samplers[i], 0))
         key.npot_s |= 1u << i;
      if (grate_npot_repeat(view, context->samplers[i], 1))
         key.npot_t |= 1u << i;
      if (view->format == PIPE_FORMAT_Z16_UNORM ||
          view->format == PIPE_FORMAT_Z24_UNORM_S8_UINT)
         key.depth_tex |= 1u << i;
      if (grate_npot_shadow(view, context->samplers[i])) {
         key.npot_shadow |= 1u << i;
         key.shadow_size[i][0] = view->texture->width0;
         key.shadow_size[i][1] = view->texture->height0;
      }
   }

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

      const uint8_t factors[] = { key.rgb_src, key.rgb_dst, key.alpha_src, key.alpha_dst };
      for (unsigned i = 0; i < ARRAY_SIZE(factors); i++) {
         unsigned f = util_blendfactor_without_invert(factors[i]);
         if (f == PIPE_BLENDFACTOR_CONST_COLOR || f == PIPE_BLENDFACTOR_CONST_ALPHA)
            memcpy(key.blend_color, context->blend_color.color, sizeof(key.blend_color));
      }
   }
   return key;
}

/* Several PSEQ passes: the chunks into GPU memory, for the PSEQ to fetch */
static void
grate_fs_upload_chunks(struct grate_context *context, struct grate_shader_blob *blob)
{
   void *map;

   if (blob->num_chunks < 2)
      return;
   blob->chunk_bo = grate_bo_alloc(context->drm, blob->num_chunk_words * 4, 0);
   if (blob->chunk_bo && !grate_bo_map(blob->chunk_bo, &map)) {
      memcpy(map, blob->chunk_words, blob->num_chunk_words * 4);
      grate_bo_unmap(blob->chunk_bo);
      return;
   }
   fprintf(stderr, "GRATE FRAG: no memory for %d PSEQ passes\n", blob->num_chunks);
   if (blob->chunk_bo)
      grate_bo_unref(blob->chunk_bo);
   blob->chunk_bo = NULL;
}

/* A fragment variant's program (the stream may still hold its chunks: the
 * BO goes when the job is done with it) */
static void
grate_fs_blob_free(struct grate_shader_blob *blob)
{
   FREE(blob->commands);
   FREE(blob->chunk_words);
   if (blob->chunk_bo)
      grate_bo_unref(blob->chunk_bo);
   memset(blob, 0, sizeof(*blob));
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
      grate_fs_blob_free(&so->variants[i].blob);
      grate_fp_info_fini(&so->variants[i].info);
   }
   grate_fs_compile_cached(context, so->base.tokens, key, &so->variants[i],
                           key->move && so->move_quick);
   grate_fs_upload_chunks(context, &so->variants[i].blob);
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
   /* loops counted by uniforms: the variant for their values, which may
    * need more of them (at most GRATE_FP_MAX_FLOW, so this ends) */
   while (v->info.num_flow > key.num_flow) {
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
      struct grate_fp_key vkey = v->key;
      key.move = 1;
      so->move_quick = !v->info.failed;
      struct grate_fs_variant *mv = grate_fs_select(context, so, key, user, user_len);
      struct grate_vs_variant *mvv = !mv->info.failed && mv->info.moved.num_outs ?
                                     grate_vs_variant(vs, &mv->info.moved) : NULL;
      if (mvv) {
         v = mv;
         vv = mvv;
      } else if (memcmp(&v->key, &vkey, sizeof(vkey))) {
         /* compiling those replaced it */
         v = grate_fs_variant(context, so, &vkey);
      }
   }

   so->blob = v->blob;
   so->info = &v->info;
   if (vv)
      grate_vs_use(vs, vv);
}

void
grate_fs_consts(struct grate_context *context, uint32_t *consts)
{
   const struct grate_fp_info *fpinfo = context->fshader->info;
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
   for (unsigned i = 0; i < so->num_variants; i++) {
      grate_fs_blob_free(&so->variants[i].blob);
      grate_fp_info_fini(&so->variants[i].info);
   }
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
