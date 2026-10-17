#include <math.h>
#include <string.h>

#include "grate_common.h"
#include "grate_compiler.h"
#include "fpir.h"

#include "tgsi/tgsi_parse.h"

#include "util/u_memory.h"

static struct fp_alu_src_operand
fp_alu_src_row(int index)
{
   assert(index >= 0 && index < 16);
   struct fp_alu_src_operand src = {
      .index = index
   };
   return src;
}

static struct fp_alu_src_operand
fp_alu_src_reg(int index)
{
   assert(index >= 0 && index < 8);
   struct fp_alu_src_operand src = {
      .index = 16 + index
   };
   return src;
}

static struct fp_alu_src_operand
fp_alu_src_uniform(int index)
{
   assert(index >= 0 && index < GRATE_FP_NUM_CONSTS);
   struct fp_alu_src_operand src = {
      .index = 32 + index,
      .datatype = FP_DATATYPE_FP20
   };
   return src;
}

uint32_t
grate_float_to_fp20(float f)
{
   union { uint32_t u; float f; } v = { .f = f };
   uint32_t sign, exponent, mantissa;

   if (f == 0.0f)
      return 0;

   sign = (v.u >> 31) & 0x1;
   exponent = (v.u >> 23) & 0xff;
   mantissa = v.u & 0x7fffff;

   if (exponent == 0xff)
      exponent = 0x3f;
   else
      exponent = (exponent - 127 + 31) & 0x3f;

   return (sign << 19) | (exponent << 13) | (mantissa >> (23 - 13));
}

/* Allocate (or reuse) an ALU uniform register holding an fp20 immediate */
static int
fp_const_slot(struct grate_fp_shader *fp, float value)
{
   uint32_t fp20 = grate_float_to_fp20(value);
   int i;

   for (i = GRATE_FP_NUM_CONSTS - 1; i >= 0; i--)
      if ((fp->info.imm_mask & (1u << i)) && fp->info.consts[i] == fp20)
         return i;

   for (i = GRATE_FP_NUM_CONSTS - 1; i >= 0; i--)
      if (!(fp->info.used_mask & (1u << i)))
         break;
   assert(i >= 0);

   fp->info.consts[i] = fp20;
   fp->info.imm_mask |= 1u << i;
   fp->info.used_mask |= 1u << i;
   return i;
}

/* ALU uniform register of a user uniform component */
static int
fp_uniform_slot(struct grate_fp_shader *fp, int index, int comp)
{
   int slot = index * 4 + comp;

   assert(slot < GRATE_FP_NUM_CONSTS);
   assert(!(fp->info.imm_mask & (1u << slot)));
   fp->info.used_mask |= 1u << slot;
   return slot;
}

static struct fp_alu_src_operand
fp_alu_src_zero()
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = 0
   };
   return src;
}

static struct fp_alu_src_operand
fp_alu_src_one()
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = 1
   };
   return src;
}

/* An ALU slot that does nothing: writes vec2(0, 1) to the read-only r31 */
static struct fp_alu_instr
fp_alu_nop(void)
{
   struct fp_alu_instr ret = {
      .op = FP_ALU_OP_MAD,
      .dst = { .index = 31 },
      .src = {
         fp_alu_src_zero(),
         fp_alu_src_zero(),
         fp_alu_src_zero(),
         fp_alu_src_zero()
      }
   };
   return ret;
}

/*
 * One TGSI instruction is compiled into one "batch": an optional MFU
 * instruction fetching varyings into row registers r0..r3, one ALU packet
 * of up to four scalar slots, and an optional DW (store) instruction.
 * Row and global registers keep their values from batch to batch.
 */
struct fp_batch {
   struct grate_fp_shader *fp;
   struct fp_instr *inst;
   struct fp_mfu_instr *mfu;
   struct fp_alu_instr_packet *alu;
   int input;      /* TRAM row fetched by this batch's MFU, -1 if none */
};

static void
fp_batch_begin(struct fp_batch *b, struct grate_fp_shader *fp)
{
   b->fp = fp;
   b->inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&b->inst->link);
   b->alu = CALLOC_STRUCT(fp_alu_instr_packet);
   list_inithead(&b->alu->link);
   for (int i = 0; i < 4; i++)
      b->alu->slots[i] = fp_alu_nop();
   b->mfu = NULL;
   b->input = -1;
}

static void
fp_batch_end(struct fp_batch *b)
{
   struct grate_fp_shader *fp = b->fp;

   if (b->mfu) {
      b->inst->mfu_sched.num_instructions = 1;
      b->inst->mfu_sched.address = list_length(&fp->mfu_instructions);
      list_addtail(&b->mfu->link, &fp->mfu_instructions);
   }

   b->inst->alu_sched.num_instructions = 1;
   b->inst->alu_sched.address = list_length(&fp->alu_instructions);
   list_addtail(&b->alu->link, &fp->alu_instructions);

   list_addtail(&b->inst->link, &fp->fp_instructions);
}

/* Temporaries live in the 8 global registers: TEMP[i].c -> g(i * 4 + c) */
static int
fp_temp_reg(int index, int comp)
{
   int reg = index * 4 + comp;

   if (reg >= 8) {
      fprintf(stderr, "GRATE FRAG: out of temporaries (TEMP[%d])\n", index);
      assert(0);
      reg = 7;
   }
   return reg;
}

static struct fp_alu_src_operand
fp_src(struct fp_batch *b, const struct tgsi_src_register *src, int chan)
{
   struct grate_fp_shader *fp = b->fp;
   struct fp_alu_src_operand ret = { 0 };
   const int swizzle[] = {
      src->SwizzleX, src->SwizzleY, src->SwizzleZ, src->SwizzleW
   };
   int comp = swizzle[chan];

   switch (src->File) {
   case TGSI_FILE_INPUT:
      /* MFU var[c] interpolates TRAM row "index" component c into row c */
      if (!b->mfu) {
         b->mfu = CALLOC_STRUCT(fp_mfu_instr);
         list_inithead(&b->mfu->link);
      }
      if (b->input >= 0 && b->input != src->Index) {
         fprintf(stderr, "GRATE FRAG: two varyings in one instruction\n");
         assert(0);
      }
      b->input = src->Index;
      b->mfu->var[comp].op = FP_VAR_OP_FP20;
      b->mfu->var[comp].tram_row = src->Index;
      fp->info.max_tram_row = MAX2(fp->info.max_tram_row, src->Index);
      ret = fp_alu_src_row(comp);
      break;

   case TGSI_FILE_IMMEDIATE: {
      assert(src->Index < fp->num_immediates);
      float value = fp->immediates[src->Index][comp];
      if (src->Absolute)
         value = fabsf(value);
      if (src->Negate)
         value = -value;
      return fp_alu_src_uniform(fp_const_slot(fp, value));
   }

   case TGSI_FILE_CONSTANT:
      ret = fp_alu_src_uniform(fp_uniform_slot(fp, src->Index, comp));
      break;

   case TGSI_FILE_TEMPORARY:
      ret = fp_alu_src_reg(fp_temp_reg(src->Index, comp));
      break;

   default:
      fprintf(stderr, "GRATE FRAG: unsupported source file %d\n", src->File);
      assert(0);
   }

   ret.negate = src->Negate;
   ret.absolute_value = src->Absolute;
   return ret;
}

static struct fp_alu_dst_operand
fp_dst(struct fp_batch *b, const struct tgsi_dst_register *dst, int chan,
       bool saturate)
{
   struct fp_alu_dst_operand ret = { 0 };

   switch (dst->File) {
   case TGSI_FILE_OUTPUT: {
      /* HACK: colour goes to r2/r3 as fixed10 BGRA, matching the
       * hard-coded store below */
      int o = chan < 3 ? (2 - chan) : 3;
      ret.index = 2 + o / 2;
      ret.write_low_sub_reg = (o % 2) == 0;
      ret.write_high_sub_reg = (o % 2) != 0;

      b->inst->dw.enable = 1;
      b->inst->dw.index = dst->Index;
      b->inst->dw.stencil_write = 0;
      b->inst->dw.src_regs = FP_DW_REGS_R2_R3;
      break;
   }

   case TGSI_FILE_TEMPORARY:
      ret.index = 16 + fp_temp_reg(dst->Index, chan);
      /* fp20 write: both halves */
      ret.write_low_sub_reg = true;
      ret.write_high_sub_reg = true;
      break;

   default:
      fprintf(stderr, "GRATE FRAG: unsupported destination file %d\n", dst->File);
      assert(0);
   }

   ret.saturate = saturate;
   return ret;
}

/* Source operand reading back what fp_dst() wrote for "chan" */
static struct fp_alu_src_operand
fp_src_of_dst(const struct tgsi_dst_register *dst, int chan)
{
   struct fp_alu_src_operand ret = { 0 };

   if (dst->File == TGSI_FILE_OUTPUT) {
      int o = chan < 3 ? (2 - chan) : 3;
      ret.index = 2 + o / 2;
      ret.datatype = FP_DATATYPE_FIXED10;
      ret.sub_reg_select_high = (o % 2) != 0;
   } else
      ret = fp_alu_src_reg(fp_temp_reg(dst->Index, chan));
   return ret;
}

static struct fp_alu_instr
fp_alu(enum fp_alu_op op, struct fp_alu_dst_operand dst,
       struct fp_alu_src_operand a, struct fp_alu_src_operand b,
       struct fp_alu_src_operand c)
{
   /* rD is left disabled: result = op(rA * rB, rC) */
   struct fp_alu_instr ret = {
      .op = op,
      .dst = dst,
      .src = { a, b, c, c }
   };
   return ret;
}

/* Component-wise instruction: one ALU slot per written channel */
static void
emit_componentwise(struct grate_fp_shader *fp,
                   const struct tgsi_full_instruction *inst)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   bool sat = inst->Instruction.Saturate != 0;
   struct fp_batch b;

   fp_batch_begin(&b, fp);
   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;

      struct fp_alu_dst_operand d = fp_dst(&b, dst, i, sat);
      struct fp_alu_src_operand s0 = fp_src(&b, &inst->Src[0].Register, i);
      struct fp_alu_src_operand s1, s2;
      struct fp_alu_instr *slot = &b.alu->slots[i];

      switch (inst->Instruction.Opcode) {
      case TGSI_OPCODE_MOV:
         *slot = fp_alu(FP_ALU_OP_MAD, d, s0, fp_alu_src_one(), fp_alu_src_zero());
         break;
      case TGSI_OPCODE_MUL:
         s1 = fp_src(&b, &inst->Src[1].Register, i);
         *slot = fp_alu(FP_ALU_OP_MAD, d, s0, s1, fp_alu_src_zero());
         break;
      case TGSI_OPCODE_ADD:
         s1 = fp_src(&b, &inst->Src[1].Register, i);
         *slot = fp_alu(FP_ALU_OP_MAD, d, s0, fp_alu_src_one(), s1);
         break;
      case TGSI_OPCODE_MAD:
         s1 = fp_src(&b, &inst->Src[1].Register, i);
         s2 = fp_src(&b, &inst->Src[2].Register, i);
         *slot = fp_alu(FP_ALU_OP_MAD, d, s0, s1, s2);
         break;
      case TGSI_OPCODE_MIN:
      case TGSI_OPCODE_MAX:
         s1 = fp_src(&b, &inst->Src[1].Register, i);
         *slot = fp_alu(inst->Instruction.Opcode == TGSI_OPCODE_MIN ?
                        FP_ALU_OP_MIN : FP_ALU_OP_MAX,
                        d, s0, fp_alu_src_one(), s1);
         break;
      case TGSI_OPCODE_CMP:
         /* dst = src0 < 0 ? src1 : src2 */
         s1 = fp_src(&b, &inst->Src[1].Register, i);
         s2 = fp_src(&b, &inst->Src[2].Register, i);
         *slot = fp_alu(FP_ALU_OP_CSEL, d, s0, s1, s2);
         break;
      default:
         UNREACHABLE("not a component-wise opcode");
      }
   }
   fp_batch_end(&b);
}

/*
 * DP2/DP3/DP4 in one ALU packet, slot k computing a[k] * b[k]. Measured on
 * Tegra20: "send" on slot 1 and 2 adds their products to slot 0, "recv" on
 * slot 0 adds slot 3's product (slot 3's own "send" would go to slot 2, and
 * "recv" on slot k only adds slot k-1's product, not a running sum). Slot 0
 * then holds the whole sum and writes the first written channel; a second
 * batch copies it to the other written channels.
 */
static void
emit_dot(struct grate_fp_shader *fp, const struct tgsi_full_instruction *inst,
         int n)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   bool sat = inst->Instruction.Saturate != 0;
   int first = ffs(dst->WriteMask) - 1;
   struct fp_batch b;

   assert(first >= 0);
   fp_batch_begin(&b, fp);
   for (int k = 0; k < n; k++) {
      struct fp_alu_src_operand a = fp_src(&b, &inst->Src[0].Register, k);
      struct fp_alu_src_operand c = fp_src(&b, &inst->Src[1].Register, k);
      struct fp_alu_dst_operand d = k == 0 ?
         fp_dst(&b, dst, first, sat) : (struct fp_alu_dst_operand) { .index = 31 };

      b.alu->slots[k] = fp_alu(FP_ALU_OP_MAD, d, a, c, fp_alu_src_zero());
      b.alu->slots[k].accumulate_other = k == 1 || k == 2;
      b.alu->slots[k].accumulate_this = k == 0 && n == 4;
   }
   fp_batch_end(&b);

   if (dst->WriteMask == (1u << first))
      return;

   fp_batch_begin(&b, fp);
   for (int i = first + 1; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      b.alu->slots[i] = fp_alu(FP_ALU_OP_MAD, fp_dst(&b, dst, i, false),
                               fp_src_of_dst(dst, first), fp_alu_src_one(),
                               fp_alu_src_zero());
   }
   fp_batch_end(&b);
}

static void
emit_tgsi_instr(struct grate_fp_shader *fp, const struct tgsi_full_instruction *inst)
{
   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_MOV:
   case TGSI_OPCODE_MUL:
   case TGSI_OPCODE_ADD:
   case TGSI_OPCODE_MAD:
   case TGSI_OPCODE_MIN:
   case TGSI_OPCODE_MAX:
   case TGSI_OPCODE_CMP:
      emit_componentwise(fp, inst);
      break;

   case TGSI_OPCODE_DP2:
      emit_dot(fp, inst, 2);
      break;
   case TGSI_OPCODE_DP3:
      emit_dot(fp, inst, 3);
      break;
   case TGSI_OPCODE_DP4:
      emit_dot(fp, inst, 4);
      break;

   default:
      fprintf(stderr, "GRATE FRAG TGSI UNIMPLEMENTED: 0x%02x\n", inst->Instruction.Opcode);
      assert(0);
   }
}

#define LINK_SRC(index) ((index) << 3)
#define LINK_DST(index, comp, type) (((comp) | (type) << 2) << ((index) * 4))
#define LINK_DST_NONE      0
#define LINK_DST_FX10_LOW  1
#define LINK_DST_FX10_HIGH 2
#define LINK_DST_FP20      3

static void
emit_tgsi_input(struct grate_fp_shader *fp, const struct tgsi_full_declaration *decl)
{
   grate_trace();
   assert(decl->Range.First == decl->Range.Last);

   uint32_t src = LINK_SRC(1);
   uint32_t dst = 0;
   for (int i = 0; i < 4; ++i)
      dst |= LINK_DST(i, i, LINK_DST_FP20);

   fp->info.inputs[fp->info.num_inputs].src = src;
   fp->info.inputs[fp->info.num_inputs].dst = dst;

   if (decl->Declaration.Semantic == TGSI_SEMANTIC_COLOR)
      fp->info.color_input = decl->Range.First;

   fp->info.num_inputs++;
}

static void
emit_tgsi_declaration(struct grate_fp_shader *fp, const struct tgsi_full_declaration *decl)
{
   switch (decl->Declaration.File) {
   case TGSI_FILE_INPUT:
      emit_tgsi_input(fp, decl);
      break;
   }
}

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi)
{
   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);

   fp->info.num_inputs = 0;
   fp->info.color_input = -1;
   fp->info.max_tram_row = 1;
   fp->info.used_mask = 0;
   fp->info.imm_mask = 0;
   fp->num_immediates = 0;

   while (!tgsi_parse_end_of_tokens(tgsi)) {
      tgsi_parse_token(tgsi);
      switch (tgsi->FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_DECLARATION:
         emit_tgsi_declaration(fp, &tgsi->FullToken.FullDeclaration);
         break;

      case TGSI_TOKEN_TYPE_IMMEDIATE: {
         const struct tgsi_full_immediate *imm = &tgsi->FullToken.FullImmediate;
         unsigned n = MIN2(imm->Immediate.NrTokens - 1, 4);
         assert(fp->num_immediates < GRATE_FP_MAX_IMMEDIATES);
         for (unsigned i = 0; i < 4; i++)
            fp->immediates[fp->num_immediates][i] = i < n ? imm->u[i].Float : 0.0f;
         fp->num_immediates++;
         break;
      }

      case TGSI_TOKEN_TYPE_INSTRUCTION:
         if (tgsi->FullToken.FullInstruction.Instruction.Opcode != TGSI_OPCODE_END)
            emit_tgsi_instr(fp, &tgsi->FullToken.FullInstruction);
         break;
      }
   }

   /*
    * Store the pixel once, from the last batch: partial colour writes
    * accumulate in r2/r3 across batches, and PSEQ_DWR_IF_STATE makes the
    * DW stage wait for all batches.
    */
   struct fp_dw_instr store = { 0 };
   list_for_each_entry(struct fp_instr, inst, &fp->fp_instructions, link) {
      if (inst->dw.enable)
         store = inst->dw;
      memset(&inst->dw, 0, sizeof(inst->dw));
   }
   if (!list_is_empty(&fp->fp_instructions))
      list_last_entry(&fp->fp_instructions, struct fp_instr, link)->dw = store;

   /*
    * HACK: insert barycentric interpolation setup
    * This will overwrite instructions in some cases, need proper scheduler
    * to fix properly
    */
    if (list_is_empty(&fp->mfu_instructions))
       return;

    struct fp_mfu_instr *first = list_first_entry(&fp->mfu_instructions, struct fp_mfu_instr, link);
    first->sfu.op = FP_SFU_OP_RCP;
    first->sfu.reg = 4;
    first->mul[0].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
    first->mul[0].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
    first->mul[0].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_0;

    first->mul[1].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
    first->mul[1].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
    first->mul[1].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_1;
}
