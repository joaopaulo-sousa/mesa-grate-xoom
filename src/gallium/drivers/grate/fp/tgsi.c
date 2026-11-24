#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "grate_common.h"
#include "fp_compiler.h"

#include "tgsi/tgsi_parse.h"

#include "util/bitscan.h"
#include "util/blend.h"
#include "util/u_debug.h"
#include "util/u_memory.h"

struct fp_alu_src_operand
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

struct fp_alu_src_operand
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
fp_const_slot(struct fp_compiler *c, float value)
{
   struct grate_fp_shader *fp = c->fp;
   uint32_t fp20 = grate_float_to_fp20(value);
   int i;

   for (i = 0; i < GRATE_FP_MAX_SLOTS; i++)
      if (BITSET_TEST(fp->info.imm_mask, i) && fp->info.consts[i] == fp20)
         return i;

   for (i = 0; i < GRATE_FP_MAX_SLOTS; i++)
      if (!BITSET_TEST(fp->info.used_mask, i))
         break;
   if (i == GRATE_FP_MAX_SLOTS) {
      c->fp->info.out_of_uniforms = true;
      fp_fail(c, "too many uniforms");
      return 0;
   }

   fp->info.consts[i] = fp20;
   BITSET_SET(fp->info.imm_mask, i);
   BITSET_SET(fp->info.used_mask, i);
   return i;
}

/* ALU uniform register of a user uniform component: the components read
 * get one each, in order of first use */
static int
fp_uniform_slot(struct fp_compiler *c, int index, int comp)
{
   struct grate_fp_info *info = &c->fp->info;
   int user = index * 4 + comp;

   for (int i = 0; i < GRATE_FP_MAX_SLOTS; i++)
      if (BITSET_TEST(info->user_mask, i) && info->user_index[i] == user)
         return i;
   for (int i = 0; i < GRATE_FP_MAX_SLOTS; i++)
      if (!BITSET_TEST(info->used_mask, i)) {
         info->user_index[i] = user;
         BITSET_SET(info->user_mask, i);
         BITSET_SET(info->used_mask, i);
         return i;
      }
   c->fp->info.out_of_uniforms = true;
   fp_fail(c, "too many uniforms");
   return 0;
}

struct fp_alu_src_operand
fp_alu_src_zero(void)
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = 0
   };
   return src;
}

/* A register the hardware fills (FP_REG_*) */
struct fp_alu_src_operand
fp_alu_src_special(int index)
{
   struct fp_alu_src_operand src = {
      .index = index,
      .datatype = index == FP_REG_FACE ? FP_DATATYPE_FIXED10 : FP_DATATYPE_FP20,
   };
   return src;
}

struct fp_alu_src_operand
fp_alu_src_one(void)
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = 1
   };
   return src;
}

/*
 * Constants of the program (immediates, not per-draw uniforms or preshader
 * results) go into the packets that read them, as embedded constants in
 * place of an unused slot 3 (measured on Tegra20, gpu/tools/asmtri): up to
 * three fp20 values a packet, which its slots read as registers 28..30.
 * When no packet reads a uniform register any more, the draw does not
 * upload them (grate_fp_info.reads_uniforms).
 */
static void
fp_embed_constants(struct grate_fp_shader *fp)
{
   struct grate_fp_info *info = &fp->info;
   const struct fp_alu_instr nop = fp_alu_nop();
   bool embed = !debug_get_bool_option("GRATE_FP_NO_EMBED", false);
   BITSET_DECLARE(pre, GRATE_FP_MAX_SLOTS);

   BITSET_ZERO(pre);
   for (int i = 0; i < info->num_pre; i++)
      if (info->pre[i].dst != GRATE_FP_PRE_NO_DST)
         BITSET_SET(pre, info->pre[i].dst);

   info->reads_uniforms = false;
   list_for_each_entry(struct fp_alu_instr_packet, pk, &fp->alu_instructions, link) {
      bool free3 = embed && !memcmp(&pk->slots[3], &nop, sizeof(nop));

      for (int k = 0; k < 4; k++) {
         struct fp_alu_instr *a = &pk->slots[k];
         unsigned old_c = a->src[2].index;

         for (int j = 0; j < 3; j++) {
            struct fp_alu_src_operand *s = &a->src[j];
            if (s->index < 32 || s->index >= 64)
               continue;
            unsigned slot = s->index - 32;
            if (!free3 || k == 3 || s->datatype != FP_DATATYPE_FP20 ||
                !BITSET_TEST(info->imm_mask, slot) || BITSET_TEST(pre, slot)) {
               info->reads_uniforms = true;
               continue;
            }
            unsigned e = 0;
            while (e < pk->num_imm && pk->imm[e] != info->consts[slot])
               e++;
            if (e == pk->num_imm) {
               if (pk->num_imm == 3) {
                  info->reads_uniforms = true;
                  continue;
               }
               pk->imm[pk->num_imm++] = info->consts[slot];
            }
            s->index = FP_EMBEDDED_REG(e);
         }
         /* rD repeats rC's register (or rB's): keep it on rC's */
         if (a->src[3].index == old_c)
            a->src[3].index = a->src[2].index;
         else if (a->rd_enable && a->src[3].index >= 32 && a->src[3].index < 64)
            info->reads_uniforms = true;
      }
   }
}

/* An ALU slot that does nothing: writes vec2(0, 1) to the read-only r31 */
struct fp_alu_instr
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

void
fp_fail(struct fp_compiler *c, const char *msg)
{
   if (!c->failed)
      fprintf(stderr, "GRATE FRAG: %s\n", msg);
   c->failed = true;
}

struct fp_unit *
fp_new_unit(struct fp_compiler *c)
{
   if (c->num_units == FP_MAX_UNITS) {
      fp_fail(c, "shader too long");
      c->num_units--;
   }
   struct fp_unit *u = &c->units[c->num_units++];
   memset(u, 0, sizeof(*u));
   u->num_slots = 1;
   u->output_chan = -1;
   u->forward = true;
   u->packet = -1;
   u->reg = -1;
   u->half = -1;
   return u;
}

static struct fp_opnd
fp_opnd_uniform(int slot)
{
   struct fp_opnd o = { .kind = FP_OPND_UNIFORM, .index = slot };
   return o;
}

struct fp_opnd
fp_opnd_value(struct fp_compiler *c, struct fp_unit *u)
{
   struct fp_opnd o = { .kind = FP_OPND_VALUE, .index = u - c->units };
   return o;
}

/* TGSI source modifiers: absolute value first, then negation */
static struct fp_opnd
fp_opnd_modify(struct fp_opnd o, bool negate, bool absolute)
{
   if (absolute) {
      o.absolute = true;
      o.negate = negate;
   } else
      o.negate ^= negate;
   return o;
}

/* 0 and 1 operands: fixed10 r31 for a colour write, fp20 uniforms otherwise
 * (the fixed10 constants give wrong fp20 results) */
struct fp_opnd
fp_opnd_zero(struct fp_compiler *c, bool output)
{
   if (output) {
      struct fp_opnd o = { .kind = FP_OPND_ZERO };
      return o;
   }
   return fp_opnd_uniform(fp_const_slot(c, 0.0f));
}

struct fp_opnd
fp_opnd_one(struct fp_compiler *c, bool output)
{
   if (output) {
      struct fp_opnd o = { .kind = FP_OPND_ONE };
      return o;
   }
   return fp_opnd_uniform(fp_const_slot(c, 1.0f));
}

/* An fp20 immediate */
struct fp_opnd
fp_opnd_const(struct fp_compiler *c, float value)
{
   return fp_opnd_uniform(fp_const_slot(c, value));
}

static struct fp_opnd
fp_val(struct fp_compiler *c, enum fp_alu_op op, struct fp_opnd a,
       struct fp_opnd b, struct fp_opnd cc);
static struct fp_opnd
fp_sfu(struct fp_compiler *c, enum fp_sfu_op op, struct fp_opnd x);

/*
 * The fragment position and the polygon face are registers, not varyings
 * (grate fragment-shader-isa.md). Measured: the position registers hold
 * 8192 plus the pixel's column and row from the top, integers; with the
 * pixel centre added that is the upper left, half integer convention the
 * screen asks for (the frontend flips y where needed). The face is 1.0 in
 * the low fixed10 half for front faces (SU_PARAM FRONT_FACE), 0.0 for back
 * ones; TGSI wants it positive and negative.
 */
static bool
fp_special_input(struct fp_compiler *c, int input, int comp, struct fp_opnd *o)
{
   unsigned name = c->input_semantic[input] >> 8;
   struct fp_unit *u;

   if (name == TGSI_SEMANTIC_FACE) {
      struct fp_opnd face = { .kind = FP_OPND_SPECIAL, .index = FP_REG_FACE };
      u = fp_new_unit(c);
      fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, face, fp_opnd_const(c, 2.0f),
                 fp_opnd_const(c, -1.0f));
      *o = fp_opnd_value(c, u);
      return true;
   }
   if (name != TGSI_SEMANTIC_POSITION)
      return false;
   if (comp > 1) {
      /*
       * z and w: the clip space z and w of the position export, a varying
       * like any other (the linker reads export 0 by its semantic), which
       * the perspective correct interpolation gives exactly. w is 1 / w_clip
       * and z the viewport's depth transform of z_clip / w_clip, its scale
       * and translate in the key. The NVIDIA blob also spends TRAM columns
       * on them (cgdump fragz: w from an interpolated component and rcp, z
       * from a "magic" linker write into r3).
       */
      struct fp_opnd w = { .kind = FP_OPND_INPUT, .index = input, .comp = 3 };
      struct fp_opnd rw = fp_sfu(c, FP_SFU_OP_RCP, w);
      if (comp == 3) {
         *o = rw;
         return true;
      }
      struct fp_opnd z = { .kind = FP_OPND_INPUT, .index = input, .comp = 2 };
      struct fp_opnd ndc = fp_val(c, FP_ALU_OP_MAD, z, rw, fp_opnd_zero(c, false));
      *o = fp_val(c, FP_ALU_OP_MAD, ndc, fp_opnd_const(c, c->key->depth[0]),
                  fp_opnd_const(c, c->key->depth[1]));
      return true;
   }

   struct fp_opnd reg = { .kind = FP_OPND_SPECIAL,
                          .index = comp ? FP_REG_POSITION_Y : FP_REG_POSITION_X };
   /* in two steps: 8191.5 is no fp20 */
   u = fp_new_unit(c);
   fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, reg, fp_opnd_one(c, false),
              fp_opnd_const(c, -8192.0f));
   struct fp_unit *centre = fp_new_unit(c);
   fp_unit_op(&centre->slots[0], FP_ALU_OP_MAD, fp_opnd_value(c, u), fp_opnd_one(c, false),
              fp_opnd_const(c, 0.5f));
   *o = fp_opnd_value(c, centre);
   return true;
}

static struct fp_opnd
fp_src(struct fp_compiler *c, const struct tgsi_src_register *src, int chan)
{
   struct grate_fp_shader *fp = c->fp;
   const int swizzle[] = {
      src->SwizzleX, src->SwizzleY, src->SwizzleZ, src->SwizzleW
   };
   int comp = swizzle[chan];
   struct fp_opnd o = { 0 };

   switch (src->File) {
   case TGSI_FILE_INPUT:
      if (src->Index >= FP_MAX_INPUTS) {
         fp_fail(c, "too many inputs");
         break;
      }
      if (fp_special_input(c, src->Index, comp, &o))
         break;
      o.kind = FP_OPND_INPUT;
      o.index = src->Index;
      o.comp = comp;
      break;

   case TGSI_FILE_IMMEDIATE: {
      assert(src->Index < fp->num_immediates);
      float value = fp->immediates[src->Index][comp];
      if (src->Absolute)
         value = fabsf(value);
      if (src->Negate)
         value = -value;
      return fp_opnd_uniform(fp_const_slot(c, value));
   }

   case TGSI_FILE_CONSTANT:
      /* a uniform a loop exits by, in a variant for its value */
      for (int i = 0; i < c->key->num_flow; i++) {
         if (c->key->flow_user[i] != src->Index * 4 + comp)
            continue;
         float value = c->key->flow[i];
         if (src->Absolute)
            value = fabsf(value);
         if (src->Negate)
            value = -value;
         return fp_opnd_uniform(fp_const_slot(c, value));
      }
      o = fp_opnd_uniform(fp_uniform_slot(c, src->Index, comp));
      break;

   case TGSI_FILE_TEMPORARY:
      if (src->Index >= FP_MAX_TEMPS) {
         fp_fail(c, "too many temporaries");
         break;
      }
      /* a temporary read before being written is 0 */
      if (!c->temp_written[src->Index][comp])
         return fp_opnd_zero(c, false);
      o = c->temps[src->Index][comp];
      break;

   default:
      fprintf(stderr, "GRATE FRAG: unsupported source file %d\n", src->File);
      c->failed = true;
   }

   return fp_opnd_modify(o, src->Negate, src->Absolute);
}

/* A value written under a predicate: the old one for the other pixels (a
 * temporary not written yet is 0) */
static struct fp_opnd
fp_merge(struct fp_compiler *c, struct fp_opnd v, bool written, struct fp_opnd old)
{
   if (c->pred.kind != FP_PRED_VALUE)
      return v;
   return fp_val(c, FP_ALU_OP_CSEL, c->pred.v, v, written ? old : fp_opnd_zero(c, false));
}

/* Unit writing "chan" of a TGSI destination; returns whether it is a colour
 * write. A temporary channel becomes the unit's value. */
static bool
fp_dst(struct fp_compiler *c, struct fp_unit *u,
       const struct tgsi_dst_register *dst, int chan)
{
   switch (dst->File) {
   case TGSI_FILE_OUTPUT:
      c->dw.enable = 1;
      c->dw.index = GRATE_RT_COLOR;
      c->dw.stencil_write = 0;
      c->dw.src_regs = FP_DW_REGS_R2_R3;
      if (c->redirect_output) {
         c->outv[chan] = fp_merge(c, fp_opnd_value(c, u), c->outv_written[chan],
                                  c->outv[chan]);
         c->outv_written[chan] = true;
         return false;
      }
      u->output_chan = chan;
      return true;

   case TGSI_FILE_TEMPORARY:
      if (dst->Index >= FP_MAX_TEMPS) {
         fp_fail(c, "too many temporaries");
         return false;
      }
      c->temps[dst->Index][chan] = fp_merge(c, fp_opnd_value(c, u),
                                            c->temp_written[dst->Index][chan],
                                            c->temps[dst->Index][chan]);
      c->temp_written[dst->Index][chan] = true;
      return false;

   default:
      fprintf(stderr, "GRATE FRAG: unsupported destination file %d\n", dst->File);
      c->failed = true;
      return false;
   }
}

/* Whether a destination is written to r2/r3 as fixed10 colour (not when the
 * colour is blended: then it is a value like a temporary) */
static bool
fp_color_write(struct fp_compiler *c, const struct tgsi_dst_register *dst)
{
   return dst->File == TGSI_FILE_OUTPUT && !c->redirect_output;
}

/* A channel of a temporary or redirected output becomes an operand without
 * code (plain copies); false if the destination can't take one */
static bool
fp_set_dst_opnd(struct fp_compiler *c, const struct tgsi_dst_register *dst,
                int chan, struct fp_opnd o)
{
   if (dst->File == TGSI_FILE_TEMPORARY && dst->Index < FP_MAX_TEMPS) {
      c->temps[dst->Index][chan] = fp_merge(c, o, c->temp_written[dst->Index][chan],
                                            c->temps[dst->Index][chan]);
      c->temp_written[dst->Index][chan] = true;
      return true;
   }
   if (dst->File == TGSI_FILE_OUTPUT && c->redirect_output) {
      c->dw.enable = 1;
      c->dw.index = GRATE_RT_COLOR;
      c->dw.src_regs = FP_DW_REGS_R2_R3;
      c->outv[chan] = fp_merge(c, o, c->outv_written[chan], c->outv[chan]);
      c->outv_written[chan] = true;
      return true;
   }
   return false;
}

void
fp_unit_op(struct fp_unit_slot *s, enum fp_alu_op op, struct fp_opnd a,
           struct fp_opnd b, struct fp_opnd cc)
{
   /* result = op(a * b, c) */
   s->op = op;
   s->src[0] = a;
   s->src[1] = b;
   s->src[2] = cc;
}

/* Component-wise instruction: one unit per written channel */

static void
emit_componentwise(struct fp_compiler *c,
                   const struct tgsi_full_instruction *inst)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   unsigned opcode = inst->Instruction.Opcode;
   bool sat = inst->Instruction.Saturate != 0;
   bool output = fp_color_write(c, dst);
   struct fp_opnd s[4][3];

   /* read every source before writing, the instruction may overwrite them */
   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      for (unsigned k = 0; k < inst->Instruction.NumSrcRegs; k++)
         s[i][k] = fp_src(c, &inst->Src[k].Register, i);
   }

   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;

      /* a plain copy into a temporary just renames the operand */
      if (opcode == TGSI_OPCODE_MOV && !sat && !output &&
          s[i][0].kind != FP_OPND_ZERO && s[i][0].kind != FP_OPND_ONE &&
          fp_set_dst_opnd(c, dst, i, s[i][0]))
         continue;

      /* LRP: dst = src0 * src1 + (1 - src0) * src2
       *         = src0 * (src1 - src2) + src2; the difference first, the
       * units stay in topological order */
      struct fp_opnd lrp = { 0 };
      if (opcode == TGSI_OPCODE_LRP) {
         float v1, v2;
         if (fp_opnd_known(c, &s[i][1], &v1) && fp_opnd_known(c, &s[i][2], &v2)) {
            /* constants: the difference as an immediate (x * 2 - 1 comes
             * as LRP(x, 1, -1), which fp_fold_minus_one then frees) */
            lrp = fp_opnd_const(c, v1 - v2);
         } else {
            struct fp_opnd neg2 = s[i][2];
            neg2.negate = !neg2.negate;
            lrp = fp_val(c, FP_ALU_OP_MAD, s[i][1], fp_opnd_one(c, false), neg2);
         }
      }

      struct fp_unit *u = fp_new_unit(c);
      struct fp_opnd one = fp_opnd_one(c, output);
      struct fp_opnd zero = fp_opnd_zero(c, output);

      u->saturate = sat;
      switch (opcode) {
      case TGSI_OPCODE_MOV:
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, s[i][0], one, zero);
         break;
      case TGSI_OPCODE_MUL:
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, s[i][0], s[i][1], zero);
         break;
      case TGSI_OPCODE_ADD:
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, s[i][0], one, s[i][1]);
         break;
      case TGSI_OPCODE_MAD:
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, s[i][0], s[i][1], s[i][2]);
         break;
      case TGSI_OPCODE_MIN:
         fp_unit_op(&u->slots[0], FP_ALU_OP_MIN, s[i][0], one, s[i][1]);
         break;
      case TGSI_OPCODE_MAX:
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAX, s[i][0], one, s[i][1]);
         break;
      case TGSI_OPCODE_CMP:
         /* dst = src0 < 0 ? src1 : src2 */
         fp_unit_op(&u->slots[0], FP_ALU_OP_CSEL, s[i][0], s[i][1], s[i][2]);
         break;
      case TGSI_OPCODE_LRP:
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, s[i][0], lrp, s[i][2]);
         break;
      default:
         UNREACHABLE("not a component-wise opcode");
      }
      fp_dst(c, u, dst, i);
   }
}

/*
 * SLT/SGE/SEQ/SNE: d = a - b, then CMP (d < 0 ? x : y) on d, or on -|d| for
 * the equality tests, selecting 1.0 or 0.0.
 */
static void
emit_setcc(struct fp_compiler *c, const struct tgsi_full_instruction *inst)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   unsigned opcode = inst->Instruction.Opcode;
   bool output = fp_color_write(c, dst);
   struct fp_opnd s[4][2];

   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      s[i][0] = fp_src(c, &inst->Src[0].Register, i);
      s[i][1] = fp_src(c, &inst->Src[1].Register, i);
   }

   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;

      struct fp_unit *d = fp_new_unit(c);
      fp_unit_op(&d->slots[0], FP_ALU_OP_MAD, s[i][0], fp_opnd_one(c, false),
                 fp_opnd_modify(s[i][1], true, false));

      bool eq = opcode == TGSI_OPCODE_SEQ || opcode == TGSI_OPCODE_SNE;
      bool lt_one = opcode == TGSI_OPCODE_SLT || opcode == TGSI_OPCODE_SNE;
      struct fp_opnd cond = fp_opnd_modify(fp_opnd_value(c, d), eq, eq);
      struct fp_opnd one = fp_opnd_one(c, output), zero = fp_opnd_zero(c, output);

      struct fp_unit *u = fp_new_unit(c);
      u->saturate = inst->Instruction.Saturate != 0;
      fp_unit_op(&u->slots[0], FP_ALU_OP_CSEL, cond, lt_one ? one : zero,
                 lt_one ? zero : one);
      fp_dst(c, u, dst, i);
   }
}

/*
 * SSG: x < 0 ? -1 : (-x < 0 ? 1 : 0). The NVIDIA compiler (cgdump
 * shaders/sign.glsl) takes the difference of two comparisons instead.
 */
static void
emit_sign(struct fp_compiler *c, const struct tgsi_full_instruction *inst)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   bool output = fp_color_write(c, dst);
   struct fp_opnd x[4];

   for (int i = 0; i < 4; i++)
      if (dst->WriteMask & (1 << i))
         x[i] = fp_src(c, &inst->Src[0].Register, i);

   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      struct fp_opnd pos = fp_val(c, FP_ALU_OP_CSEL, fp_opnd_modify(x[i], true, false),
                                  fp_opnd_one(c, false), fp_opnd_zero(c, false));
      struct fp_unit *u = fp_new_unit(c);
      u->saturate = inst->Instruction.Saturate != 0;
      fp_unit_op(&u->slots[0], FP_ALU_OP_CSEL, x[i],
                 fp_opnd_modify(fp_opnd_one(c, output), true, false), pos);
      fp_dst(c, u, dst, i);
   }
}

/*
 * DP2/DP3/DP4 in slots 0..n-1 of one packet, slot k computing a[k] * b[k].
 * Measured on Tegra20: "send" on slot 1 and 2 adds their products to slot 0,
 * "recv" on slot 0 adds slot 3's product. Slot 0 then holds the sum; whether
 * the next packet sees the sum as alu0 is not known, so it is only read back
 * from its register.
 */
static bool fp_opnd_equal(const struct fp_opnd *a, const struct fp_opnd *b);

static void
emit_dot(struct fp_compiler *c, const struct tgsi_full_instruction *inst,
         int n)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   struct fp_unit *u = fp_new_unit(c);

   struct fp_opnd a[4], b[4];
   bool squares = true;
   for (int k = 0; k < n; k++) {
      a[k] = fp_src(c, &inst->Src[0].Register, k);
      b[k] = fp_src(c, &inst->Src[1].Register, k);
      squares &= fp_opnd_equal(&a[k], &b[k]);
   }

   u->forward = false;
   if (squares) {
      /*
       * dot(v, v): slot 0 computes rA * rB + rC * rD with rD = rC, two
       * squares, the others send it one square each. Measured: a sending
       * slot gives its rA * rB only, without the rC * rD term.
       */
      u->num_slots = n - 1;
      fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, a[0], a[0], a[1]);
      u->slots[0].square_c = true;
      for (int k = 1; k < u->num_slots; k++) {
         fp_unit_op(&u->slots[k], FP_ALU_OP_MAD, a[k + 1], a[k + 1], fp_opnd_zero(c, false));
         u->slots[k].accumulate_other = true;
      }
   } else {
      u->num_slots = n;
      for (int k = 0; k < n; k++) {
         struct fp_unit_slot *s = &u->slots[k];
         fp_unit_op(s, FP_ALU_OP_MAD, a[k], b[k], fp_opnd_zero(c, false));
         s->accumulate_other = k == 1 || k == 2;
         s->accumulate_this = k == 0 && n == 4;
      }
   }

   if (!fp_color_write(c, dst)) {
      u->saturate = inst->Instruction.Saturate != 0;
      for (int i = 0; i < 4; i++)
         if (dst->WriteMask & (1 << i))
            fp_dst(c, u, dst, i);
      return;
   }

   /* colour: copy the sum to every written channel */
   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      struct fp_unit *m = fp_new_unit(c);
      m->saturate = inst->Instruction.Saturate != 0;
      fp_unit_op(&m->slots[0], FP_ALU_OP_MAD, fp_opnd_value(c, u),
                 fp_opnd_one(c, true), fp_opnd_zero(c, true));
      fp_dst(c, m, dst, i);
   }
}

/* value unit op(a * b, c) */
static struct fp_opnd
fp_val(struct fp_compiler *c, enum fp_alu_op op, struct fp_opnd a,
       struct fp_opnd b, struct fp_opnd cc)
{
   struct fp_unit *u = fp_new_unit(c);
   fp_unit_op(&u->slots[0], op, a, b, cc);
   return fp_opnd_value(c, u);
}

/*
 * TEX with varying coordinates: one texture fetch per EXEC, the texel
 * channels become fixed10 operands. A colour destination gets copies.
 */
/* Texture coordinates a multi-row program can fetch with: two components
 * of a varying, interpolated straight into a register pair (the TRAM layout
 * puts any component in any column) */
static bool
fp_tex_varying(const struct fp_opnd *s, const struct fp_opnd *t)
{
   return s->kind == FP_OPND_INPUT && t->kind == FP_OPND_INPUT &&
          s->index == t->index && !s->negate && !s->absolute &&
          !t->negate && !t->absolute;
}

/*
 * proj: TXP, s and t divided by q (the w of the coordinates). As the NVIDIA
 * compiler does it (cgdump shaders/proj3.glsl, proj4.glsl): q goes to the
 * first register of the texel pair, and the MFU of the TEX EXEC divides in
 * place, "rcp q" then s, t *= the result, like a cube map's major axis.
 */
static void
emit_tex(struct fp_compiler *c, const struct tgsi_full_instruction *inst,
         const struct fp_opnd *bias, bool proj)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   struct fp_opnd s = fp_src(c, &inst->Src[0].Register, 0);
   struct fp_opnd t = fp_src(c, &inst->Src[0].Register, 1);
   unsigned smp = inst->Src[1].Register.Index;

   bool cube = inst->Texture.Texture == TGSI_TEXTURE_CUBE;
   if (inst->Texture.Texture != TGSI_TEXTURE_2D &&
       inst->Texture.Texture != TGSI_TEXTURE_RECT && !cube) {
      fp_fail(c, "only 2D and cube textures are supported");
      return;
   }
   struct fp_opnd cube_r = { 0 };
   bool cube_computed = false;
   int r_comp = 2;
   if (proj && cube) {
      fp_fail(c, "projective cube map fetch");
      return;
   }
   if (proj) {
      struct fp_opnd q = fp_src(c, &inst->Src[0].Register, 3);
      bool wrap = c->key && smp < 8 &&
                  ((c->key->npot_s | c->key->npot_t) & (1u << smp));
      if (wrap) {
         /* the wrap below is of s/q, t/q: divided first */
         struct fp_opnd rq = fp_sfu(c, FP_SFU_OP_RCP, q);
         s = fp_val(c, FP_ALU_OP_MAD, s, rq, fp_opnd_zero(c, false));
         t = fp_val(c, FP_ALU_OP_MAD, t, rq, fp_opnd_zero(c, false));
         proj = false;
      } else if (fp_tex_varying(&s, &t) && q.kind == FP_OPND_INPUT &&
                 q.index == s.index && !q.negate && !q.absolute) {
         r_comp = q.comp;
      } else {
         /* computed: the ALU writes copies the MFU divides in place */
         s = fp_val(c, FP_ALU_OP_MAD, s, fp_opnd_one(c, false), fp_opnd_zero(c, false));
         t = fp_val(c, FP_ALU_OP_MAD, t, fp_opnd_one(c, false), fp_opnd_zero(c, false));
         cube_r = fp_val(c, FP_ALU_OP_MAD, q, fp_opnd_one(c, false), fp_opnd_zero(c, false));
         cube_computed = true;
      }
   }
   if (cube) {
      struct fp_opnd r = fp_src(c, &inst->Src[0].Register, 2);
      if (!fp_tex_varying(&s, &t) || r.kind != FP_OPND_INPUT || r.index != s.index ||
          r.negate || r.absolute || s.comp != 0 || t.comp != 1 || r.comp != 2 ||
          bias) {
         /*
          * Other coordinates, as the NVIDIA compiler handles them (cgdump
          * shaders/cube-computed.glsl): the ALU writes s, t, r to the
          * coordinate pair and the first register of the other one, and
          * the TEX EXEC's MFU divides them by the major axis in place
          * (sched_seq.c). In place: each gets a copy of its own.
          */
         s = fp_val(c, FP_ALU_OP_MAD, s, fp_opnd_one(c, false), fp_opnd_zero(c, false));
         t = fp_val(c, FP_ALU_OP_MAD, t, fp_opnd_one(c, false), fp_opnd_zero(c, false));
         cube_r = fp_val(c, FP_ALU_OP_MAD, r, fp_opnd_one(c, false), fp_opnd_zero(c, false));
         cube_computed = true;
      }
   }
   if (c->num_tex == FP_MAX_TEXTURES) {
      fp_fail(c, "too many texture fetches");
      return;
   }

   /* repeat of a non-power-of-two texture: wrapped here, the TEX unit
    * does not (grate_npot_repeat) */
   if (!cube && c->key && smp < 8 && (c->key->npot_s & (1u << smp)))
      s = fp_sfu(c, FP_SFU_OP_FRC, s);
   if (!cube && c->key && smp < 8 && (c->key->npot_t & (1u << smp)))
      t = fp_sfu(c, FP_SFU_OP_FRC, t);
   /* read from a power-of-two copy (grate_npot_shadow): scaled to the part
    * the texture takes of it. w/W is exact in fp20; on a varying the vertex
    * shader computes it (move.c), the TEX then reads it as it would s */
   if (!cube && c->key && smp < 8 && (c->key->npot_shadow & (1u << smp))) {
      unsigned w = c->key->shadow_size[smp][0], h = c->key->shadow_size[smp][1];
      s = fp_val(c, FP_ALU_OP_MAD, s,
                 fp_opnd_const(c, (float)w / util_next_power_of_two(w)),
                 fp_opnd_zero(c, false));
      t = fp_val(c, FP_ALU_OP_MAD, t,
                 fp_opnd_const(c, (float)h / util_next_power_of_two(h)),
                 fp_opnd_zero(c, false));
   }

   /* coordinates other than two varying components (a dependent read)
    * need a sequential program: the ALU computes them in an earlier pass */
   bool varying = fp_tex_varying(&s, &t);
   if (bias)
      c->seq = true;
   if (!varying) {
      /* the TEX reads fp20 coordinates from a register pair: anything but
       * a plain varying component or a value of its own gets copied */
      c->seq = true;
      for (int k = 0; k < 2; k++) {
         struct fp_opnd *o = k ? &t : &s;
         bool plain = !o->negate && !o->absolute &&
                      (o->kind == FP_OPND_INPUT || o->kind == FP_OPND_VALUE);
         if (plain && !(k == 1 && t.kind == s.kind && t.index == s.index &&
                        t.comp == s.comp))
            continue;
         *o = fp_val(c, FP_ALU_OP_MAD, *o, fp_opnd_one(c, false),
                     fp_opnd_zero(c, false));
      }
   }

   int n = c->num_tex++;
   c->tex[n].coord[0] = s;
   c->tex[n].role[0] = FP_TEX_S;
   c->tex[n].coord[1] = t;
   c->tex[n].role[1] = FP_TEX_T;
   c->tex[n].num_coords = 2;
   if (cube_computed) {
      c->tex[n].coord[2] = cube_r;
      c->tex[n].role[2] = FP_TEX_R;
      c->tex[n].num_coords = 3;
   }
   if (bias) {
      /* per pixel: read by the TEX from a register of its own */
      int k = c->tex[n].num_coords++;
      c->tex[n].coord[k] = fp_val(c, FP_ALU_OP_MAD, *bias, fp_opnd_one(c, false),
                                  fp_opnd_zero(c, false));
      c->tex[n].role[k] = FP_TEX_BIAS;
   }
   c->tex[n].input = varying ? s.index : -1;
   /* the pair of the components, when they are .xy or .zw */
   c->tex[n].pair = varying && !(s.comp & 1) && t.comp == s.comp + 1 ? s.comp / 2 : 0;
   c->tex[n].sampler = inst->Src[1].Register.Index;
   c->tex[n].cube = cube;
   c->tex[n].proj = proj;
   c->tex[n].r_comp = r_comp;
   c->tex[n].exec = -1;
   c->tex[n].live = false;

   /* the sampler view's swizzle: e.g. alpha 1 for BGRX, luminance */
   unsigned sampler = inst->Src[1].Register.Index;
   unsigned swizzle = GRATE_SWIZZLE_IDENTITY;
   if (sampler < GRATE_FP_MAX_SWIZZLED)
      swizzle ^= c->key->tex_swizzle[sampler];

   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      bool output = fp_color_write(c, dst);
      unsigned sw = (swizzle >> (3 * i)) & 7;
      struct fp_opnd texel = { .kind = FP_OPND_TEXEL, .index = n, .comp = sw };
      if (sw == PIPE_SWIZZLE_0)
         texel = fp_opnd_zero(c, output);
      else if (sw == PIPE_SWIZZLE_1)
         texel = fp_opnd_one(c, output);
      if (!inst->Instruction.Saturate && fp_set_dst_opnd(c, dst, i, texel))
         continue;
      struct fp_unit *m = fp_new_unit(c);
      m->saturate = inst->Instruction.Saturate != 0;
      fp_unit_op(&m->slots[0], FP_ALU_OP_MAD, texel, fp_opnd_one(c, output),
                 fp_opnd_zero(c, output));
      fp_dst(c, m, dst, i);
   }
}


/*
 * Special function on a scalar. The SFU works in place on a register, so
 * the operand gets a value of its own first (fp_fold_sfu_copies drops the
 * copy when nothing else reads the operand). ex2/sin/cos take a pre-op in
 * the EXEC before. SFU units force a sequential program.
 */
static struct fp_opnd
fp_sfu_inplace(struct fp_compiler *c, enum fp_sfu_op op, struct fp_opnd x)
{
   struct fp_unit *u = fp_new_unit(c);

   u->num_slots = 0;
   u->sfu = op;
   u->forward = false;
   u->slots[0].src[0] = x;
   c->seq = true;
   return fp_opnd_value(c, u);
}

static struct fp_opnd
fp_sfu(struct fp_compiler *c, enum fp_sfu_op op, struct fp_opnd x)
{
   struct fp_unit *m = fp_new_unit(c);

   m->sfu_copy = true;
   fp_unit_op(&m->slots[0], FP_ALU_OP_MAD, x, fp_opnd_one(c, false),
              fp_opnd_zero(c, false));
   x = fp_opnd_value(c, m);

   switch (op) {
   case FP_SFU_OP_EX2:
      x = fp_sfu_inplace(c, FP_SFU_OP_PREEX2, x);
      break;
   case FP_SFU_OP_SIN:
      x = fp_sfu_inplace(c, FP_SFU_OP_PRESIN, x);
      break;
   case FP_SFU_OP_COS:
      x = fp_sfu_inplace(c, FP_SFU_OP_PRECOS, x);
      break;
   default:
      break;
   }
   return fp_sfu_inplace(c, op, x);
}

/* Write a value to the channels of a TGSI destination */
static void
fp_write_value(struct fp_compiler *c, const struct tgsi_full_instruction *inst,
               int chan, struct fp_opnd v)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   bool sat = inst->Instruction.Saturate != 0;

   if (!sat && fp_set_dst_opnd(c, dst, chan, v))
      return;
   bool output = fp_color_write(c, dst);
   struct fp_unit *m = fp_new_unit(c);
   m->saturate = sat;
   fp_unit_op(&m->slots[0], FP_ALU_OP_MAD, v, fp_opnd_one(c, output),
              fp_opnd_zero(c, output));
   fp_dst(c, m, dst, chan);
}

static void
emit_sfu(struct fp_compiler *c, const struct tgsi_full_instruction *inst)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   unsigned opcode = inst->Instruction.Opcode;
   struct fp_opnd s[4][2], r = { 0 };
   bool scalar = opcode != TGSI_OPCODE_FRC && opcode != TGSI_OPCODE_FLR &&
                 opcode != TGSI_OPCODE_CEIL && opcode != TGSI_OPCODE_TRUNC;

   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      for (unsigned k = 0; k < inst->Instruction.NumSrcRegs; k++)
         s[i][k] = fp_src(c, &inst->Src[k].Register, scalar ? 0 : i);
   }

   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      /* scalar functions compute once (TGSI .x) and replicate */
      if (!scalar || r.kind != FP_OPND_VALUE) {
         switch (opcode) {
         case TGSI_OPCODE_RCP:  r = fp_sfu(c, FP_SFU_OP_RCP, s[i][0]); break;
         case TGSI_OPCODE_RSQ:  r = fp_sfu(c, FP_SFU_OP_RSQ, s[i][0]); break;
         case TGSI_OPCODE_SQRT: r = fp_sfu(c, FP_SFU_OP_SQRT, s[i][0]); break;
         case TGSI_OPCODE_EX2:  r = fp_sfu(c, FP_SFU_OP_EX2, s[i][0]); break;
         case TGSI_OPCODE_LG2:  r = fp_sfu(c, FP_SFU_OP_LG2, s[i][0]); break;
         case TGSI_OPCODE_SIN:  r = fp_sfu(c, FP_SFU_OP_SIN, s[i][0]); break;
         case TGSI_OPCODE_COS:  r = fp_sfu(c, FP_SFU_OP_COS, s[i][0]); break;
         case TGSI_OPCODE_FRC:  r = fp_sfu(c, FP_SFU_OP_FRC, s[i][0]); break;
         case TGSI_OPCODE_FLR:
            /* floor(x) = x - fract(x) */
            r = fp_val(c, FP_ALU_OP_MAD, s[i][0], fp_opnd_one(c, false),
                       fp_opnd_modify(fp_sfu(c, FP_SFU_OP_FRC, s[i][0]), true, false));
            break;
         case TGSI_OPCODE_CEIL:
            /* ceil(x) = -floor(-x) = x + fract(-x) */
            r = fp_val(c, FP_ALU_OP_MAD, s[i][0], fp_opnd_one(c, false),
                       fp_sfu(c, FP_SFU_OP_FRC, fp_opnd_modify(s[i][0], true, false)));
            break;
         case TGSI_OPCODE_TRUNC: {
            if (s[i][0].kind == FP_OPND_INPUT && !s[i][0].negate && !s[i][0].absolute &&
                !c->trunc_abs) {
               /* (cheaper in a sequential program: the |x| form below,
                * which grate_fs_compile retries with when this fails) */
               c->fp->info.varying_trunc = true;
               /*
                * A varying: fract of x itself, so the SFU can run on the
                * interpolated register in a multi-row program (no operand
                * modifiers there). f = fract(x), l = x - f = floor(x);
                * x < 0 with f > 0 rounds up: l + 1.
                */
               struct fp_opnd x = s[i][0];
               struct fp_opnd f = fp_sfu(c, FP_SFU_OP_FRC, x);
               struct fp_opnd l = fp_val(c, FP_ALU_OP_MAD, f,
                                         fp_opnd_modify(fp_opnd_one(c, false), true, false), x);
               /* 1 when f > 0: saturate(f * 2^30) (f = 0 gives 0 exactly; a
                * select on -f would take -0.0 as negative) */
               struct fp_unit *uu = fp_new_unit(c);
               uu->saturate = true;
               fp_unit_op(&uu->slots[0], FP_ALU_OP_MAD, f, fp_opnd_const(c, 1073741824.0f),
                          fp_opnd_zero(c, false));
               struct fp_opnd up = fp_opnd_value(c, uu);
               struct fp_opnd lu = fp_val(c, FP_ALU_OP_MAD, up, fp_opnd_one(c, false), l);
               r = fp_val(c, FP_ALU_OP_CSEL, x, lu, l);
               break;
            }
            /* t = |x| - fract(|x|), x < 0 ? -t : t */
            struct fp_opnd a = fp_opnd_modify(s[i][0], false, true);
            struct fp_opnd t = fp_val(c, FP_ALU_OP_MAD, a, fp_opnd_one(c, false),
                                      fp_opnd_modify(fp_sfu(c, FP_SFU_OP_FRC, a), true, false));
            r = fp_val(c, FP_ALU_OP_CSEL, s[i][0], fp_opnd_modify(t, true, false), t);
            break;
         }
         case TGSI_OPCODE_POW: {
            /* x^y = 2^(y * log2(x)) */
            struct fp_opnd l = fp_sfu(c, FP_SFU_OP_LG2, s[i][0]);
            r = fp_sfu(c, FP_SFU_OP_EX2,
                       fp_val(c, FP_ALU_OP_MAD, l, s[i][1], fp_opnd_zero(c, false)));
            break;
         }
         default:
            UNREACHABLE("not an SFU opcode");
         }
      }
      fp_write_value(c, inst, i, r);
   }
}

/* Whether o is the constant v (an fp20 immediate or the fixed10 0/1) */
bool
fp_opnd_is_const(struct fp_compiler *c, const struct fp_opnd *o, float v)
{
   if (o->negate || o->absolute)
      return false;
   if (o->kind == FP_OPND_ZERO)
      return v == 0.0f;
   if (o->kind == FP_OPND_ONE)
      return v == 1.0f;
   return o->kind == FP_OPND_UNIFORM &&
          BITSET_TEST(c->fp->info.imm_mask, o->index) &&
          c->fp->info.consts[o->index] == grate_float_to_fp20(v);
}

/*
 * Plain copies (x * 1 + 0, as specialization leaves of a product by a
 * known 1) are read as what they copy: a copied texel channel otherwise
 * takes an fp20 register of its own instead of its fixed10 half. SFU units
 * keep theirs, they work in place on a register.
 */
static void
fp_fold_copies(struct fp_compiler *c)
{
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (u->sfu)
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind != FP_OPND_VALUE || o->twice || o->minus_one)
            continue;
         struct fp_unit *m = &c->units[o->index];
         struct fp_unit_slot *ms = &m->slots[0];
         if (!fp_unit_has_value(m) || m->num_slots != 1 || m->sfu || m->sfu_copy ||
             m->saturate || m->scale || m->condition || m->blend_inv ||
             ms->op != FP_ALU_OP_MAD || ms->accumulate_this ||
             ms->accumulate_other || ms->square_c ||
             !fp_opnd_is_const(c, &ms->src[2], 0.0f))
            continue;
         int x = fp_opnd_is_const(c, &ms->src[1], 1.0f) ? 0 :
                 fp_opnd_is_const(c, &ms->src[0], 1.0f) ? 1 : -1;
         if (x < 0 || ms->src[x].twice || ms->src[x].minus_one)
            continue;
         struct fp_opnd y = ms->src[x];
         if (ms->src[1 - x].negate)
            y.negate = !y.negate;
         /* abs(-y) = abs(y); -(y) composes */
         if (o->absolute) {
            y.absolute = true;
            y.negate = o->negate;
         } else {
            y.negate ^= o->negate;
         }
         *o = y;
      }
   }
}

/* An SFU operand copy is unnecessary when it copies a plain value nothing
 * else reads: the SFU can then work on that value's register directly */
static void
fp_fold_sfu_copies(struct fp_compiler *c)
{
   int readers[FP_MAX_UNITS] = { 0 };

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j)
         if (u->slots[k].src[j].kind == FP_OPND_VALUE)
            readers[u->slots[k].src[j].index]++;
   }

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      struct fp_opnd *x = &u->slots[0].src[0];
      if (!u->live || !u->sfu || x->kind != FP_OPND_VALUE)
         continue;
      struct fp_unit *m = &c->units[x->index];
      struct fp_opnd *y = &m->slots[0].src[0];
      if (!m->sfu_copy || readers[x->index] != 1 || y->kind != FP_OPND_VALUE ||
          !fp_opnd_is_const(c, &m->slots[0].src[1], 1.0f) ||
          !fp_opnd_is_const(c, &m->slots[0].src[2], 0.0f) ||
          y->negate || y->absolute || readers[y->index] != 1 ||
          c->units[y->index].output_chan >= 0)
         continue;
      *x = *y;
      m->live = false;
   }
}

/*
 * Discard: a non-zero write to the kill register kills the fragment (zero
 * leaves it alone). KILL_IF kills when any component is negative, one CMP
 * per distinct component.
 */
static void
emit_kill(struct fp_compiler *c, const struct tgsi_full_instruction *inst)
{
   struct fp_opnd one = fp_opnd_one(c, false), zero = fp_opnd_zero(c, false);

   /* only the pixels the instruction is for */
   if (c->pred_full.kind == FP_PRED_VALUE)
      one = fp_val(c, FP_ALU_OP_CSEL, c->pred_full.v, one, zero);

   if (inst->Instruction.Opcode == TGSI_OPCODE_KILL) {
      struct fp_unit *u = fp_new_unit(c);
      u->kill = true;
      fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, one, one, zero);
      return;
   }

   struct fp_opnd x[4];
   for (int i = 0; i < 4; i++) {
      x[i] = fp_src(c, &inst->Src[0].Register, i);
      bool dup = false;
      for (int k = 0; k < i; k++)
         dup |= !memcmp(&x[k], &x[i], sizeof(x[i]));
      if (dup)
         continue;
      /* x < 0 ? 1 : 0 */
      struct fp_unit *u = fp_new_unit(c);
      u->kill = true;
      fp_unit_op(&u->slots[0], FP_ALU_OP_CSEL, x[i], one, zero);
   }
}

static void
emit_tgsi_instr(struct fp_compiler *c, const struct tgsi_full_instruction *inst)
{
   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_MOV:
   case TGSI_OPCODE_MUL:
   case TGSI_OPCODE_ADD:
   case TGSI_OPCODE_MAD:
   case TGSI_OPCODE_MIN:
   case TGSI_OPCODE_MAX:
   case TGSI_OPCODE_CMP:
   case TGSI_OPCODE_LRP:
      emit_componentwise(c, inst);
      break;

   case TGSI_OPCODE_DP2:
      emit_dot(c, inst, 2);
      break;
   case TGSI_OPCODE_DP3:
      emit_dot(c, inst, 3);
      break;
   case TGSI_OPCODE_DP4:
      emit_dot(c, inst, 4);
      break;

   case TGSI_OPCODE_TEX:
      emit_tex(c, inst, NULL, false);
      break;
   case TGSI_OPCODE_TXP:
      emit_tex(c, inst, NULL, true);
      break;

   case TGSI_OPCODE_TXB: {
      /*
       * The bias goes into the sampler's texture descriptor: a constant
       * one at compile time (as the NVIDIA guide recommends), a uniform
       * one at draw time. Any other is read per pixel by the TEX (txb, as
       * the NVIDIA compiler does: cgdump shaders/tex2d-varbias.glsl).
       */
      struct fp_opnd b = fp_src(c, &inst->Src[0].Register, 3);
      unsigned smp = inst->Src[1].Register.Index;
      float v;
      if (smp < 16 && fp_opnd_known(c, &b, &v)) {
         int16_t bias = (int16_t)lroundf(CLAMP(v, -16.0f, 15.9375f) * 16.0f);
         struct grate_fp_info *info = &c->fp->info;
         if ((info->lod_bias_mask & (1u << smp)) && info->lod_bias[smp] != bias) {
            fp_fail(c, "different LOD biases on one sampler");
            break;
         }
         info->lod_bias[smp] = bias;
         info->lod_bias_mask |= 1u << smp;
      } else if (smp < 16 && b.kind == FP_OPND_UNIFORM && !b.absolute &&
                 BITSET_TEST(c->fp->info.user_mask, b.index)) {
         /* a uniform: known at draw time (grate_state.c emit_textures) */
         struct grate_fp_info *info = &c->fp->info;
         unsigned user = info->user_index[b.index];
         bool neg = b.negate;
         if ((info->lod_bias_user_mask & (1u << smp)) &&
             (info->lod_bias_user[smp] != user ||
              !!(info->lod_bias_user_neg & (1u << smp)) != neg)) {
            fp_fail(c, "different LOD biases on one sampler");
            break;
         }
         info->lod_bias_user[smp] = user;
         info->lod_bias_user_mask |= 1u << smp;
         if (neg)
            info->lod_bias_user_neg |= 1u << smp;
      } else {
         emit_tex(c, inst, &b, false);
         break;
      }
      emit_tex(c, inst, NULL, false);
      break;
   }

   case TGSI_OPCODE_RCP:
   case TGSI_OPCODE_RSQ:
   case TGSI_OPCODE_SQRT:
   case TGSI_OPCODE_EX2:
   case TGSI_OPCODE_LG2:
   case TGSI_OPCODE_SIN:
   case TGSI_OPCODE_COS:
   case TGSI_OPCODE_FRC:
   case TGSI_OPCODE_FLR:
   case TGSI_OPCODE_CEIL:
   case TGSI_OPCODE_TRUNC:
   case TGSI_OPCODE_POW:
      emit_sfu(c, inst);
      break;

   case TGSI_OPCODE_KILL:
   case TGSI_OPCODE_KILL_IF:
      emit_kill(c, inst);
      break;

   case TGSI_OPCODE_SLT:
   case TGSI_OPCODE_SGE:
   case TGSI_OPCODE_SEQ:
   case TGSI_OPCODE_SNE:
      emit_setcc(c, inst);
      break;

   case TGSI_OPCODE_SSG:
      emit_sign(c, inst);
      break;

   default:
      fprintf(stderr, "GRATE FRAG TGSI UNIMPLEMENTED: 0x%02x\n", inst->Instruction.Opcode);
      c->failed = true;
   }
}

/* MFU barycentric interpolation setup: the SFU computes 1/w (r4) */
void
fp_mfu_bary_setup(struct fp_mfu_instr *mfu)
{
   mfu->sfu.op = FP_SFU_OP_RCP;
   mfu->sfu.reg = 4;
   mfu->mul[0].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
   mfu->mul[0].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
   mfu->mul[0].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_0;
   mfu->mul[1].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
   mfu->mul[1].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
   mfu->mul[1].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_1;
}

/* PSEQ load of the colour render target into a register pair of the row:
 * OP LD (bit 23), surface (16..19), ARGS bit 3, bit 1 = r2/r3 */
uint32_t
fp_pseq_word(const struct fp_compiler *c, int pair)
{
   return 1u << 23 | (c->dw.index & 0xf) << 16 | 1 << 3 | pair << 1;
}

/* TEX: sampler, coordinates in r2/r3 (bit 4), texel to the other pair
 * (bit 5: r2/r3), enable (bit 10) */
uint32_t
fp_tex_word(const struct fp_compiler *c, int n, int coord_pair)
{
   /* bit 12: txb, the LOD bias from the second register of the texel pair */
   return (c->tex[n].sampler & 0xf) | coord_pair << 4 |
          (1 - coord_pair) << 5 | 1 << 10 |
          (fp_tex_coord(c, n, FP_TEX_BIAS) >= 0) << 12;
}


/*
 * x * blend factor for channel ch; *zero when the term vanishes. s/d are the
 * source and destination colour operands.
 */
static struct fp_opnd
fp_blend_factor(struct fp_compiler *c, enum pipe_blendfactor factor, int ch,
                const struct fp_opnd *s, const struct fp_opnd *d, bool *inv);

static struct fp_opnd
fp_blend_term(struct fp_compiler *c, struct fp_opnd x, enum pipe_blendfactor factor,
              int ch, const struct fp_opnd *s, const struct fp_opnd *d, bool *zero)
{
   bool inv = util_blendfactor_is_inverted(factor);
   struct fp_opnd y;

   *zero = false;
   switch (util_blendfactor_without_invert(factor)) {
   case PIPE_BLENDFACTOR_ONE:
      *zero = inv;               /* ZERO is the inverted ONE */
      return x;
   case PIPE_BLENDFACTOR_SRC_COLOR: y = s[ch]; break;
   case PIPE_BLENDFACTOR_SRC_ALPHA: y = s[3]; break;
   case PIPE_BLENDFACTOR_DST_COLOR: y = d[ch]; break;
   case PIPE_BLENDFACTOR_DST_ALPHA: y = d[3]; break;
   case PIPE_BLENDFACTOR_CONST_COLOR:
   case PIPE_BLENDFACTOR_CONST_ALPHA:
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
      y = fp_blend_factor(c, factor, ch, s, d, &inv);
      break;
   default:
      /* SRC1: not yet */
      fprintf(stderr, "GRATE FRAG: blend factor %u unsupported, using ONE\n", factor);
      return x;
   }
   /* x * (1 - y) = x - x * y */
   if (inv)
      return fp_val(c, FP_ALU_OP_MAD, x, fp_opnd_modify(y, true, false), x);
   return fp_val(c, FP_ALU_OP_MAD, x, y, fp_opnd_zero(c, false));
}

/* The operand y of a blend factor (y or 1 - y when *inv); ONE and ZERO
 * are y = 1 and y = 0 */
static struct fp_opnd
fp_blend_factor(struct fp_compiler *c, enum pipe_blendfactor factor, int ch,
                const struct fp_opnd *s, const struct fp_opnd *d, bool *inv)
{
   *inv = util_blendfactor_is_inverted(factor);
   switch (util_blendfactor_without_invert(factor)) {
   case PIPE_BLENDFACTOR_ONE:
      *inv = false;
      return util_blendfactor_is_inverted(factor) ? fp_opnd_zero(c, false) :
                                                    fp_opnd_one(c, false);
   case PIPE_BLENDFACTOR_SRC_COLOR: return s[ch];
   case PIPE_BLENDFACTOR_SRC_ALPHA: return s[3];
   case PIPE_BLENDFACTOR_DST_COLOR: return d[ch];
   case PIPE_BLENDFACTOR_DST_ALPHA: return d[3];
   case PIPE_BLENDFACTOR_CONST_COLOR:
   case PIPE_BLENDFACTOR_CONST_ALPHA: {
      /* glBlendColor from the key, 1 - c folded in */
      float v = c->key->blend_color[
         util_blendfactor_without_invert(factor) == PIPE_BLENDFACTOR_CONST_ALPHA ? 3 : ch];
      v = CLAMP(v, 0.0f, 1.0f);
      if (*inv)
         v = 1.0f - v;
      *inv = false;
      return fp_opnd_const(c, v);
   }
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
      /* min(As, 1 - Ad) for colour, 1 for alpha */
      *inv = false;
      if (ch == 3)
         return fp_opnd_one(c, false);
      return fp_val(c, FP_ALU_OP_MIN, s[3], fp_opnd_one(c, false),
                    fp_val(c, FP_ALU_OP_MAD, d[3],
                           fp_opnd_modify(fp_opnd_one(c, false), true, false),
                           fp_opnd_one(c, false)));
   default:
      fprintf(stderr, "GRATE FRAG: blend factor %u unsupported, using ONE\n", factor);
      *inv = false;
      return fp_opnd_one(c, false);
   }
}

/*
 * Blending and the colour write mask in the shader: the destination colour
 * comes from a PSEQ load, each channel is combined per the blend equation
 * and written to r2/r3 as usual.
 */
static void
emit_blend(struct fp_compiler *c)
{
   const struct grate_fp_key *key = c->key;
   struct fp_opnd s[4], d[4];

   for (int ch = 0; ch < 4; ch++) {
      s[ch] = c->outv_written[ch] ? c->outv[ch] : fp_opnd_zero(c, false);
      d[ch] = (struct fp_opnd) { .kind = FP_OPND_DST, .comp = ch };

      /* blending sees the colour clamped to [0, 1] (unorm target); texels
       * are, and an already saturated result too */
      bool in_range = s[ch].kind == FP_OPND_TEXEL ||
                      (s[ch].kind == FP_OPND_VALUE && !s[ch].negate &&
                       c->units[s[ch].index].saturate);
      if (key->blend && c->outv_written[ch] && !in_range) {
         /* one clamp per distinct operand (scalar results are replicated) */
         int same = -1;
         for (int k = 0; k < ch; k++)
            if (c->outv_written[k] && !memcmp(&c->outv[k], &c->outv[ch], sizeof(c->outv[ch])))
               same = k;
         if (same >= 0) {
            s[ch] = s[same];
            continue;
         }
         struct fp_unit *u = fp_new_unit(c);
         u->saturate = true;
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, s[ch], fp_opnd_one(c, false),
                    fp_opnd_zero(c, false));
         s[ch] = fp_opnd_value(c, u);
      }
   }

   for (int ch = 0; ch < 4; ch++) {
      bool alpha = ch == 3;
      unsigned func = alpha ? key->alpha_func : key->rgb_func;
      unsigned sf = alpha ? key->alpha_src : key->rgb_src;
      unsigned df = alpha ? key->alpha_dst : key->rgb_dst;
      struct fp_opnd one = fp_opnd_one(c, true), zero = fp_opnd_zero(c, true);
      struct fp_unit *u;

      if (!(key->colormask & (1 << ch))) {
         c->dst_used = true;
         u = fp_new_unit(c);
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, d[ch], one, zero);
      } else if (!key->blend) {
         u = fp_new_unit(c);
         u->saturate = true;
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, s[ch], one, zero);
      } else if (func == PIPE_BLEND_MIN || func == PIPE_BLEND_MAX) {
         c->dst_used = true;
         u = fp_new_unit(c);
         u->saturate = true;
         fp_unit_op(&u->slots[0], func == PIPE_BLEND_MIN ? FP_ALU_OP_MIN :
                    FP_ALU_OP_MAX, s[ch], one, d[ch]);
      } else {
         /*
          * out = ts +/- d * DF, one MAD reading the source term ts (the
          * previous packet can forward it): d * y directly, d * (1 - y)
          * as d - d * y with the d added to ts first.
          */
         bool zs, inv;
         c->dst_used = true;
         struct fp_opnd ts = fp_blend_term(c, s[ch], sf, ch, s, d, &zs);
         if (zs)
            ts = fp_opnd_zero(c, false);
         bool sub = func == PIPE_BLEND_SUBTRACT;        /* ts - d * DF */
         bool rsub = func == PIPE_BLEND_REVERSE_SUBTRACT;  /* d * DF - ts */
         if (rsub)
            ts = fp_opnd_modify(ts, true, false);
         struct fp_opnd dd = fp_opnd_modify(d[ch], sub, false);
         struct fp_opnd y = fp_blend_factor(c, df, ch, s, d, &inv);
         /*
          * 1 - y is y read less one, negated: one MAD (fp_expand_blend_inv
          * undoes it for sequential programs). Measured: "minus one" takes
          * fixed10 operands only (a texel, the destination); for an fp20 y
          * d - d * y, with d added to ts first.
          */
         bool fixed10 = y.kind == FP_OPND_TEXEL || y.kind == FP_OPND_DST;
         if (inv && fixed10) {
            y.minus_one = true;
            y.negate = !y.negate;
         } else if (inv) {
            ts = fp_val(c, FP_ALU_OP_MAD, dd, fp_opnd_one(c, false), ts);
            dd = fp_opnd_modify(dd, true, false);
         }
         u = fp_new_unit(c);
         u->saturate = true;
         u->blend_inv = inv && fixed10;
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, dd, y, ts);
      }
      u->output_chan = ch;
   }
}


static bool
fp_opnd_equal(const struct fp_opnd *a, const struct fp_opnd *b)
{
   return a->kind == b->kind && a->index == b->index && a->comp == b->comp &&
          a->negate == b->negate && a->absolute == b->absolute && a->twice == b->twice &&
          a->minus_one == b->minus_one;
}

/* What an SFU unit works on: through the copy that gives it a register */
static const struct fp_opnd *
fp_cse_sfu_source(const struct fp_compiler *c, const struct fp_unit *u)
{
   const struct fp_opnd *x = &u->slots[0].src[0];
   if (x->kind == FP_OPND_VALUE && c->units[x->index].sfu_copy)
      return &c->units[x->index].slots[0].src[0];
   return x;
}

/* Whether a texture fetch reads unit i as a coordinate */
static bool
fp_cse_tex_reads(const struct fp_compiler *c, int i)
{
   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < c->tex[n].num_coords; k++)
         if (c->tex[n].coord[k].kind == FP_OPND_VALUE && c->tex[n].coord[k].index == i)
            return true;
   return false;
}

/*
 * Common subexpressions: a plain value unit identical to an earlier one
 * (TGSI repeats the same operation per channel for replicated scalars), or
 * an SFU unit with the same function of the same value, hands its readers
 * to the earlier one. In unit order, so whole chains collapse;
 * fp_dead_code then drops the copies.
 */
static void
fp_cse(struct fp_compiler *c)
{
   int remap[FP_MAX_UNITS];

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];

      remap[i] = i;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE)
            o->index = remap[o->index];
      }
      if (u->sfu && u->output_chan < 0) {
         /* the same function of the same value (scalar TGSI ops replicated,
          * rcp of an int divisor per channel): once. Not for texture
          * coordinates: s and t take a register each, even when equal */
         if (fp_cse_tex_reads(c, i))
            continue;
         const struct fp_opnd *x = fp_cse_sfu_source(c, u);
         for (int p = 0; p < i; p++) {
            struct fp_unit *q = &c->units[p];
            if (remap[p] == p && q->sfu == u->sfu && q->output_chan < 0 &&
                !fp_cse_tex_reads(c, p) &&
                (fp_cse_sfu_source(c, q) == &q->slots[0].src[0]) ==
                   (x == &u->slots[0].src[0]) &&
                fp_opnd_equal(fp_cse_sfu_source(c, q), x)) {
               remap[i] = p;
               break;
            }
         }
         continue;
      }
      /* nor texture coordinates: each takes a register of its own, which
       * the MFU of a cube or projective fetch divides in place */
      if (!fp_unit_has_value(u) || u->num_slots != 1 || u->sfu || u->sfu_copy ||
          fp_cse_tex_reads(c, i))
         continue;
      for (int p = 0; p < i; p++) {
         struct fp_unit *q = &c->units[p];
         if (remap[p] != p || q->output_chan >= 0 || q->num_slots != 1 ||
             fp_cse_tex_reads(c, p) ||
             q->sfu || q->sfu_copy || q->saturate != u->saturate ||
             q->slots[0].op != u->slots[0].op ||
             q->slots[0].accumulate_this || q->slots[0].accumulate_other ||
             u->slots[0].accumulate_this || u->slots[0].accumulate_other ||
             q->slots[0].square_c || u->slots[0].square_c)
            continue;
         bool same = true;
         for (int j = 0; j < 3; j++)
            same &= fp_opnd_equal(&q->slots[0].src[j], &u->slots[0].src[j]);
         if (same) {
            remap[i] = p;
            break;
         }
      }
   }

   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < c->tex[n].num_coords; k++)
         if (c->tex[n].coord[k].kind == FP_OPND_VALUE)
            c->tex[n].coord[k].index = remap[c->tex[n].coord[k].index];
}

/* A plain multiply: MAD x, y, 0 */
static bool
fp_unit_is_mul(struct fp_compiler *c, const struct fp_unit *u)
{
   return fp_unit_has_value(u) && u->num_slots == 1 && !u->sfu &&
          !u->sfu_copy && !u->saturate && u->slots[0].op == FP_ALU_OP_MAD &&
          !u->slots[0].accumulate_this && !u->slots[0].accumulate_other &&
          !u->slots[0].square_c &&
          fp_opnd_is_const(c, &u->slots[0].src[2], 0.0f);
}

/*
 * MUL + ADD -> MAD: an addition MAD p, 1, q (or MAD q, 1, p) where p is
 * a plain multiply x * y nothing else reads becomes MAD x, y, q. A negated
 * p carries its sign over to x.
 */
static void
fp_fuse_mad(struct fp_compiler *c)
{
   int readers[FP_MAX_UNITS] = { 0 };

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      fp_unit_foreach_src(u, k, j)
         if (u->slots[k].src[j].kind == FP_OPND_VALUE)
            readers[u->slots[k].src[j].index]++;
   }
   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < c->tex[n].num_coords; k++)
         if (c->tex[n].coord[k].kind == FP_OPND_VALUE)
            readers[c->tex[n].coord[k].index] += 2;   /* keep */

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      struct fp_unit_slot *s = &u->slots[0];
      /* SFU operand copies stay plain copies (fp_fold_sfu_copies) */
      if (u->num_slots != 1 || u->sfu || u->sfu_copy || s->op != FP_ALU_OP_MAD ||
          s->accumulate_this || s->accumulate_other || s->square_c)
         continue;

      /* addition: one of a/b is 1, the other and c are the addends */
      int p_at = -1;
      struct fp_opnd addend;
      if (fp_opnd_is_const(c, &s->src[1], 1.0f)) {
         if (s->src[0].kind == FP_OPND_VALUE) {
            p_at = 0;
            addend = s->src[2];
         }
         if ((p_at < 0 || !fp_unit_is_mul(c, &c->units[s->src[0].index])) &&
             s->src[2].kind == FP_OPND_VALUE &&
             fp_unit_is_mul(c, &c->units[s->src[2].index])) {
            p_at = 2;
            addend = s->src[0];
         }
      }
      if (p_at < 0)
         continue;

      struct fp_opnd p = s->src[p_at];
      struct fp_unit *m = &c->units[p.index];
      if (!fp_unit_is_mul(c, m) || readers[p.index] != 1 || p.absolute ||
          p.minus_one || p.twice)
         continue;

      struct fp_opnd x = m->slots[0].src[0], y = m->slots[0].src[1];
      if (p.negate)
         x.negate = !x.negate;
      s->src[0] = x;
      s->src[1] = y;
      s->src[2] = addend;
      readers[p.index] = 0;
      m->live = false;
      /* the multiply's operands are read here now */
   }
}


/* What a value is a plain copy of (MAD x, 1, 0, possibly over a chain) */
static struct fp_opnd
fp_copy_source(struct fp_compiler *c, struct fp_opnd o)
{
   while (o.kind == FP_OPND_VALUE && !o.negate && !o.absolute) {
      const struct fp_unit *u = &c->units[o.index];
      const struct fp_unit_slot *s = &u->slots[0];
      if (!fp_unit_has_value(u) || u->num_slots != 1 || u->sfu ||
          u->saturate || s->op != FP_ALU_OP_MAD || s->accumulate_this ||
          s->accumulate_other || !fp_opnd_is_const(c, &s->src[1], 1.0f) ||
          !fp_opnd_is_const(c, &s->src[2], 0.0f))
         break;
      o = s->src[0];
   }
   return o;
}

/*
 * After folding, a texture read may turn out to use a varying after all
 * and the special functions may be gone (preshader, specialization): then
 * the program needs no sequential layout. Liveness is computed afresh.
 */
static void
fp_relayout(struct fp_compiler *c)
{
   for (int n = 0; n < c->num_tex; n++) {
      struct fp_opnd s = fp_copy_source(c, c->tex[n].coord[0]);
      struct fp_opnd t = fp_copy_source(c, c->tex[n].coord[1]);
      if (c->tex[n].input >= 0 || !fp_tex_varying(&s, &t))
         continue;
      if (c->tex[n].num_coords > 2) {
         /* a cube map's r or a projective q of the same varying (moved,
          * move.c): the TEX divides by it, as for one the program reads;
          * a cube map wants s, t, r in .xyz */
         struct fp_opnd r = fp_copy_source(c, c->tex[n].coord[2]);
         if (c->tex[n].num_coords > 3 || c->tex[n].role[2] != FP_TEX_R ||
             r.kind != FP_OPND_INPUT || r.index != s.index || r.negate ||
             r.absolute || r.comp == s.comp || r.comp == t.comp ||
             (c->tex[n].cube && (s.comp != 0 || t.comp != 1 || r.comp != 2)))
            continue;
         c->tex[n].num_coords = 2;
         c->tex[n].r_comp = r.comp;
      }
      c->tex[n].coord[0] = s;
      c->tex[n].coord[1] = t;
      c->tex[n].input = s.index;
      c->tex[n].pair = s.comp / 2;
   }

   for (int i = 0; i < c->num_units; i++)
      c->units[i].live = false;
   for (int n = 0; n < c->num_tex; n++)
      c->tex[n].live = false;
   fp_dead_code(c);

   /* SFUs need a sequential program, but for those on a varying (pseudo
    * inputs of a multi-row one) */
   c->seq = false;
   for (int i = 0; i < c->num_units; i++)
      c->seq |= c->units[i].live && c->units[i].sfu && !fp_rows_sfu(c, &c->units[i]);
   /* and texture fetches of computed coordinates, cube r or a bias */
   for (int n = 0; n < c->num_tex; n++)
      c->seq |= c->tex[n].live && (c->tex[n].input < 0 || c->tex[n].num_coords > 2);
}

/*
 * The uniform slots the program still reads get the hardware's uniform
 * registers: user uniforms and immediates of code specialized away, and
 * preshader results nothing reads any more, take none. Preshader entries
 * keep their index (grate_fp_spec refers to them) and are evaluated all
 * the same.
 */
static void
fp_compact_uniforms(struct fp_compiler *c)
{
   struct grate_fp_info *info = &c->fp->info;
   BITSET_DECLARE(read, GRATE_FP_MAX_SLOTS);
   int map[GRATE_FP_MAX_SLOTS];
   int n = 0;

   BITSET_ZERO(read);
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j)
         if (u->slots[k].src[j].kind == FP_OPND_UNIFORM)
            BITSET_SET(read, u->slots[k].src[j].index);
   }
   /* the vertex shader reads those of the moved coordinates */
   struct grate_moved_src *msrc[GRATE_MOVED_MAX_NODES * 3 + GRATE_MOVED_MAX * 4];
   int num_msrc = 0;
   for (int i = 0; i < info->moved.num_nodes; i++)
      for (int k = 0; k < 3; k++)
         msrc[num_msrc++] = &info->moved.nodes[i][k];
   for (int i = 0; i < info->moved.num_outs; i++)
      for (int k = 0; k < info->moved.out[i].num_roots; k++)
         msrc[num_msrc++] = &info->moved.out[i].root[k];
   for (int i = 0; i < num_msrc; i++)
      if (msrc[i]->kind == GRATE_MOVED_UNIFORM)
         BITSET_SET(read, msrc[i]->index);

   for (int i = 0; i < GRATE_FP_MAX_SLOTS; i++)
      map[i] = BITSET_TEST(read, i) ? n++ : -1;
   if (n > GRATE_FP_NUM_CONSTS) {
      c->fp->info.out_of_uniforms = true;
      fp_fail(c, "too many uniforms");
      return;
   }

   uint32_t consts[GRATE_FP_NUM_CONSTS] = { 0 };
   uint8_t user_index[GRATE_FP_NUM_CONSTS] = { 0 };
   BITSET_DECLARE(imm_mask, GRATE_FP_MAX_SLOTS);
   BITSET_DECLARE(user_mask, GRATE_FP_MAX_SLOTS);
   BITSET_ZERO(imm_mask);
   BITSET_ZERO(user_mask);
   for (int i = 0; i < GRATE_FP_MAX_SLOTS; i++) {
      if (map[i] < 0)
         continue;
      consts[map[i]] = info->consts[i];
      user_index[map[i]] = info->user_index[i];
      if (BITSET_TEST(info->imm_mask, i))
         BITSET_SET(imm_mask, map[i]);
      if (BITSET_TEST(info->user_mask, i))
         BITSET_SET(user_mask, map[i]);
   }

   memset(info->consts, 0, sizeof(info->consts));
   memset(info->user_index, 0, sizeof(info->user_index));
   memcpy(info->consts, consts, sizeof(consts));
   memcpy(info->user_index, user_index, sizeof(user_index));
   BITSET_COPY(info->imm_mask, imm_mask);
   BITSET_COPY(info->user_mask, user_mask);
   BITSET_ZERO(info->used_mask);
   for (int i = 0; i < n; i++)
      BITSET_SET(info->used_mask, i);

   for (int i = 0; i < info->num_pre; i++)
      info->pre[i].dst = map[info->pre[i].dst] >= 0 ? map[info->pre[i].dst] :
                                                      GRATE_FP_PRE_NO_DST;

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j)
         if (u->slots[k].src[j].kind == FP_OPND_UNIFORM)
            u->slots[k].src[j].index = map[u->slots[k].src[j].index];
   }
   for (int i = 0; i < num_msrc; i++)
      if (msrc[i]->kind == GRATE_MOVED_UNIFORM)
         msrc[i]->index = map[msrc[i]->index];
}

/* Drop units whose value is never used */
void
fp_dead_code(struct fp_compiler *c)
{
   for (int i = c->num_units - 1; i >= 0; i--) {
      struct fp_unit *u = &c->units[i];
      if (u->output_chan >= 0 || u->kill)
         u->live = true;
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE)
            c->units[o->index].live = true;
         else if (o->kind == FP_OPND_TEXEL) {
            c->tex[o->index].live = true;
            /* computed coordinates (sequential programs) */
            for (int k = 0; k < c->tex[o->index].num_coords; k++)
               if (c->tex[o->index].coord[k].kind == FP_OPND_VALUE)
                  c->units[c->tex[o->index].coord[k].index].live = true;
         }
      }
   }
}

static void
emit_tgsi_input(struct fp_compiler *c, const struct tgsi_full_declaration *decl)
{
   int index = decl->Range.First;

   grate_trace();
   assert(decl->Range.First == decl->Range.Last);
   if (index >= FP_MAX_INPUTS) {
      fp_fail(c, "too many inputs");
      return;
   }

   c->input_semantic[index] = GRATE_SEMANTIC(decl->Semantic.Name,
                                             decl->Semantic.Index);
   c->input_interp[index] = decl->Interp.Interpolate;
   c->num_inputs = MAX2(c->num_inputs, index + 1);
   if (decl->Declaration.Semantic == TGSI_SEMANTIC_COLOR)
      c->fp->info.color_input = index;
}

static void
emit_tgsi_declaration(struct fp_compiler *c, const struct tgsi_full_declaration *decl)
{
   switch (decl->Declaration.File) {
   case TGSI_FILE_INPUT:
      emit_tgsi_input(c, decl);
      break;
   }
}


/*
 * Control flow. The fragment unit has no jumps: an IF runs both sides, the
 * writes of each merged by its condition (fp_merge), and a loop is unrolled
 * until it is known to have ended. Its exit has to be known when compiling:
 * from constants, or from uniforms, which then make variants compiled for
 * their values (grate_fp_key.flow); a loop left by a test on the pixel runs
 * every iteration up to FP_MAX_LOOP_ITERATIONS, each pixel keeping the
 * values of its last one. The NVIDIA compiler unrolls only loops with a
 * constant count (cgdump shaders/loopuni.glsl, loopdyn.glsl: "profile does
 * not support for statements").
 */

static bool fp_eval_unit(struct fp_compiler *c, int i, float *v);

/* The value of an operand if it is known when compiling */
static bool
fp_eval(struct fp_compiler *c, struct fp_opnd o, float *v)
{
   float x;

   switch (o.kind) {
   case FP_OPND_ZERO:
      x = 0.0f;
      break;
   case FP_OPND_ONE:
      x = 1.0f;
      break;
   case FP_OPND_UNIFORM:
      if (!BITSET_TEST(c->fp->info.imm_mask, o.index))
         return false;
      x = fp20_to_float(c->fp->info.consts[o.index]);
      break;
   case FP_OPND_VALUE:
      if (!fp_eval_unit(c, o.index, &x))
         return false;
      break;
   default:
      return false;
   }
   if (o.absolute)
      x = fabsf(x);
   if (o.negate)
      x = -x;
   *v = x;
   return true;
}

static bool
fp_eval_unit(struct fp_compiler *c, int i, float *v)
{
   struct fp_unit *u = &c->units[i];
   struct fp_unit_slot *s = &u->slots[0];
   float a, b, cc, x;
   bool ok = false;

   if (c->eval[i]) {
      *v = c->eval_v[i];
      return c->eval[i] > 0;
   }

   if (u->kill || u->output_chan >= 0) {
      ok = false;
   } else if (u->num_slots == 0) {
      ok = fp_eval(c, s->src[0], &a);
      switch (u->sfu) {
      case FP_SFU_OP_RCP:  x = 1.0f / a; break;
      case FP_SFU_OP_RSQ:  x = 1.0f / sqrtf(a); break;
      case FP_SFU_OP_SQRT: x = sqrtf(a); break;
      case FP_SFU_OP_LG2:  x = log2f(a); break;
      case FP_SFU_OP_FRC:  x = a - floorf(a); break;
      default:             ok = false; break;
      }
   } else if (u->num_slots == 1 && !s->accumulate_this && !s->accumulate_other &&
              !s->square_c) {
      switch (s->op) {
      case FP_ALU_OP_CSEL:
         ok = fp_eval(c, s->src[0], &a) && fp_eval(c, s->src[a < 0.0f ? 1 : 2], &x);
         break;
      case FP_ALU_OP_MAD:
      case FP_ALU_OP_MIN:
      case FP_ALU_OP_MAX:
         ok = fp_eval(c, s->src[0], &a) && fp_eval(c, s->src[1], &b) &&
              fp_eval(c, s->src[2], &cc);
         x = s->op == FP_ALU_OP_MAD ? a * b + cc :
             s->op == FP_ALU_OP_MIN ? fminf(a * b, cc) : fmaxf(a * b, cc);
         break;
      default:
         break;
      }
   }
   if (ok && u->saturate)
      x = CLAMP(x, 0.0f, 1.0f);

   c->eval[i] = ok ? 1 : -1;
   c->eval_v[i] = ok ? x : 0.0f;
   *v = x;
   return ok;
}

/* Whether an operand depends on uniforms only; *users: which */
static bool
fp_deps(struct fp_compiler *c, struct fp_opnd o, BITSET_WORD *users)
{
   switch (o.kind) {
   case FP_OPND_ZERO:
   case FP_OPND_ONE:
      return true;
   case FP_OPND_UNIFORM:
      if (BITSET_TEST(c->fp->info.user_mask, o.index))
         BITSET_SET(users, c->fp->info.user_index[o.index]);
      return true;
   case FP_OPND_VALUE:
      break;
   default:
      return false;
   }

   int i = o.index;
   if (!c->dep[i]) {
      struct fp_unit *u = &c->units[i];
      bool uniform = true;
      BITSET_ZERO(c->dep_users[i]);
      fp_unit_foreach_src(u, k, j)
         uniform &= fp_deps(c, u->slots[k].src[j], c->dep_users[i]);
      c->dep[i] = uniform ? 1 : -1;
   }
   __bitset_or(users, users, c->dep_users[i], BITSET_WORDS(GRATE_FP_USER_CONSTS));
   return c->dep[i] > 0;
}

/* A predicate of value p (-1.0: true, 0.0: false), constant if known */
static struct fp_pred
fp_pred_value(struct fp_compiler *c, struct fp_opnd p)
{
   float v;

   if (fp_eval(c, p, &v))
      return (struct fp_pred) { v < 0.0f ? FP_PRED_ALWAYS : FP_PRED_NEVER };
   return (struct fp_pred) { FP_PRED_VALUE, p };
}

static struct fp_opnd
fp_opnd_minus_one(struct fp_compiler *c)
{
   return fp_opnd_modify(fp_opnd_one(c, false), true, false);
}

/* IF x: x != 0, as -|x| < 0 */
static struct fp_pred
fp_pred_cond(struct fp_compiler *c, struct fp_opnd x)
{
   float v;

   if (fp_eval(c, x, &v))
      return (struct fp_pred) { v != 0.0f ? FP_PRED_ALWAYS : FP_PRED_NEVER };
   return fp_pred_value(c, fp_val(c, FP_ALU_OP_CSEL, fp_opnd_modify(x, true, true),
                                  fp_opnd_minus_one(c), fp_opnd_zero(c, false)));
}

static struct fp_pred
fp_pred_not(struct fp_compiler *c, struct fp_pred a)
{
   if (a.kind != FP_PRED_VALUE)
      return (struct fp_pred) { a.kind == FP_PRED_ALWAYS ? FP_PRED_NEVER : FP_PRED_ALWAYS };
   return fp_pred_value(c, fp_val(c, FP_ALU_OP_CSEL, a.v, fp_opnd_zero(c, false),
                                  fp_opnd_minus_one(c)));
}

static struct fp_pred
fp_pred_and(struct fp_compiler *c, struct fp_pred a, struct fp_pred b)
{
   if (a.kind != FP_PRED_VALUE)
      return a.kind == FP_PRED_ALWAYS ? b : a;
   if (b.kind != FP_PRED_VALUE)
      return b.kind == FP_PRED_ALWAYS ? a : b;
   return fp_pred_value(c, fp_val(c, FP_ALU_OP_CSEL, a.v, b.v, fp_opnd_zero(c, false)));
}

/* The predicate of the flow entries from "first" on; full: also whether
 * the pixels are still in the loops */
static struct fp_pred
fp_pred_from(struct fp_compiler *c, int first, bool full)
{
   struct fp_pred p = { FP_PRED_ALWAYS };

   for (int i = first; i < c->flow_depth; i++) {
      if (c->flow[i].loop) {
         if (full)
            p = fp_pred_and(c, p, c->flow[i].alive);
         p = fp_pred_and(c, p, c->flow[i].iter);
      } else {
         p = fp_pred_and(c, p, c->flow[i].cond);
      }
   }
   return p;
}

static void
fp_update_pred(struct fp_compiler *c)
{
   c->pred = fp_pred_from(c, 0, false);
   c->pred_full = fp_pred_from(c, 0, true);
}

/* Pixels of q (in loop f) leave it with the values they have now */
static void
fp_keep_exit(struct fp_compiler *c, struct fp_flow *f, struct fp_pred q)
{
   for (int t = 0; t < FP_MAX_TEMPS; t++)
      for (int ch = 0; ch < 4; ch++) {
         if (!c->temp_written[t][ch])
            continue;
         struct fp_opnd old = f->exit_written[t][ch] ? f->exit[t][ch] : fp_opnd_zero(c, false);
         f->exit[t][ch] = q.kind == FP_PRED_ALWAYS ? c->temps[t][ch] :
                          fp_val(c, FP_ALU_OP_CSEL, q.v, c->temps[t][ch], old);
         f->exit_written[t][ch] = true;
      }
   for (int ch = 0; ch < 4; ch++) {
      if (!c->outv_written[ch])
         continue;
      struct fp_opnd old = f->exit_outv_written[ch] ? f->exit_outv[ch] : fp_opnd_zero(c, false);
      f->exit_outv[ch] = q.kind == FP_PRED_ALWAYS ? c->outv[ch] :
                         fp_val(c, FP_ALU_OP_CSEL, q.v, c->outv[ch], old);
      f->exit_outv_written[ch] = true;
   }
}

/* BRK or CONT: the pixels running it leave the loop (iteration) */
static void
fp_leave_loop(struct fp_compiler *c, bool brk)
{
   int l = c->flow_depth - 1;
   while (l >= 0 && !c->flow[l].loop)
      l--;
   if (l < 0) {
      fp_fail(c, "BRK/CONT outside a loop");
      return;
   }

   /* the pixels running it: in the IFs it is in, inside the loop */
   struct fp_pred q = fp_pred_and(c, c->flow[l].iter, fp_pred_from(c, l + 1, false));
   if (q.kind == FP_PRED_VALUE && brk) {
      BITSET_DECLARE(users, GRATE_FP_USER_CONSTS);
      BITSET_ZERO(users);
      if (fp_deps(c, q.v, users) && !BITSET_IS_EMPTY(users)) {
         /* the loop count comes from uniforms: compile for their values */
         struct grate_fp_info *info = &c->fp->info;
         unsigned user;
         BITSET_FOREACH_SET(user, users, GRATE_FP_USER_CONSTS) {
            bool have = false;
            for (int i = 0; i < info->num_flow; i++)
               have |= info->flow_user[i] == user;
            if (!have && info->num_flow < GRATE_FP_MAX_FLOW)
               info->flow_user[info->num_flow++] = user;
         }
         c->needs_flow = true;
         c->failed = true;
         return;
      }
   }

   struct fp_pred stay = fp_pred_not(c, q);
   if (brk) {
      if (q.kind != FP_PRED_NEVER)
         fp_keep_exit(c, &c->flow[l], fp_pred_and(c, c->flow[l].alive, q));
      c->flow[l].alive = fp_pred_and(c, c->flow[l].alive, stay);
   } else {
      c->flow[l].iter = fp_pred_and(c, c->flow[l].iter, stay);
   }
   fp_update_pred(c);
}

struct fp_program {
   struct tgsi_full_instruction *insts;
   int *match;          /* IF: ELSE or ENDIF, ELSE: ENDIF, BGNLOOP: ENDLOOP */
   int num;
};

static void
fp_run(struct fp_compiler *c, const struct fp_program *p, int start, int end)
{
   for (int pc = start; pc < end && !c->failed; pc++) {
      const struct tgsi_full_instruction *inst = &p->insts[pc];

      switch (inst->Instruction.Opcode) {
      case TGSI_OPCODE_IF:
      case TGSI_OPCODE_UIF: {
         int els = p->match[pc];
         if (els < 0) {
            fp_fail(c, "unmatched IF");
            return;
         }
         bool has_else = p->insts[els].Instruction.Opcode == TGSI_OPCODE_ELSE;
         int endif = has_else ? p->match[els] : els;

         if (c->pred_full.kind != FP_PRED_NEVER) {
            if (c->flow_depth == FP_MAX_FLOW_DEPTH) {
               fp_fail(c, "control flow too deep");
               return;
            }
            struct fp_pred cond = fp_pred_cond(c, fp_src(c, &inst->Src[0].Register, 0));
            int d = c->flow_depth++;
            c->flow[d].loop = false;
            c->flow[d].cond = cond;
            fp_update_pred(c);
            fp_run(c, p, pc + 1, els);
            if (has_else) {
               c->flow[d].cond = fp_pred_not(c, cond);
               fp_update_pred(c);
               fp_run(c, p, els + 1, endif);
            }
            c->flow_depth--;
            fp_update_pred(c);
         }
         pc = endif;
         break;
      }

      case TGSI_OPCODE_BGNLOOP: {
         int endloop = p->match[pc];
         if (endloop < 0) {
            fp_fail(c, "unmatched BGNLOOP");
            return;
         }

         if (c->pred_full.kind != FP_PRED_NEVER) {
            if (c->flow_depth == FP_MAX_FLOW_DEPTH) {
               fp_fail(c, "control flow too deep");
               return;
            }
            int d = c->flow_depth++;
            struct fp_flow *f = &c->flow[d];
            f->loop = true;
            f->alive = (struct fp_pred) { FP_PRED_ALWAYS };
            memset(f->exit_written, 0, sizeof(f->exit_written));
            memset(f->exit_outv_written, 0, sizeof(f->exit_outv_written));
            for (int n = 0; ; n++) {
               if (n == (f->alive.kind == FP_PRED_VALUE ? FP_MAX_LOOP_ITERATIONS :
                         FP_MAX_KNOWN_LOOP_ITERATIONS)) {
                  fp_fail(c, "loop runs too long to unroll");
                  return;
               }
               c->flow[d].iter = (struct fp_pred) { FP_PRED_ALWAYS };
               fp_update_pred(c);
               fp_run(c, p, pc + 1, endloop);
               if (c->failed || f->alive.kind == FP_PRED_NEVER)
                  break;
            }
            /* every pixel has left: with the values it had then */
            memcpy(c->temps, f->exit, sizeof(c->temps));
            memcpy(c->temp_written, f->exit_written, sizeof(c->temp_written));
            memcpy(c->outv, f->exit_outv, sizeof(c->outv));
            memcpy(c->outv_written, f->exit_outv_written, sizeof(c->outv_written));
            c->flow_depth--;
            fp_update_pred(c);
         }
         pc = endloop;
         break;
      }

      case TGSI_OPCODE_BRK:
      case TGSI_OPCODE_CONT:
         if (c->pred_full.kind != FP_PRED_NEVER)
            fp_leave_loop(c, inst->Instruction.Opcode == TGSI_OPCODE_BRK);
         break;

      default:
         if (c->pred_full.kind != FP_PRED_NEVER)
            emit_tgsi_instr(c, inst);
         break;
      }
   }
}

/* Pair the control flow instructions; false if they are not nested right */
static bool
fp_match_flow(struct fp_program *p)
{
   int stack[FP_MAX_FLOW_DEPTH * 2], depth = 0;
   bool flow = false;

   for (int pc = 0; pc < p->num; pc++) {
      p->match[pc] = -1;
      switch (p->insts[pc].Instruction.Opcode) {
      case TGSI_OPCODE_IF:
      case TGSI_OPCODE_UIF:
      case TGSI_OPCODE_BGNLOOP:
         if (depth == ARRAY_SIZE(stack))
            return false;
         stack[depth++] = pc;
         flow = true;
         break;
      case TGSI_OPCODE_ELSE:
         if (!depth)
            return false;
         p->match[stack[depth - 1]] = pc;
         stack[depth - 1] = pc;
         break;
      case TGSI_OPCODE_ENDIF:
      case TGSI_OPCODE_ENDLOOP:
         if (!depth)
            return false;
         p->match[stack[--depth]] = pc;
         break;
      case TGSI_OPCODE_BRK:
      case TGSI_OPCODE_CONT:
         flow = true;
         break;
      }
   }
   return !depth && flow;
}

/*
 * A cube map or projective fetch of a varying in a program that ends up
 * sequential: the MFU divides in the TEX EXEC there too, on copies of the
 * components the ALU writes (emit_tex)
 */
static void
fp_tex_div_computed(struct fp_compiler *c, int n)
{
   struct fp_opnd in = { .kind = FP_OPND_INPUT, .index = c->tex[n].input };

   /* only s and t so far: a bias is not on varyings (emit_tex) */
   for (int k = 0; k < 3; k++) {
      struct fp_unit *u = fp_new_unit(c);
      in.comp = k < 2 ? c->tex[n].coord[k].comp : c->tex[n].r_comp;
      fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, in, fp_opnd_one(c, false),
                 fp_opnd_zero(c, false));
      u->live = true;
      c->tex[n].coord[k] = fp_opnd_value(c, u);
      c->tex[n].role[k] = k;
   }
   c->tex[n].num_coords = 3;
   c->tex[n].input = -1;
   c->tex[n].pair = 0;
}

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi,
                 const struct grate_fp_key *key, bool trunc_abs, int max_pre)
{
   static const struct grate_fp_key key_none = { .colormask = 0xf };
   struct fp_compiler *c = CALLOC_STRUCT(fp_compiler);
   struct fp_program prog = { 0 };
   int max_insts = 0;

   if (!key)
      key = &key_none;
   c->key = key;
   c->trunc_abs = trunc_abs;
   c->max_pre = CLAMP(max_pre, 0, GRATE_FP_MAX_PRE);
   c->out_pair = 1;
   c->redirect_output = key && (key->blend || (key->colormask & 0xf) != 0xf);

   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);

   fp->info.num_links = 0;
   fp->info.color_input = -1;
   fp->info.max_tram_row = 1;
   BITSET_ZERO(fp->info.used_mask);
   BITSET_ZERO(fp->info.imm_mask);
   BITSET_ZERO(fp->info.user_mask);
   fp->info.sampler_mask = 0;
   fp->info.varying_trunc = false;
   fp->info.out_of_uniforms = false;
   fp->info.lod_bias_mask = 0;
   fp->info.lod_bias_user_mask = fp->info.lod_bias_user_neg = 0;
   memset(fp->info.lod_bias, 0, sizeof(fp->info.lod_bias));
   fp->info.pre = c->pre;
   fp->info.num_pre = 0;
   /* the uniforms of the variant's key, then any a loop needs as well (a
    * counter stepped by another uniform shows only once the first is
    * known): grate_fs_select compiles again for all of them */
   fp->info.num_flow = key->num_flow;
   memcpy(fp->info.flow_user, key->flow_user, sizeof(fp->info.flow_user));
   fp->num_immediates = 0;

   c->fp = fp;
   memset(c->tram, -1, sizeof(c->tram));

   while (!tgsi_parse_end_of_tokens(tgsi)) {
      tgsi_parse_token(tgsi);
      switch (tgsi->FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_DECLARATION:
         emit_tgsi_declaration(c, &tgsi->FullToken.FullDeclaration);
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
         if (tgsi->FullToken.FullInstruction.Instruction.Opcode == TGSI_OPCODE_END)
            break;
         if (prog.num == max_insts) {
            max_insts = MAX2(64, max_insts * 2);
            prog.insts = REALLOC(prog.insts, prog.num * sizeof(*prog.insts),
                                 max_insts * sizeof(*prog.insts));
         }
         prog.insts[prog.num++] = tgsi->FullToken.FullInstruction;
         break;
      }
   }

   prog.match = MALLOC(MAX2(prog.num, 1) * sizeof(int));
   if (fp_match_flow(&prog)) {
      /* colour writes are merged as values too */
      c->redirect_output = true;
   }
   c->pred = c->pred_full = (struct fp_pred) { FP_PRED_ALWAYS };
   fp_run(c, &prog, 0, prog.num);
   FREE(prog.match);
   FREE(prog.insts);
   /* the generic version of a program compiled per loop count: unused */
   if (c->needs_flow)
      fp->info.failed = true;

   if (c->redirect_output && !c->failed)
      emit_blend(c);

   fp_cse(c);
   fp_reassociate(c);
   fp_cse(c);
   fp_fold_constants(c);
   fp_preshade(c);
   fp_specialize(c);
   fp_fold_copies(c);
   fp_fuse_mad(c);
   fp_fold_scales(c);
   fp_fold_conditions(c);
   fp_dead_code(c);
   fp_fold_sfu_copies(c);
   fp_move_coords(c);
   fp_relayout(c);
   if (!c->failed)
      fp_compact_uniforms(c);

   /*
    * Multi-row programs have more registers, sequential ones room for more
    * instructions (16 EXECs instead of 4): fall back to the latter when the
    * former run out of packets.
    */
   /* GRATE_FP_SEQ=1: sequential programs only (to try them out) */
   if (getenv("GRATE_FP_SEQ") && atoi(getenv("GRATE_FP_SEQ")))
      c->seq = true;
   if (!c->failed && !c->seq && !fp_sched_rows(c))
      c->seq = true;
   for (int n = 0; n < c->num_tex && !c->failed && c->seq; n++)
      if (c->tex[n].live && (c->tex[n].cube || c->tex[n].proj) &&
          fp_tex_coord(c, n, FP_TEX_R) < 0)
         fp_tex_div_computed(c, n);
   if (!c->failed && c->seq) {
      /* and the sum it adds may take in a multiply, as when written so */
      fp_expand_blend_inv(c);
      fp_fuse_mad(c);
   }
   if (!c->failed && c->seq && !fp_sched_seq(c))
      fp_fail(c, "shader too big (out of ALU packets or registers)");

   if ((grate_debug & GRATE_DEBUG_TGSI) && !c->seq)
      fp_dump_rows(c);

   if (c->failed) {
      /* still emit something sane: magenta, one packet */
      memset(c->exec_input, -1, sizeof(c->exec_input));
      memset(c->tram, -1, sizeof(c->tram));
      c->num_units = 0;
      c->num_tex = 0;
      c->dst_used = false;
      c->seq = false;
      for (int i = 0; i < 4; i++) {
         struct fp_unit *u = fp_new_unit(c);
         struct fp_opnd one = fp_opnd_one(c, true);
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, i == 1 ? fp_opnd_zero(c, true) : one,
                    one, fp_opnd_zero(c, true));
         u->output_chan = i;
         u->live = true;
         u->packet = 0;
         u->slot = i;
      }
      c->dw.enable = 1;
      c->dw.src_regs = FP_DW_REGS_R2_R3;
      c->num_packets = 1;
      c->num_execs = 1;
      c->packet_exec[0] = 0;
   }

   if (c->seq)
      fp_emit_seq(c);
   else
      fp_emit_rows(c);
   fp_embed_constants(fp);
   fp_build_links(c);
   fp->info.num_rows = c->seq ? MAX2(c->seq_rows, 1) : c->num_execs;
   fp->info.failed = c->failed;
   fp->info.pre = fp->info.num_pre ?
                  mem_dup(c->pre, fp->info.num_pre * sizeof(c->pre[0])) : NULL;
   if (!fp->info.pre && fp->info.num_pre) {
      fp->info.num_pre = 0;
      fp->info.failed = true;
   }
   FREE(c);
}

void
grate_fp_info_fini(struct grate_fp_info *info)
{
   FREE(info->pre);
   info->pre = NULL;
   info->num_pre = 0;
}
