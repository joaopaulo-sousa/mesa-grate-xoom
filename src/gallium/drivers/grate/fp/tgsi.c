#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "grate_common.h"
#include "grate_compiler.h"
#include "fpir.h"

#include "tgsi/tgsi_parse.h"

#include "util/blend.h"
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
 * Fragment program layout (measured on Tegra20, see gpu/tools/asmtri):
 *
 * - The program is N "EXECs" (N <= 4) and the pixel packet has N rows, row k
 *   being the registers r4k..r4k+3. Each EXEC has an optional MFU instruction,
 *   up to 3 ALU packets of 4 scalar slots and, in the first EXEC, the DW
 *   store, which runs after the last EXEC.
 * - The MFU of EXEC k interpolates varying component c into r(4k+c); the ALU
 *   can read it from EXEC k on.
 * - A register written by ALU packet P can be read from packet P+2 on (the
 *   packets are counted across EXECs). Packet P+1 reads the result of slot s
 *   of packet P as "alu<s>" instead.
 * - The PSEQ of EXEC k loads the render target pixel (the destination colour)
 *   as fixed10 halves into a register pair of row k before that EXEC's ALU:
 *   blending is done in the shader (Tegra20 has no blend unit).
 * - The TEX of EXEC k samples with the coordinates in a register pair of row
 *   k, r(4k)/r(4k+1) or r(4k+2)/r(4k+3), and writes the texel as four fixed10
 *   halves to the other pair, readable by the ALU from EXEC k on. Only
 *   coordinates that are a varying are supported (the NVIDIA compiler uses
 *   one-row programs for dependent reads).
 *
 * TGSI is lowered to scalar units (one slot, or slots 0..n-1 of a packet for
 * a DPn) in SSA form, the units are list-scheduled into packets and EXECs
 * honouring the latencies above, and the values that have to outlive the next
 * packet get a register. The colour is written as fixed10 to r2/r3.
 */

#define FP_MAX_UNITS 128
#define FP_MAX_TEMPS 32
#define FP_MAX_INPUTS 16
#define FP_MAX_EXECS 4
#define FP_MAX_PACKETS_PER_EXEC 3
#define FP_MAX_PACKETS 32
#define FP_MAX_TEXTURES FP_MAX_EXECS

#define FP_ALU_RESULT_REG(s) (24 + (s))

enum fp_opnd_kind {
   FP_OPND_UNIFORM = 0,   /* ALU uniform register (constants, immediates) */
   FP_OPND_VALUE,         /* result of a unit */
   FP_OPND_INPUT,         /* interpolated varying component */
   FP_OPND_TEXEL,         /* texture sample channel */
   FP_OPND_DST,           /* destination colour channel (PSEQ load) */
   FP_OPND_ZERO,          /* fixed10 0 and 1 of r31, for fixed10 results */
   FP_OPND_ONE,
};

struct fp_opnd {
   enum fp_opnd_kind kind;
   int index;      /* uniform slot, unit, TGSI input or texture fetch */
   int comp;       /* input component, texel channel */
   bool negate;
   bool absolute;
};

struct fp_unit_slot {
   enum fp_alu_op op;
   struct fp_opnd src[3];
   bool accumulate_this;
   bool accumulate_other;
};

struct fp_unit {
   int num_slots;          /* 1, or a DPn taking slots 0..n-1 */
   struct fp_unit_slot slots[4];
   bool saturate;
   int output_chan;        /* colour channel written, or -1: defines a value */
   bool forward;           /* the next packet may read the result as aluN */
   bool live;

   /* scheduling */
   int packet;
   int slot;
   int reg;                /* register holding the value, -1 if none */
};

struct fp_compiler {
   struct grate_fp_shader *fp;
   struct fp_unit units[FP_MAX_UNITS];
   int num_units;
   struct fp_opnd temps[FP_MAX_TEMPS][4];
   bool temp_written[FP_MAX_TEMPS][4];
   struct fp_dw_instr dw;
   bool failed;

   /* EXEC whose MFU fetches TGSI input i component c, and the reverse */
   int fetch_exec[FP_MAX_INPUTS][4];
   int exec_input[FP_MAX_EXECS][4];

   /* blending: colour writes become outv[] values, combined with the
    * destination (PSEQ load into a register pair of EXEC dst_exec) */
   const struct grate_fp_key *key;
   bool redirect_output;
   struct fp_opnd outv[4];
   bool outv_written[4];
   bool dst_used;
   int dst_exec, dst_pair;

   /* texture fetches: coordinates = TGSI input components 2*pair, +1 */
   struct {
      int input;
      int pair;
      int sampler;
      int exec;
      bool live;
   } tex[FP_MAX_TEXTURES];
   int num_tex;

   int packet_exec[FP_MAX_PACKETS];
   int num_packets;
   int num_execs;
};

static void
fp_fail(struct fp_compiler *c, const char *msg)
{
   if (!c->failed)
      fprintf(stderr, "GRATE FRAG: %s\n", msg);
   c->failed = true;
}

static struct fp_unit *
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
   return u;
}

static struct fp_opnd
fp_opnd_uniform(int slot)
{
   struct fp_opnd o = { .kind = FP_OPND_UNIFORM, .index = slot };
   return o;
}

static struct fp_opnd
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
static struct fp_opnd
fp_opnd_zero(struct fp_compiler *c, bool output)
{
   if (output) {
      struct fp_opnd o = { .kind = FP_OPND_ZERO };
      return o;
   }
   return fp_opnd_uniform(fp_const_slot(c->fp, 0.0f));
}

static struct fp_opnd
fp_opnd_one(struct fp_compiler *c, bool output)
{
   if (output) {
      struct fp_opnd o = { .kind = FP_OPND_ONE };
      return o;
   }
   return fp_opnd_uniform(fp_const_slot(c->fp, 1.0f));
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
      o.kind = FP_OPND_INPUT;
      o.index = src->Index;
      o.comp = comp;
      /* rows stored per triangle: index + 1 (only one row was kept with two
       * varyings, the second read garbage) */
      fp->info.max_tram_row = MAX2(fp->info.max_tram_row, src->Index + 1);
      break;

   case TGSI_FILE_IMMEDIATE: {
      assert(src->Index < fp->num_immediates);
      float value = fp->immediates[src->Index][comp];
      if (src->Absolute)
         value = fabsf(value);
      if (src->Negate)
         value = -value;
      return fp_opnd_uniform(fp_const_slot(fp, value));
   }

   case TGSI_FILE_CONSTANT:
      o = fp_opnd_uniform(fp_uniform_slot(fp, src->Index, comp));
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

/* Unit writing "chan" of a TGSI destination; returns whether it is a colour
 * write. A temporary channel becomes the unit's value. */
static bool
fp_dst(struct fp_compiler *c, struct fp_unit *u,
       const struct tgsi_dst_register *dst, int chan)
{
   switch (dst->File) {
   case TGSI_FILE_OUTPUT:
      c->dw.enable = 1;
      c->dw.index = dst->Index;
      c->dw.stencil_write = 0;
      c->dw.src_regs = FP_DW_REGS_R2_R3;
      if (c->redirect_output) {
         c->outv[chan] = fp_opnd_value(c, u);
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
      c->temps[dst->Index][chan] = fp_opnd_value(c, u);
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
      c->temps[dst->Index][chan] = o;
      c->temp_written[dst->Index][chan] = true;
      return true;
   }
   if (dst->File == TGSI_FILE_OUTPUT && c->redirect_output) {
      c->dw.enable = 1;
      c->dw.index = dst->Index;
      c->dw.src_regs = FP_DW_REGS_R2_R3;
      c->outv[chan] = o;
      c->outv_written[chan] = true;
      return true;
   }
   return false;
}

static void
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
 * DP2/DP3/DP4 in slots 0..n-1 of one packet, slot k computing a[k] * b[k].
 * Measured on Tegra20: "send" on slot 1 and 2 adds their products to slot 0,
 * "recv" on slot 0 adds slot 3's product. Slot 0 then holds the sum; whether
 * the next packet sees the sum as alu0 is not known, so it is only read back
 * from its register.
 */
static void
emit_dot(struct fp_compiler *c, const struct tgsi_full_instruction *inst,
         int n)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   struct fp_unit *u = fp_new_unit(c);

   u->num_slots = n;
   u->forward = false;
   for (int k = 0; k < n; k++) {
      struct fp_unit_slot *s = &u->slots[k];
      fp_unit_op(s, FP_ALU_OP_MAD, fp_src(c, &inst->Src[0].Register, k),
                 fp_src(c, &inst->Src[1].Register, k), fp_opnd_zero(c, false));
      s->accumulate_other = k == 1 || k == 2;
      s->accumulate_this = k == 0 && n == 4;
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

/*
 * TEX with varying coordinates: one texture fetch per EXEC, the texel
 * channels become fixed10 operands. A colour destination gets copies.
 */
static void
emit_tex(struct fp_compiler *c, const struct tgsi_full_instruction *inst)
{
   const struct tgsi_dst_register *dst = &inst->Dst[0].Register;
   struct fp_opnd s = fp_src(c, &inst->Src[0].Register, 0);
   struct fp_opnd t = fp_src(c, &inst->Src[0].Register, 1);

   if (inst->Texture.Texture != TGSI_TEXTURE_2D &&
       inst->Texture.Texture != TGSI_TEXTURE_RECT) {
      fp_fail(c, "only 2D textures are supported");
      return;
   }
   if (s.kind != FP_OPND_INPUT || t.kind != FP_OPND_INPUT ||
       s.index != t.index || s.negate || s.absolute || t.negate ||
       t.absolute || (s.comp & 1) || t.comp != s.comp + 1) {
      fp_fail(c, "texture coordinates must be a varying .xy or .zw "
                 "(dependent texture reads are not supported)");
      return;
   }
   if (c->num_tex == FP_MAX_TEXTURES) {
      fp_fail(c, "too many texture fetches");
      return;
   }

   int n = c->num_tex++;
   c->tex[n].input = s.index;
   c->tex[n].pair = s.comp / 2;
   c->tex[n].sampler = inst->Src[1].Register.Index;
   c->tex[n].exec = -1;
   c->tex[n].live = false;

   for (int i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      struct fp_opnd texel = { .kind = FP_OPND_TEXEL, .index = n, .comp = i };
      if (!inst->Instruction.Saturate && fp_set_dst_opnd(c, dst, i, texel))
         continue;
      bool output = fp_color_write(c, dst);
      struct fp_unit *m = fp_new_unit(c);
      m->saturate = inst->Instruction.Saturate != 0;
      fp_unit_op(&m->slots[0], FP_ALU_OP_MAD, texel, fp_opnd_one(c, output),
                 fp_opnd_zero(c, output));
      fp_dst(c, m, dst, i);
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
      emit_tex(c, inst);
      break;

   case TGSI_OPCODE_SLT:
   case TGSI_OPCODE_SGE:
   case TGSI_OPCODE_SEQ:
   case TGSI_OPCODE_SNE:
      emit_setcc(c, inst);
      break;

   default:
      fprintf(stderr, "GRATE FRAG TGSI UNIMPLEMENTED: 0x%02x\n", inst->Instruction.Opcode);
      c->failed = true;
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
 * x * blend factor for channel ch; *zero when the term vanishes. s/d are the
 * source and destination colour operands.
 */
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
   default:
      /* constant colour, SRC1, SRC_ALPHA_SATURATE: not yet */
      fprintf(stderr, "GRATE FRAG: blend factor %u unsupported, using ONE\n", factor);
      return x;
   }
   /* x * (1 - y) = x - x * y */
   if (inv)
      return fp_val(c, FP_ALU_OP_MAD, x, fp_opnd_modify(y, true, false), x);
   return fp_val(c, FP_ALU_OP_MAD, x, y, fp_opnd_zero(c, false));
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
         bool zs, zd;
         c->dst_used = true;
         struct fp_opnd ts = fp_blend_term(c, s[ch], sf, ch, s, d, &zs);
         struct fp_opnd td = fp_blend_term(c, d[ch], df, ch, s, d, &zd);
         struct fp_opnd a = zero, b = zero;
         if (!zs)
            a = ts;
         if (!zd)
            b = td;
         if (func == PIPE_BLEND_SUBTRACT)
            b = fp_opnd_modify(b, true, false);
         else if (func == PIPE_BLEND_REVERSE_SUBTRACT)
            a = fp_opnd_modify(a, true, false);
         u = fp_new_unit(c);
         u->saturate = true;
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, a, one, b);
      }
      u->output_chan = ch;
   }
}

#define fp_unit_foreach_src(u, k, j) \
   for (int k = 0; k < (u)->num_slots; k++) \
      for (int j = 0; j < 3; j++)

/* Drop units whose value is never used */
static void
fp_dead_code(struct fp_compiler *c)
{
   for (int i = c->num_units - 1; i >= 0; i--) {
      struct fp_unit *u = &c->units[i];
      if (u->output_chan >= 0)
         u->live = true;
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE)
            c->units[o->index].live = true;
         else if (o->kind == FP_OPND_TEXEL)
            c->tex[o->index].live = true;
      }
   }
}

/* Give every varying component read an MFU of the earliest EXEC that has its
 * component slot free, in order of first use */
static void
fp_assign_inputs(struct fp_compiler *c)
{
   memset(c->fetch_exec, -1, sizeof(c->fetch_exec));
   memset(c->exec_input, -1, sizeof(c->exec_input));

   /*
    * Texture fetches first, one EXEC each: the coordinate pair is fetched by
    * that EXEC's MFU, the other pair receives the texel (-2: no varying).
    */
   int e = 0;
   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live)
         continue;
      int p = c->tex[n].pair, in = c->tex[n].input;
      c->tex[n].exec = e;
      for (int k = 0; k < 2; k++) {
         c->exec_input[e][2 * p + k] = in;
         if (c->fetch_exec[in][2 * p + k] < 0)
            c->fetch_exec[in][2 * p + k] = e;
         c->exec_input[e][2 * (1 - p) + k] = -2;
      }
      e++;
   }

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind != FP_OPND_INPUT || c->fetch_exec[o->index][o->comp] >= 0)
            continue;
         for (e = 0; e < FP_MAX_EXECS; e++)
            if (c->exec_input[e][o->comp] == -1)
               break;
         if (e == FP_MAX_EXECS) {
            fp_fail(c, "too many varyings");
            return;
         }
         c->exec_input[e][o->comp] = o->index;
         c->fetch_exec[o->index][o->comp] = e;
      }
   }

   /* the destination colour: a free register pair of the earliest EXEC */
   if (c->dst_used) {
      for (e = 0; e < FP_MAX_EXECS; e++)
         for (int p = 0; p < 2; p++)
            if (c->exec_input[e][2 * p] == -1 && c->exec_input[e][2 * p + 1] == -1) {
               c->dst_exec = e;
               c->dst_pair = p;
               c->exec_input[e][2 * p] = c->exec_input[e][2 * p + 1] = -2;
               return;
            }
      fp_fail(c, "no registers left for the destination colour");
   }
}

/* fixed10 colour channel -> r2/r3 half, BGRA (also the texel layout) */
static int
fp_output_half(int chan)
{
   return chan < 3 ? (2 - chan) : 3;
}

/* EXEC from which an input or texel operand can be read */
static int
fp_opnd_exec(struct fp_compiler *c, const struct fp_opnd *o)
{
   if (o->kind == FP_OPND_INPUT)
      return c->fetch_exec[o->index][o->comp];
   if (o->kind == FP_OPND_TEXEL)
      return c->tex[o->index].exec;
   if (o->kind == FP_OPND_DST)
      return c->dst_exec;
   return 0;
}

/* Row register holding an input component or a texel channel */
static int
fp_opnd_reg(struct fp_compiler *c, const struct fp_opnd *o)
{
   if (o->kind == FP_OPND_INPUT)
      return 4 * c->fetch_exec[o->index][o->comp] + o->comp;
   if (o->kind == FP_OPND_DST)
      return 4 * c->dst_exec + 2 * c->dst_pair + fp_output_half(o->comp) / 2;
   assert(o->kind == FP_OPND_TEXEL);
   int base = c->tex[o->index].pair ? 0 : 2;
   return 4 * c->tex[o->index].exec + base + fp_output_half(o->comp) / 2;
}

static bool
fp_unit_ready(struct fp_compiler *c, int index, int packet, int exec)
{
   struct fp_unit *u = &c->units[index];

   fp_unit_foreach_src(u, k, j) {
      struct fp_opnd *o = &u->slots[k].src[j];
      if (o->kind == FP_OPND_VALUE) {
         struct fp_unit *d = &c->units[o->index];
         if (d->packet < 0 || d->packet >= packet)
            return false;
         if (!d->forward && d->packet == packet - 1)
            return false;
      } else if (o->kind == FP_OPND_INPUT || o->kind == FP_OPND_TEXEL ||
                 o->kind == FP_OPND_DST) {
         if (fp_opnd_exec(c, o) > exec)
            return false;
      }
   }

   if (u->output_chan < 0)
      return true;

   /*
    * A colour write overwrites r2 or r3: the varyings or texels the first
    * EXEC put there must have been read, and earlier writes of the channel
    * done.
    */
   int half = fp_output_half(u->output_chan);
   int reg = 2 + half / 2;
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *r = &c->units[i];
      if (!r->live || i == index)
         continue;
      if (i < index && r->output_chan == u->output_chan &&
          (r->packet < 0 || r->packet >= packet))
         return false;
      if (r->packet >= 0)
         continue;
      fp_unit_foreach_src(r, k, j) {
         struct fp_opnd *o = &r->slots[k].src[j];
         /* an fp20 varying uses the whole register, a texel one half */
         if (o->kind == FP_OPND_INPUT && fp_opnd_reg(c, o) == reg)
            return false;
         if ((o->kind == FP_OPND_TEXEL || o->kind == FP_OPND_DST) &&
             fp_opnd_reg(c, o) == reg &&
             fp_output_half(o->comp) % 2 == half % 2)
            return false;
      }
   }
   return true;
}

static bool
fp_any_ready(struct fp_compiler *c, int packet, int exec)
{
   for (int i = 0; i < c->num_units; i++)
      if (c->units[i].live && c->units[i].packet < 0 &&
          fp_unit_ready(c, i, packet, exec))
         return true;
   return false;
}

/* Distinct values a unit reads */
static int
fp_unit_values(struct fp_compiler *c, struct fp_unit *u, int *vals)
{
   int n = 0;

   fp_unit_foreach_src(u, k, j) {
      struct fp_opnd *o = &u->slots[k].src[j];
      if (o->kind != FP_OPND_VALUE)
         continue;
      int i;
      for (i = 0; i < n && vals[i] != o->index; i++)
         ;
      if (i == n)
         vals[n++] = o->index;
   }
   return n;
}

/* Change of the number of open values (scheduled, with readers left) */
static int
fp_unit_pressure(struct fp_compiler *c, struct fp_unit *u, const int *uses_left)
{
   int vals[12], n = fp_unit_values(c, u, vals);
   int delta = u->output_chan < 0 ? 1 : 0;

   for (int i = 0; i < n; i++)
      delta -= uses_left[vals[i]] == 1;
   return delta;
}

/*
 * List scheduling into packets, 3 packets per EXEC. Among the ready units
 * the ones closing values go first, then program order; at most "limit"
 * values are open at a time, which bounds the registers needed.
 */
static bool
fp_schedule(struct fp_compiler *c, int limit)
{
   int uses_left[FP_MAX_UNITS] = { 0 };
   int remaining = 0, packet = 0, exec = 0, in_exec = 0, stalls = 0, open = 0;

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      u->packet = -1;
      if (!u->live)
         continue;
      remaining++;
      int vals[12], n = fp_unit_values(c, u, vals);
      for (int k = 0; k < n; k++)
         uses_left[vals[k]]++;
   }

   while (remaining) {
      unsigned mask = 0;
      bool limited = false;

      while (mask != 0xf) {
         int best = -1, best_delta = 0;

         for (int i = 0; i < c->num_units; i++) {
            struct fp_unit *u = &c->units[i];
            if (!u->live || u->packet >= 0)
               continue;
            if (u->num_slots > 1 ? (mask & ((1u << u->num_slots) - 1)) != 0 :
                                   mask == 0xf)
               continue;
            if (!fp_unit_ready(c, i, packet, exec))
               continue;
            int delta = fp_unit_pressure(c, u, uses_left);
            if (delta > 0 && open + delta > limit) {
               limited = true;
               continue;
            }
            if (best < 0 || delta < best_delta) {
               best = i;
               best_delta = delta;
            }
         }
         if (best < 0)
            break;

         struct fp_unit *u = &c->units[best];
         int vals[12], n = fp_unit_values(c, u, vals);
         u->packet = packet;
         u->slot = u->num_slots > 1 ? 0 : ffs(~mask & 0xf) - 1;
         mask |= ((1u << u->num_slots) - 1) << u->slot;
         open += best_delta;
         for (int k = 0; k < n; k++)
            uses_left[vals[k]]--;
         remaining--;
      }

      if (!mask) {
         if (limited)
            return false;
         /* waiting for a varying of a later EXEC: start it */
         if (exec + 1 < FP_MAX_EXECS && fp_any_ready(c, packet, exec + 1)) {
            if (!in_exec)
               c->packet_exec[packet++] = exec;
            exec++;
            in_exec = 0;
            continue;
         }
         /* otherwise a latency stall: empty packet */
         if (++stalls > 16)
            return false;
      }

      if (packet == FP_MAX_PACKETS)
         return false;
      c->packet_exec[packet++] = exec;
      if (++in_exec == FP_MAX_PACKETS_PER_EXEC && remaining) {
         exec++;
         in_exec = 0;
      }
      if (exec == FP_MAX_EXECS)
         return false;
   }

   if (!packet)   /* no ALU work at all: one empty packet */
      c->packet_exec[packet++] = 0;
   c->num_packets = packet;
   c->num_execs = c->packet_exec[packet - 1] + 1;
   return true;
}

/* Pad the program with one-packet EXECs up to "rows" */
static void
fp_pad_execs(struct fp_compiler *c, int rows)
{
   while (c->num_execs < rows && c->num_packets < FP_MAX_PACKETS)
      c->packet_exec[c->num_packets++] = c->num_execs++;
}

/*
 * Register allocation over the schedule. A value needs a register when it
 * is read 2 or more packets after it was produced (or cannot be forwarded);
 * r2/r3 are the colour, r(4k+c) hold varyings until their last read, only
 * r0..r(4 * rows - 1) exist.
 */
static bool
fp_alloc_regs(struct fp_compiler *c)
{
   int last_read[FP_MAX_UNITS];
   int busy_until[16];
   int rows = c->num_execs;

   for (int r = 0; r < 16; r++)
      busy_until[r] = r < 4 * rows && r != 2 && r != 3 ? -1 : INT_MAX;

   for (int i = 0; i < c->num_units; i++) {
      last_read[i] = -1;
      c->units[i].reg = -1;
   }

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE) {
            struct fp_unit *d = &c->units[o->index];
            if (u->packet >= d->packet + 2 || !d->forward)
               last_read[o->index] = MAX2(last_read[o->index], u->packet);
         } else if (o->kind == FP_OPND_INPUT) {
            int r = fp_opnd_reg(c, o);
            if (r != 2 && r != 3)
               busy_until[r] = MAX2(busy_until[r], u->packet);
         } else if (o->kind == FP_OPND_TEXEL || o->kind == FP_OPND_DST) {
            int r = fp_opnd_reg(c, o);
            if (r != 2 && r != 3)
               busy_until[r] = MAX2(busy_until[r], u->packet);
         }
      }
   }

   /* the TEX of EXEC e reads and writes row e during that EXEC */
   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live)
         continue;
      int e = c->tex[n].exec, last = 0;
      for (int p = 0; p < c->num_packets; p++)
         if (c->packet_exec[p] == e)
            last = p;
      for (int r = 4 * e; r < 4 * e + 4; r++)
         if (r != 2 && r != 3)
            busy_until[r] = MAX2(busy_until[r], last);
   }

   for (int p = 0; p < c->num_packets; p++) {
      for (int i = 0; i < c->num_units; i++) {
         struct fp_unit *u = &c->units[i];
         if (!u->live || u->packet != p || last_read[i] < 0)
            continue;
         int r;
         for (r = 0; r < 16; r++)
            if (busy_until[r] <= p)
               break;
         if (r == 16)
            return false;
         u->reg = r;
         busy_until[r] = last_read[i];
      }
   }
   return true;
}

static struct fp_alu_src_operand
fp_resolve(struct fp_compiler *c, const struct fp_opnd *o, int packet)
{
   struct fp_alu_src_operand ret = { 0 };

   switch (o->kind) {
   case FP_OPND_ZERO:
      ret = fp_alu_src_zero();
      break;
   case FP_OPND_ONE:
      ret = fp_alu_src_one();
      break;
   case FP_OPND_UNIFORM:
      ret = fp_alu_src_uniform(o->index);
      break;
   case FP_OPND_INPUT:
      ret = fp_alu_src_row(fp_opnd_reg(c, o));
      break;
   case FP_OPND_TEXEL:
   case FP_OPND_DST:
      ret = fp_alu_src_row(fp_opnd_reg(c, o));
      ret.datatype = FP_DATATYPE_FIXED10;
      ret.sub_reg_select_high = fp_output_half(o->comp) % 2;
      break;
   case FP_OPND_VALUE: {
      struct fp_unit *d = &c->units[o->index];
      if (d->forward && d->packet == packet - 1)
         ret.index = FP_ALU_RESULT_REG(d->slot);
      else {
         assert(d->reg >= 0);
         ret.index = d->reg;
      }
      break;
   }
   }

   ret.negate = o->negate;
   ret.absolute_value = o->absolute;
   return ret;
}

static void
fp_emit_unit(struct fp_compiler *c, struct fp_unit *u,
             struct fp_alu_instr_packet *pkt)
{
   for (int k = 0; k < u->num_slots; k++) {
      struct fp_unit_slot *s = &u->slots[k];
      struct fp_alu_instr *a = &pkt->slots[u->slot + k];
      struct fp_alu_src_operand c2 = fp_resolve(c, &s->src[2], u->packet);

      /* rD is left disabled: result = op(rA * rB, rC) */
      *a = (struct fp_alu_instr) {
         .op = s->op,
         .accumulate_this = s->accumulate_this,
         .accumulate_other = s->accumulate_other,
         .src = {
            fp_resolve(c, &s->src[0], u->packet),
            fp_resolve(c, &s->src[1], u->packet),
            c2, c2
         },
      };

      if (k > 0) {
         a->dst.index = 31;     /* DP partial product, added to slot 0 */
         continue;
      }

      a->dst.saturate = u->saturate;
      if (u->output_chan >= 0) {
         int o = fp_output_half(u->output_chan);
         a->dst.index = 2 + o / 2;
         a->dst.write_low_sub_reg = (o % 2) == 0;
         a->dst.write_high_sub_reg = (o % 2) != 0;
      } else {
         /* fp20 value: both halves; only forwarded ones go to r31 */
         a->dst.index = u->reg >= 0 ? u->reg : 31;
         a->dst.write_low_sub_reg = true;
         a->dst.write_high_sub_reg = true;
      }
   }
}

static void
fp_emit(struct fp_compiler *c)
{
   struct grate_fp_shader *fp = c->fp;
   int packet = 0;

   for (int e = 0; e < c->num_execs; e++) {
      struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
      list_inithead(&inst->link);

      bool fetch = false;
      for (int k = 0; k < 4; k++)
         fetch |= c->exec_input[e][k] >= 0;
      if (fetch) {
         struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
         list_inithead(&mfu->link);
         for (int k = 0; k < 4; k++) {
            if (c->exec_input[e][k] < 0)
               continue;
            mfu->var[k].op = FP_VAR_OP_FP20;
            mfu->var[k].tram_row = c->exec_input[e][k];
         }
         if (e == 0) {
            /* barycentric interpolation setup */
            mfu->sfu.op = FP_SFU_OP_RCP;
            mfu->sfu.reg = 4;
            mfu->mul[0].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
            mfu->mul[0].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
            mfu->mul[0].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_0;
            mfu->mul[1].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
            mfu->mul[1].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
            mfu->mul[1].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_1;
         }
         inst->mfu_sched.num_instructions = 1;
         inst->mfu_sched.address = list_length(&fp->mfu_instructions);
         list_addtail(&mfu->link, &fp->mfu_instructions);
      }

      /* PSEQ load of the render target: OP LD (bit 23), surface (16..19),
       * ARGS bit 3, bit 1 = into r2/r3 (measured, gpu/tools/asmtri) */
      if (c->dst_used && c->dst_exec == e)
         inst->pseq = 1u << 23 | (c->dw.index & 0xf) << 16 | 1 << 3 |
                      c->dst_pair << 1;

      /* TEX word: sampler, coordinates r2/r3 (bit 4), texel to r2/r3 (bit 5),
       * enable (bit 10) */
      for (int n = 0; n < c->num_tex; n++) {
         if (!c->tex[n].live || c->tex[n].exec != e)
            continue;
         inst->tex = (c->tex[n].sampler & 0xf) | c->tex[n].pair << 4 |
                     (1 - c->tex[n].pair) << 5 | 1 << 10;
         fp->info.sampler_mask |= 1u << c->tex[n].sampler;
      }

      inst->alu_sched.address = list_length(&fp->alu_instructions);
      for (; packet < c->num_packets && c->packet_exec[packet] == e; packet++) {
         struct fp_alu_instr_packet *pkt = CALLOC_STRUCT(fp_alu_instr_packet);
         list_inithead(&pkt->link);
         for (int k = 0; k < 4; k++)
            pkt->slots[k] = fp_alu_nop();
         for (int i = 0; i < c->num_units; i++)
            if (c->units[i].live && c->units[i].packet == packet)
               fp_emit_unit(c, &c->units[i], pkt);
         list_addtail(&pkt->link, &fp->alu_instructions);
         inst->alu_sched.num_instructions++;
      }

      /* the store of the first EXEC runs after the last one */
      if (e == 0)
         inst->dw = c->dw;

      list_addtail(&inst->link, &fp->fp_instructions);
   }
}

static void
fp_dump(struct fp_compiler *c)
{
   fprintf(stderr, "GRATE FRAG: %d EXECs, %d packets:", c->num_execs,
           c->num_packets);
   for (int p = 0; p < c->num_packets; p++) {
      fprintf(stderr, " [e%d", c->packet_exec[p]);
      for (int i = 0; i < c->num_units; i++) {
         struct fp_unit *u = &c->units[i];
         if (u->live && u->packet == p)
            fprintf(stderr, " u%d%s", i, u->output_chan >= 0 ? "o" :
                    u->reg >= 0 ? "r" : "");
      }
      fprintf(stderr, "]");
   }
   fprintf(stderr, "\n");
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

   uint32_t dst = 0;
   for (int i = 0; i < 4; ++i)
      dst |= LINK_DST(i, i, LINK_DST_FP20);

   /* the source VS export is chosen at draw time by semantic: it used to be
    * export 1 for every input, so a second varying read the first one */
   fp->info.inputs[fp->info.num_inputs].src = LINK_SRC(1);
   fp->info.inputs[fp->info.num_inputs].dst = dst;
   fp->info.max_tram_row = MAX2(fp->info.max_tram_row, (int)decl->Range.First + 1);
   fp->info.inputs[fp->info.num_inputs].tram_row = decl->Range.First;
   fp->info.inputs[fp->info.num_inputs].semantic =
      GRATE_SEMANTIC(decl->Semantic.Name, decl->Semantic.Index);

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
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi,
                 const struct grate_fp_key *key)
{
   struct fp_compiler *c = CALLOC_STRUCT(fp_compiler);

   c->key = key;
   c->redirect_output = key && (key->blend || (key->colormask & 0xf) != 0xf);

   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);

   fp->info.num_inputs = 0;
   fp->info.color_input = -1;
   fp->info.max_tram_row = 1;
   fp->info.used_mask = 0;
   fp->info.imm_mask = 0;
   fp->info.sampler_mask = 0;
   fp->num_immediates = 0;

   c->fp = fp;

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
         if (tgsi->FullToken.FullInstruction.Instruction.Opcode == TGSI_OPCODE_END)
            break;
         emit_tgsi_instr(c, &tgsi->FullToken.FullInstruction);
         break;
      }
   }

   if (c->redirect_output && !c->failed)
      emit_blend(c);

   fp_dead_code(c);
   fp_assign_inputs(c);

   /*
    * Schedule for parallelism first, then with fewer values open at a time
    * until the registers suffice. More rows give more registers;
    * GRATE_FP_ROWS forces a minimum.
    */
   int min_rows = 1;
   if (getenv("GRATE_FP_ROWS"))
      min_rows = CLAMP(atoi(getenv("GRATE_FP_ROWS")), 1, FP_MAX_EXECS);
   bool done = c->failed;
   for (int limit = 16; limit > 0 && !done; limit--) {
      if (!fp_schedule(c, limit))
         continue;
      fp_pad_execs(c, min_rows);
      while (!(done = fp_alloc_regs(c)) && c->num_execs < FP_MAX_EXECS)
         fp_pad_execs(c, c->num_execs + 1);
   }
   if (!done)
      fp_fail(c, "shader too big (out of ALU packets or registers)");

   if (grate_debug & GRATE_DEBUG_TGSI)
      fp_dump(c);

   if (c->failed) {
      /* still emit something sane: magenta, one packet */
      memset(c->exec_input, -1, sizeof(c->exec_input));
      c->num_units = 0;
      c->num_tex = 0;
      c->dst_used = false;
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

   fp_emit(c);
   fp->info.num_rows = c->num_execs;
   FREE(c);
}
