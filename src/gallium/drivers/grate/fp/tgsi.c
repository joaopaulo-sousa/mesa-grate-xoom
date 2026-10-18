#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "grate_common.h"
#include "fp_compiler.h"

#include "tgsi/tgsi_parse.h"

#include "util/blend.h"
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
   return fp_opnd_uniform(fp_const_slot(c->fp, 0.0f));
}

struct fp_opnd
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
   if (c->num_tex == FP_MAX_TEXTURES) {
      fp_fail(c, "too many texture fetches");
      return;
   }

   /* coordinates other than a varying .xy/.zw (a dependent read) need a
    * sequential program: the ALU computes them in an earlier pass */
   bool varying = s.kind == FP_OPND_INPUT && t.kind == FP_OPND_INPUT &&
                  s.index == t.index && !s.negate && !s.absolute &&
                  !t.negate && !t.absolute && !(s.comp & 1) &&
                  t.comp == s.comp + 1;
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
   c->tex[n].coord[1] = t;
   c->tex[n].input = varying ? s.index : -1;
   c->tex[n].pair = varying ? s.comp / 2 : 0;
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
   bool scalar = opcode != TGSI_OPCODE_FRC && opcode != TGSI_OPCODE_FLR;

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

   case TGSI_OPCODE_RCP:
   case TGSI_OPCODE_RSQ:
   case TGSI_OPCODE_SQRT:
   case TGSI_OPCODE_EX2:
   case TGSI_OPCODE_LG2:
   case TGSI_OPCODE_SIN:
   case TGSI_OPCODE_COS:
   case TGSI_OPCODE_FRC:
   case TGSI_OPCODE_FLR:
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
   return (c->tex[n].sampler & 0xf) | coord_pair << 4 |
          (1 - coord_pair) << 5 | 1 << 10;
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
         /* (units are in dependency order: the operand first) */
         if (inv) {
            ts = fp_val(c, FP_ALU_OP_MAD, dd, fp_opnd_one(c, false), ts);
            dd = fp_opnd_modify(dd, true, false);
         }
         u = fp_new_unit(c);
         u->saturate = true;
         fp_unit_op(&u->slots[0], FP_ALU_OP_MAD, dd, y, ts);
      }
      u->output_chan = ch;
   }
}


static bool
fp_opnd_equal(const struct fp_opnd *a, const struct fp_opnd *b)
{
   return a->kind == b->kind && a->index == b->index && a->comp == b->comp &&
          a->negate == b->negate && a->absolute == b->absolute;
}

/*
 * Common subexpressions: a plain value unit identical to an earlier one
 * (TGSI repeats the same operation per channel for replicated scalars)
 * hands its readers to the earlier one. In unit order, so whole chains
 * collapse; fp_dead_code then drops the copies.
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
      if (!fp_unit_has_value(u) || u->num_slots != 1 || u->sfu || u->sfu_copy)
         continue;
      for (int p = 0; p < i; p++) {
         struct fp_unit *q = &c->units[p];
         if (remap[p] != p || q->output_chan >= 0 || q->num_slots != 1 ||
             q->sfu || q->sfu_copy || q->saturate != u->saturate ||
             q->slots[0].op != u->slots[0].op ||
             q->slots[0].accumulate_this || q->slots[0].accumulate_other ||
             u->slots[0].accumulate_this || u->slots[0].accumulate_other)
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
      for (int k = 0; k < 2; k++)
         if (c->tex[n].coord[k].kind == FP_OPND_VALUE)
            c->tex[n].coord[k].index = remap[c->tex[n].coord[k].index];
}

/* Drop units whose value is never used */
static void
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
            for (int k = 0; k < 2; k++)
               if (c->tex[o->index].coord[k].kind == FP_OPND_VALUE)
                  c->units[c->tex[o->index].coord[k].index].live = true;
         }
      }
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

   fp_cse(c);
   fp_dead_code(c);
   fp_fold_sfu_copies(c);

   if (!c->failed && !(c->seq ? fp_sched_seq(c) : fp_sched_rows(c)))
      fp_fail(c, "shader too big (out of ALU packets or registers)");

   if ((grate_debug & GRATE_DEBUG_TGSI) && !c->seq)
      fp_dump_rows(c);

   if (c->failed) {
      /* still emit something sane: magenta, one packet */
      memset(c->exec_input, -1, sizeof(c->exec_input));
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
   fp->info.num_rows = c->seq ? 1 : c->num_execs;
   FREE(c);
}
