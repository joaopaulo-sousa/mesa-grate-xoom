#include <assert.h>
#include "grate_compiler.h"
#include "unistd.h"
#include "vpir.h"

#include "tgsi/tgsi_info.h"
#include "tgsi/tgsi_parse.h"

#include "util/u_memory.h"

static struct vp_src_operand
src_undef()
{
   struct vp_src_operand ret = {
      .file = VP_SRC_FILE_UNDEF,
      .index = 0,
      .swizzle = { VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z, VP_SWZ_W }
   };
   return ret;
}

static struct vp_src_operand
attrib(int index, const enum vp_swz swizzle[4], bool negate, bool absolute)
{
   struct vp_src_operand ret = {
      .file = VP_SRC_FILE_ATTRIB,
      .index = index,
      .negate = negate,
      .absolute = absolute
   };
   memcpy(ret.swizzle, swizzle, sizeof(ret.swizzle));
   return ret;
}

static struct vp_src_operand
uniform(int index, const enum vp_swz swizzle[4], bool negate, bool absolute)
{
   struct vp_src_operand ret = {
      .file = VP_SRC_FILE_UNIFORM,
      .index = index,
      .negate = negate,
      .absolute = absolute
   };
   memcpy(ret.swizzle, swizzle, sizeof(ret.swizzle));
   return ret;
}

static struct vp_src_operand
src_temp(int index, const enum vp_swz swizzle[4], bool negate, bool absolute)
{
   struct vp_src_operand ret = {
      .file = VP_SRC_FILE_TEMP,
      .index = index,
      .negate = negate,
      .absolute = absolute
   };
   memcpy(ret.swizzle, swizzle, sizeof(ret.swizzle));
   return ret;
}

static struct vp_dst_operand
dst_undef()
{
   struct vp_dst_operand ret = {
      .file = VP_DST_FILE_UNDEF,
      .index = 0,
      .write_mask = 0,
      .saturate = 0
   };
   return ret;
}

static struct vp_dst_operand
emit_output(struct grate_vp_shader *vp, int index,
            unsigned int write_mask, bool saturate)
{
   vp->output_mask |= 1 << index;
   struct vp_dst_operand ret = {
      .file = VP_DST_FILE_OUTPUT,
      .index = index,
      .write_mask = write_mask,
      .saturate = saturate
   };
   return ret;
}

static struct vp_dst_operand
dst_temp(int index, unsigned int write_mask, bool saturate)
{
   struct vp_dst_operand ret = {
      .file = VP_DST_FILE_TEMP,
      .index = index,
      .write_mask = write_mask,
      .saturate = saturate
   };
   return ret;
}

static struct vp_vec_instr
emit_vec_binop(enum vp_vec_op op, struct vp_dst_operand dst,
              struct vp_src_operand src0, struct vp_src_operand src1)
{
   struct vp_vec_instr ret = {
      .op = op,
      .dst = dst,
      .src = { src0, src1, src_undef() }
   };
   return ret;
}

static struct vp_vec_instr
emit_vNOP()
{
   struct vp_vec_instr ret = {
      .op = VP_VEC_OP_NOP,
      .dst = dst_undef(),
      .src = { src_undef(), src_undef(), src_undef() }
   };
   return ret;
}

static struct vp_vec_instr
emit_vADD(struct vp_dst_operand dst, struct vp_src_operand src0,
          struct vp_src_operand src2)
{
   struct vp_vec_instr ret = {
      .op = VP_VEC_OP_ADD,
      .dst = dst,
      .src = { src0, src_undef(), src2 } // add is "strange" in that it takes src0 and src2
   };
   return ret;
}

#define GEN_V_BINOP(OP) \
static struct vp_vec_instr \
emit_v ## OP (struct vp_dst_operand dst, struct vp_src_operand src0, \
          struct vp_src_operand src1) \
{ \
   return emit_vec_binop(VP_VEC_OP_ ## OP, dst, src0, src1); \
}

GEN_V_BINOP(MUL)

static struct vp_scalar_instr
emit_sNOP()
{
   struct vp_scalar_instr ret = {
      .op = VP_SCALAR_OP_NOP,
      .dst = dst_undef(),
      .src = src_undef()
   };
   return ret;
}

/* the last two of the 32 temporaries hold intermediate results of the
 * TGSI opcodes that take several instructions, the two below them sources
 * moved out of the attribute and constant files (vp_fetch_once) */
#define VP_SCRATCH0 31
#define VP_SCRATCH1 30
#define VP_FETCH0 29
#define VP_MAX_TGSI_TEMPS 28

static const enum vp_swz vp_swz_xyzw[4] = { VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z, VP_SWZ_W };

/* component swz of a scratch register, replicated */
static struct vp_src_operand
src_scratch(int index, enum vp_swz swz)
{
   enum vp_swz swizzle[4] = { swz, swz, swz, swz };
   return src_temp(index, swizzle, false, false);
}

static struct vp_src_operand
src_negated(struct vp_src_operand src)
{
   src.negate = !src.negate;
   return src;
}

static void
emit_packed(struct grate_vp_shader *vp, struct vp_vec_instr vec,
            struct vp_scalar_instr scalar)
{
   struct vp_instr *ret = CALLOC_STRUCT(vp_instr);
   list_inithead(&ret->link);
   ret->vec = vec;
   ret->scalar = scalar;
   list_addtail(&ret->link, &vp->instructions);
}

static struct vp_dst_operand
tgsi_dst_to_vp(struct grate_vp_shader *vp, const struct tgsi_dst_register *dst, bool saturate)
{
   switch (dst->File) {
   case TGSI_FILE_OUTPUT:
      return emit_output(vp, dst->Index, dst->WriteMask, saturate);

   case TGSI_FILE_TEMPORARY:
      assert(dst->Index < VP_MAX_TGSI_TEMPS);
      return dst_temp(dst->Index, dst->WriteMask, saturate);

   default:
      UNREACHABLE("unsupported output");
   }
}

static struct vp_src_operand
tgsi_src_to_vp(struct grate_vp_shader *vp, const struct tgsi_src_register *src)
{
   enum vp_swz swizzle[4] = {
      src->SwizzleX,
      src->SwizzleY,
      src->SwizzleZ,
      src->SwizzleW
   };
   bool negate = src->Negate != 0;
   bool absolute = src->Absolute != 0;

   switch (src->File) {
   case TGSI_FILE_INPUT:
      return attrib(src->Index, swizzle, negate, absolute);

   case TGSI_FILE_CONSTANT:
      return uniform(src->Index, swizzle, negate, absolute);

   case TGSI_FILE_TEMPORARY:
      return src_temp(src->Index, swizzle, negate, absolute);

   case TGSI_FILE_IMMEDIATE:
      /* immediates live at the top of the constant memory, see emit_vs_uniforms() */
      return uniform(GRATE_VP_NUM_CONSTS - 1 - src->Index, swizzle, negate, absolute);

   default:
      UNREACHABLE("unsupported input!");
   }
}

/*
 * TGSI opcodes the vector or scalar unit has one instruction for; the
 * scalar unit works on the first component of its source
 */
static const struct {
   bool scalar;
   int op, num_src;
} vp_direct_ops[TGSI_OPCODE_LAST] = {
   [TGSI_OPCODE_MOV] = { false, VP_VEC_OP_MOV, 1 },
   [TGSI_OPCODE_MUL] = { false, VP_VEC_OP_MUL, 2 },
   [TGSI_OPCODE_MAD] = { false, VP_VEC_OP_MAD, 3 },
   [TGSI_OPCODE_DP3] = { false, VP_VEC_OP_DP3, 2 },
   [TGSI_OPCODE_DP4] = { false, VP_VEC_OP_DP4, 2 },
   [TGSI_OPCODE_MIN] = { false, VP_VEC_OP_MIN, 2 },
   [TGSI_OPCODE_MAX] = { false, VP_VEC_OP_MAX, 2 },
   [TGSI_OPCODE_SLT] = { false, VP_VEC_OP_SLT, 2 },
   [TGSI_OPCODE_SGE] = { false, VP_VEC_OP_SGE, 2 },
   [TGSI_OPCODE_SEQ] = { false, VP_VEC_OP_SEQ, 2 },
   [TGSI_OPCODE_SNE] = { false, VP_VEC_OP_SNE, 2 },
   [TGSI_OPCODE_SGT] = { false, VP_VEC_OP_SGT, 2 },
   [TGSI_OPCODE_SLE] = { false, VP_VEC_OP_SLE, 2 },
   [TGSI_OPCODE_FRC] = { false, VP_VEC_OP_FRC, 1 },
   [TGSI_OPCODE_FLR] = { false, VP_VEC_OP_FLR, 1 },
   [TGSI_OPCODE_SSG] = { false, VP_VEC_OP_SSG, 1 },
   [TGSI_OPCODE_RCP] = { true, VP_SCALAR_OP_RCP, 1 },
   [TGSI_OPCODE_RSQ] = { true, VP_SCALAR_OP_RSQ, 1 },
   [TGSI_OPCODE_EX2] = { true, VP_SCALAR_OP_EX2, 1 },
   [TGSI_OPCODE_LG2] = { true, VP_SCALAR_OP_LG2, 1 },
   [TGSI_OPCODE_SIN] = { true, VP_SCALAR_OP_SIN, 1 },
   [TGSI_OPCODE_COS] = { true, VP_SCALAR_OP_COS, 1 },
};

static struct vp_vec_instr
emit_vec(enum vp_vec_op op, struct vp_dst_operand dst, struct vp_src_operand a,
         struct vp_src_operand b, struct vp_src_operand c)
{
   struct vp_vec_instr ret = { .op = op, .dst = dst, .src = { a, b, c } };
   return ret;
}

static struct vp_scalar_instr
emit_scalar(enum vp_scalar_op op, struct vp_dst_operand dst, struct vp_src_operand src)
{
   struct vp_scalar_instr ret = { .op = op, .dst = dst, .src = src };
   return ret;
}

/*
 * d = a < 0 ? b : c by the condition register, so that the value not taken
 * never enters the result (an arithmetic select turned b * 0 into NaN when
 * b was infinite: Qt's antialiasing shader computes normalize(0) in the
 * branch it does not take). The condition is set by moving a, then c is
 * written and b over it where a < 0; through a scratch register when d is
 * b's register, which the second move would overwrite.
 */
static void
emit_cmp(struct grate_vp_shader *vp, struct vp_dst_operand dst,
         struct vp_src_operand a, struct vp_src_operand b, struct vp_src_operand c)
{
   struct vp_vec_instr set = emit_vec(VP_VEC_OP_MOV, dst_temp(VP_SCRATCH0, dst.write_mask, false),
                                      a, src_undef(), src_undef());
   set.cond_set = true;
   emit_packed(vp, set, emit_sNOP());

   bool via_scratch = dst.file != VP_DST_FILE_TEMP ||
                      (b.file == VP_SRC_FILE_TEMP && b.index == dst.index);
   struct vp_dst_operand d = via_scratch ? dst_temp(VP_SCRATCH1, dst.write_mask, false) : dst;
   d.saturate = false;
   emit_packed(vp, emit_vec(VP_VEC_OP_MOV, d, c, src_undef(), src_undef()), emit_sNOP());
   struct vp_vec_instr take_b = emit_vec(VP_VEC_OP_MOV, d, b, src_undef(), src_undef());
   take_b.cond_test = VP_COND_LT;
   emit_packed(vp, take_b, emit_sNOP());

   if (via_scratch || dst.saturate)
      emit_packed(vp, emit_vec(VP_VEC_OP_MOV, dst,
                               src_temp(d.index, vp_swz_xyzw, false, false),
                               src_undef(), src_undef()),
                  emit_sNOP());
}

/*
 * An instruction fetches one attribute and one constant (immediates are
 * constants too): sources reading another one are moved to a temporary
 * first, keeping their swizzle and modifiers.
 */
static void
vp_fetch_once(struct grate_vp_shader *vp, struct vp_src_operand *src, int num)
{
   int attrib = -1, constant = -1, temp = VP_FETCH0;

   for (int i = 0; i < num; i++) {
      int *fetch = src[i].file == VP_SRC_FILE_ATTRIB ? &attrib :
                   src[i].file == VP_SRC_FILE_UNIFORM ? &constant : NULL;
      if (!fetch)
         continue;
      if (*fetch < 0 || *fetch == src[i].index) {
         *fetch = src[i].index;
         continue;
      }
      emit_packed(vp, emit_vec(VP_VEC_OP_MOV, dst_temp(temp, 0xf, false), src[i],
                               src_undef(), src_undef()),
                  emit_sNOP());
      src[i] = src_temp(temp--, vp_swz_xyzw, false, false);
   }
}

static bool
tgsi_to_vp(struct grate_vp_shader *vp, const struct tgsi_full_instruction *inst)
{
   unsigned opcode = inst->Instruction.Opcode;
   bool saturate = inst->Instruction.Saturate != 0;
   struct vp_dst_operand dst = tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate);
   struct vp_src_operand src[3];

   for (unsigned i = 0; i < inst->Instruction.NumSrcRegs && i < 3; i++)
      src[i] = tgsi_src_to_vp(vp, &inst->Src[i].Register);
   vp_fetch_once(vp, src, MIN2(inst->Instruction.NumSrcRegs, 3));

   if (vp_direct_ops[opcode].num_src) {
      if (vp_direct_ops[opcode].scalar) {
         emit_packed(vp, emit_vNOP(), emit_scalar(vp_direct_ops[opcode].op, dst, src[0]));
      } else {
         struct vp_src_operand s[3] = { src_undef(), src_undef(), src_undef() };
         for (int i = 0; i < vp_direct_ops[opcode].num_src; i++)
            s[i] = src[i];
         emit_packed(vp, emit_vec(vp_direct_ops[opcode].op, dst, s[0], s[1], s[2]),
                     emit_sNOP());
      }
      return true;
   }

   struct vp_src_operand ts = src_temp(VP_SCRATCH0, vp_swz_xyzw, false, false);

   switch (opcode) {
   case TGSI_OPCODE_ADD:
      emit_packed(vp, emit_vADD(dst, src[0], src[1]), emit_sNOP());
      return true;

   case TGSI_OPCODE_DP2:
      /* t = a * b, d = t.x + t.y */
      emit_packed(vp, emit_vMUL(dst_temp(VP_SCRATCH0, 0x3, false), src[0], src[1]),
                  emit_sNOP());
      emit_packed(vp, emit_vADD(dst, src_scratch(VP_SCRATCH0, VP_SWZ_X),
                                src_scratch(VP_SCRATCH0, VP_SWZ_Y)),
                  emit_sNOP());
      return true;

   case TGSI_OPCODE_CMP:
      emit_cmp(vp, dst, src[0], src[1], src[2]);
      return true;

   case TGSI_OPCODE_LRP: {
      /* d = a * b + (c - a * c) */
      struct vp_dst_operand u = dst_temp(VP_SCRATCH0, dst.write_mask, false);
      emit_packed(vp, emit_vec(VP_VEC_OP_MAD, u, src_negated(src[0]), src[2], src[2]),
                  emit_sNOP());
      emit_packed(vp, emit_vec(VP_VEC_OP_MAD, dst, src[0], src[1], ts), emit_sNOP());
      return true;
   }

   case TGSI_OPCODE_POW: {
      /* d = 2^(b * log2(a)) */
      struct vp_dst_operand t1 = dst_temp(VP_SCRATCH0, 0x1, false);
      emit_packed(vp, emit_vNOP(), emit_scalar(VP_SCALAR_OP_LG2, t1, src[0]));
      emit_packed(vp, emit_vMUL(t1, src_scratch(VP_SCRATCH0, VP_SWZ_X), src[1]),
                  emit_sNOP());
      emit_packed(vp, emit_vNOP(),
                  emit_scalar(VP_SCALAR_OP_EX2, dst, src_scratch(VP_SCRATCH0, VP_SWZ_X)));
      return true;
   }

   default:
      fprintf(stderr, "GRATE VERTEX TGSI UNIMPLEMENTED: %s\n", tgsi_get_opcode_name(opcode));
      return false;
   }
}

void
grate_tgsi_to_vp(struct grate_vp_shader *vp, struct tgsi_parse_context *tgsi)
{
   list_inithead(&vp->instructions);
   vp->output_mask = 0;
   vp->num_immediates = 0;
   memset(vp->output_semantic, 0xff, sizeof(vp->output_semantic));

   while (!tgsi_parse_end_of_tokens(tgsi)) {
      tgsi_parse_token(tgsi);
      switch (tgsi->FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_DECLARATION: {
         const struct tgsi_full_declaration *decl = &tgsi->FullToken.FullDeclaration;
         if (decl->Declaration.File != TGSI_FILE_OUTPUT)
            break;
         for (unsigned i = decl->Range.First; i <= decl->Range.Last && i < 16; i++)
            vp->output_semantic[i] =
               GRATE_SEMANTIC(decl->Semantic.Name,
                              decl->Semantic.Index + i - decl->Range.First);
         break;
      }
      case TGSI_TOKEN_TYPE_IMMEDIATE: {
         const struct tgsi_full_immediate *imm = &tgsi->FullToken.FullImmediate;
         unsigned n = MIN2(imm->Immediate.NrTokens - 1, 4);
         assert(vp->num_immediates < GRATE_VP_MAX_IMMEDIATES);
         for (unsigned i = 0; i < 4; i++)
            vp->immediates[vp->num_immediates][i] = i < n ? imm->u[i].Float : 0.0f;
         vp->num_immediates++;
         break;
      }
      case TGSI_TOKEN_TYPE_INSTRUCTION:
         if (tgsi->FullToken.FullInstruction.Instruction.Opcode != TGSI_OPCODE_END) {
            if (!tgsi_to_vp(vp, &tgsi->FullToken.FullInstruction))
               assert(!"unsupported vertex opcode");
         }
         break;
      }
   }
}
