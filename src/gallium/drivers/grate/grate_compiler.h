#ifndef GRATE_COMPILER_H
#define GRATE_COMPILER_H

#include "util/list.h"

#include <stdint.h>

struct tgsi_parse_context;

/* VPE constant memory uploaded by the driver: c[0..255] (as the blob does).
 * TGSI immediates are allocated from the top of it. */
#define GRATE_VP_NUM_CONSTS 256
#define GRATE_VP_MAX_IMMEDIATES 32

/* TGSI semantic of a shader input/output, for linking VS outputs to FS
 * inputs: name << 8 | index */
#define GRATE_SEMANTIC(name, index) ((name) << 8 | (index))

struct grate_vp_shader {
   struct list_head instructions;
   uint16_t output_mask;
   uint16_t output_semantic[16];
   float immediates[GRATE_VP_MAX_IMMEDIATES][4];
   int num_immediates;
};

/* ALU uniform registers 32..63, loaded from TGR3D_ALU_GLOBALS(0..31) */
#define GRATE_FP_NUM_CONSTS 32
#define GRATE_FP_MAX_IMMEDIATES 32

/*
 * Preshader (fp/preshader.c): values that depend on uniforms only, computed
 * on the CPU at draw time instead of for every pixel. Entry i sets uniform
 * slot dst to op(a[0] * b[0] + ... + a[n-1] * b[n-1], c), as the ALU would.
 */
#define GRATE_FP_MAX_PRE 24

enum grate_fp_pre_kind {
   GRATE_FP_PRE_USER,      /* user uniform slot "index" */
   GRATE_FP_PRE_RESULT,    /* result of preshader entry "index" */
   GRATE_FP_PRE_CONST,     /* "value" */
};

struct grate_fp_pre_opnd {
   uint8_t kind;
   uint8_t index;
   bool negate;
   bool absolute;
   float value;
};

struct grate_fp_pre {
   uint8_t op;             /* enum fp_alu_op */
   uint8_t dst;
   uint8_t num_products;
   bool saturate;
   struct grate_fp_pre_opnd a[4], b[4], c;
};

/* linker (SU) instructions: up to 32 */
#define GRATE_FP_MAX_LINKS 32

/*
 * Linker instruction: copies components of one VS export (found at draw time
 * by semantic) to columns of a TRAM row, which the fragment MFU interpolates
 * into registers: column k into r(k) (of its row of the pixel packet). The
 * compiler chooses the TRAM layout (fp/linker.c).
 */
struct grate_fp_link {
   uint32_t dst;           /* per export component: TRAM column and type */
   uint16_t semantic;
   uint8_t tram_row;
   uint8_t input;          /* TGSI input */
};

struct grate_fp_info {
   struct grate_fp_link links[GRATE_FP_MAX_LINKS];
   int num_links;
   int color_input;
   int max_tram_row;       /* TRAM rows used */
   int num_rows;            /* pixel packet rows (r0-r3, r4-r7, ...) */
   /* ALU uniform registers: user uniforms at index * 4 + component from the
    * bottom, immediates (fp20, already converted) allocated from the top */
   uint32_t consts[GRATE_FP_NUM_CONSTS];
   uint32_t used_mask;      /* slots read by the shader */
   uint32_t imm_mask;       /* slots holding immediates */
   uint32_t sampler_mask;   /* samplers read */
   struct grate_fp_pre pre[GRATE_FP_MAX_PRE];
   int num_pre;
};

uint32_t
grate_float_to_fp20(float f);

/* Fill in the preshader slots of consts[] (user: the user uniforms) */
void
grate_fp_eval_pre(const struct grate_fp_info *info, const float *user,
                  unsigned user_len, uint32_t *consts);

struct grate_fp_shader {
   struct list_head fp_instructions;
   struct list_head alu_instructions;
   struct list_head mfu_instructions;
   struct grate_fp_info info;
   float immediates[GRATE_FP_MAX_IMMEDIATES][4];
   int num_immediates;
};

void
grate_tgsi_to_vp(struct grate_vp_shader *vp, struct tgsi_parse_context *tgsi);

/*
 * Fragment program variant: blending and the colour write mask are done by
 * the program (PSEQ loads the destination). Zeroed + colormask 0xf = none.
 */
struct grate_fp_key {
   uint8_t blend;
   uint8_t rgb_func, rgb_src, rgb_dst;       /* PIPE_BLEND_*, PIPE_BLENDFACTOR_* */
   uint8_t alpha_func, alpha_src, alpha_dst;
   uint8_t colormask;                        /* PIPE_MASK_R.. bits */
};

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi,
                 const struct grate_fp_key *key);

#endif
