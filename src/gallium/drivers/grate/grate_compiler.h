#ifndef GRATE_COMPILER_H
#define GRATE_COMPILER_H

#include "util/list.h"

#include <stdint.h>

struct tgsi_parse_context;

/* VPE constant memory uploaded by the driver: c[0..255] (as the blob does).
 * TGSI immediates are allocated from the top of it. */
#define GRATE_VP_NUM_CONSTS 256
#define GRATE_VP_MAX_IMMEDIATES 32

struct grate_vp_shader {
   struct list_head instructions;
   uint16_t output_mask;
   float immediates[GRATE_VP_MAX_IMMEDIATES][4];
   int num_immediates;
};

/* ALU uniform registers 32..63, loaded from TGR3D_ALU_GLOBALS(0..31) */
#define GRATE_FP_NUM_CONSTS 32
#define GRATE_FP_MAX_IMMEDIATES 32

struct grate_fp_info {
   struct {
      uint32_t src;
      uint32_t dst;
   } inputs[16];
   int num_inputs;
   int color_input;
   int max_tram_row;
   int num_rows;            /* pixel packet rows (r0-r3, r4-r7, ...) */
   /* ALU uniform registers: user uniforms at index * 4 + component from the
    * bottom, immediates (fp20, already converted) allocated from the top */
   uint32_t consts[GRATE_FP_NUM_CONSTS];
   uint32_t used_mask;      /* slots read by the shader */
   uint32_t imm_mask;       /* slots holding immediates */
};

uint32_t
grate_float_to_fp20(float f);

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

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi);

#endif
