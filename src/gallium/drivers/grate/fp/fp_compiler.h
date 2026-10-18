/*
 * Internal interface of the grate fragment compiler: the TGSI frontend
 * (tgsi.c) lowers a shader to scalar units, then one of two schedulers lays
 * them out for the hardware:
 *
 * - sched_rows.c: multi-row programs (up to 4 EXECs run as pixel packet
 *   rows, 16 registers, varying texture coordinates only);
 * - sched_seq.c: one-row sequential programs (EXECs are passes over r0..r3),
 *   needed for the special functions (SFU) and dependent texture reads.
 *
 * The hardware rules both follow were measured on Tegra20, see
 * gpu/tools/asmtri/README.md.
 */
#ifndef FP_COMPILER_H
#define FP_COMPILER_H

#include <stdbool.h>
#include <stdint.h>

#include "grate_compiler.h"
#include "fpir.h"

#define FP_MAX_UNITS 128
#define FP_MAX_TEMPS 32
#define FP_MAX_INPUTS 16
#define FP_MAX_EXECS 4           /* multi-row programs */
#define FP_SEQ_MAX_EXECS 16      /* one-row sequential programs */
#define FP_MAX_PACKETS_PER_EXEC 3
#define FP_MAX_PACKETS 64
#define FP_MAX_TEXTURES 4

#define FP_ALU_RESULT_REG(s) (24 + (s))
#define FP_KILL_REG 76           /* non-zero written discards the fragment */

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
   int num_slots;          /* 1, or a DPn taking slots 0..n-1; 0 for an SFU unit */
   struct fp_unit_slot slots[4];
   enum fp_sfu_op sfu;     /* SFU unit: op on slots[0].src[0], in place */
   bool sfu_copy;          /* MOV giving an SFU operand a register of its own */
   bool kill;              /* writes the kill register: no value, always live */
   bool saturate;
   int output_chan;        /* colour channel written, or -1: defines a value */
   bool forward;           /* the next packet may read the result as aluN */
   bool live;

   /* scheduling */
   int exec;
   int packet;
   int slot;
   int reg;                /* register holding the value, -1 if none */
};

/* What one EXEC of a sequential program does (sched_seq.c) */
struct fp_seq_exec {
   int sfu_unit;           /* SFU unit run by the MFU, -1 if none */
   int ipl_input[4];       /* TGSI input interpolated into r(c), -1 if none */
   int tex;                /* texture fetch, -1 if none */
   bool pseq;              /* destination colour load */
   int first_packet;
   int num_packets;
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

   /* texture fetches; multi-row programs need coord = TGSI input
    * components 2*pair, +1 */
   struct {
      struct fp_opnd coord[2];
      int input;
      int pair;            /* register pair of the coordinates */
      int sampler;
      int exec;
      bool live;
   } tex[FP_MAX_TEXTURES];
   int num_tex;

   /* SFU units or dependent texture reads: a sequential program */
   bool seq;
   struct fp_seq_exec seq_exec[FP_SEQ_MAX_EXECS];

   int packet_exec[FP_MAX_PACKETS];
   int num_packets;
   int num_execs;
};

#define fp_unit_foreach_src(u, k, j) \
   for (int k = 0; k < MAX2((u)->num_slots, 1); k++) \
      for (int j = 0; j < ((u)->num_slots ? 3 : 1); j++)

/* Whether a unit defines a value (not a colour write or a discard) */
static inline bool
fp_unit_has_value(const struct fp_unit *u)
{
   return u->output_chan < 0 && !u->kill;
}

/* fixed10 colour channel -> r2/r3 half, BGRA (also the texel and the
 * destination colour layout) */
static inline int
fp_output_half(int chan)
{
   return chan < 3 ? (2 - chan) : 3;
}

/* tgsi.c */
void fp_fail(struct fp_compiler *c, const char *msg);
struct fp_unit *fp_new_unit(struct fp_compiler *c);
struct fp_opnd fp_opnd_value(struct fp_compiler *c, struct fp_unit *u);
struct fp_opnd fp_opnd_zero(struct fp_compiler *c, bool output);
struct fp_opnd fp_opnd_one(struct fp_compiler *c, bool output);
void fp_unit_op(struct fp_unit_slot *s, enum fp_alu_op op, struct fp_opnd a,
                struct fp_opnd b, struct fp_opnd cc);
struct fp_alu_src_operand fp_alu_src_row(int index);
struct fp_alu_src_operand fp_alu_src_uniform(int index);
struct fp_alu_src_operand fp_alu_src_zero(void);
struct fp_alu_src_operand fp_alu_src_one(void);
struct fp_alu_instr fp_alu_nop(void);
void fp_mfu_bary_setup(struct fp_mfu_instr *mfu);
uint32_t fp_pseq_word(const struct fp_compiler *c, int pair);
uint32_t fp_tex_word(const struct fp_compiler *c, int n, int coord_pair);

/* preshader.c */
void fp_preshade(struct fp_compiler *c);

/* sched_rows.c */
bool fp_sched_rows(struct fp_compiler *c);
void fp_emit_rows(struct fp_compiler *c);
void fp_dump_rows(struct fp_compiler *c);

/* sched_seq.c */
bool fp_sched_seq(struct fp_compiler *c);
void fp_emit_seq(struct fp_compiler *c);

#endif
