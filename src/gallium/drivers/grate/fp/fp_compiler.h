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

#define FP_MAX_UNITS 512
#define FP_MAX_TEMPS 32
#define FP_MAX_INPUTS 16
#define FP_MAX_EXECS 4           /* multi-row programs */
#define FP_SEQ_MAX_EXECS 16      /* one-row sequential programs */
#ifndef FP_MAX_PACKETS_PER_EXEC   /* fphost-relaxed: how many a program needs */
#define FP_MAX_PACKETS_PER_EXEC 3
#endif
#define FP_MAX_PACKETS 64
#define FP_MAX_TEXTURES 4
#define FP_TRAM_ROWS 16

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
   FP_OPND_SPECIAL,       /* a register the hardware fills, index FP_REG_* */
};

/* Registers with the fragment position (fp20, 8192 + the pixel's integer
 * coordinate) and the polygon face (fixed10 low half: 1 front, 0 back) */
#define FP_REG_POSITION_X 72
#define FP_REG_POSITION_Y 73
#define FP_REG_FACE 75

struct fp_opnd {
   enum fp_opnd_kind kind;
   int index;      /* uniform slot, unit, TGSI input or texture fetch */
   int comp;       /* input component, texel channel */
   bool negate;
   bool absolute;
   bool twice;     /* operand modifier: scaled by two (fp_fold_scales) */
   bool minus_one; /* operand modifier: read less one */
};

struct fp_unit_slot {
   enum fp_alu_op op;
   struct fp_opnd src[3];
   bool accumulate_this;
   bool accumulate_other;
   bool square_c;    /* the added term is src[2] * src[2] (rD = rC) */
};

struct fp_unit {
   int num_slots;          /* 1, or a DPn taking slots 0..n-1; 0 for an SFU unit */
   struct fp_unit_slot slots[4];
   enum fp_sfu_op sfu;     /* SFU unit: op on slots[0].src[0], in place */
   bool sfu_copy;          /* MOV giving an SFU operand a register of its own */
   bool kill;              /* writes the kill register: no value, always live */
   bool saturate;
   enum fp_scale scale;    /* result scaled by 2, 4 or 1/2 before saturating */
   enum fp_condition condition;   /* result made 1.0 or 0.0 by testing it */
   bool blend_inv;         /* d * (1 - y) + ts of a blend, y read less one
                              (fp_expand_blend_inv) */
   int output_chan;        /* colour channel written, or -1: defines a value */
   bool forward;           /* the next packet may read the result as aluN */
   bool live;

   /* scheduling */
   int exec;
   int packet;
   int slot;
   int reg;                /* register holding the value, -1 if none */
   bool fx10;              /* may be kept as a fixed10 half (fx10.c) */
   int half;               /* ... and is, in this half of reg; -1: fp20 */
};

/* What one EXEC of a sequential program does (sched_seq.c) */
struct fp_seq_exec {
   int sfu_unit;           /* SFU unit run by the MFU, -1 if none */
   int ipl_input[4];       /* TGSI input interpolated into r(k), -1 if none */
   int ipl_comp[4];        /* ... its component */
   int tex;                /* texture fetch, -1 if none */
   bool pseq;              /* destination colour load */
   int first_packet;
   int num_packets;
};

/*
 * Control flow (fp_run): the fragment unit cannot jump, so both sides of
 * an IF run and loops are unrolled; a write merges with the old value by
 * the predicate of the pixels it is for (-1.0 for those, 0.0 for others).
 * A pixel leaving a loop keeps the values it has then (fp_flow.exit), so
 * writes in a loop need not test whether a pixel is still in it.
 */
enum fp_pred_kind {
   FP_PRED_ALWAYS,
   FP_PRED_NEVER,
   FP_PRED_VALUE,
};

struct fp_pred {
   enum fp_pred_kind kind;
   struct fp_opnd v;
};

#define FP_MAX_FLOW_DEPTH 16
#define FP_MAX_LOOP_ITERATIONS 32

struct fp_compiler {
   struct grate_fp_shader *fp;
   struct fp_unit units[FP_MAX_UNITS];
   int num_units;
   struct fp_opnd temps[FP_MAX_TEMPS][4];
   bool temp_written[FP_MAX_TEMPS][4];
   struct fp_dw_instr dw;
   bool failed;

   struct fp_pred pred;    /* pixels the current writes are for */
   struct fp_pred pred_full;  /* ... that have not left a loop either */
   struct fp_flow {
      bool loop;
      struct fp_pred cond;            /* IF: of the side being run */
      struct fp_pred alive, iter;     /* loop: not left by BRK, CONT */
      /* loop: values of the pixels that left it */
      struct fp_opnd exit[FP_MAX_TEMPS][4], exit_outv[4];
      bool exit_written[FP_MAX_TEMPS][4], exit_outv_written[4];
   } flow[FP_MAX_FLOW_DEPTH];
   int flow_depth;
   bool needs_flow;        /* a loop exits by uniforms: compile per value */
   int8_t eval[FP_MAX_UNITS];        /* fp_eval: 1 known, -1 not, 0 unseen */
   float eval_v[FP_MAX_UNITS];
   int8_t dep[FP_MAX_UNITS];         /* fp_deps: 1 uniforms only, -1 pixel */
   uint64_t dep_users[FP_MAX_UNITS];

   /* TGSI inputs: semantic; TRAM layout, input * 4 + component per row and
    * column (-1: free) */
   uint16_t input_semantic[FP_MAX_INPUTS];
   uint8_t input_interp[FP_MAX_INPUTS];    /* TGSI_INTERPOLATE_* */
   int num_inputs;                         /* declared, then moved ones */
   int tram[FP_TRAM_ROWS][4];

   /* EXEC whose MFU fetches TGSI input i component c, into which column
    * (register of the row; the component's own but for texture coordinates,
    * which must be a pair), and the reverse */
   int fetch_exec[FP_MAX_INPUTS][4];
   int fetch_col[FP_MAX_INPUTS][4];
   int exec_input[FP_MAX_EXECS][4];
   int exec_comp[FP_MAX_EXECS][4];
   int fetch_reg[FP_MAX_INPUTS][4];   /* sequential: register it goes to */

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
   int out_pair;           /* colour register pair: 1 = r2/r3, 0 = r0/r1 */
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
struct fp_opnd fp_opnd_const(struct fp_compiler *c, float value);
void fp_unit_op(struct fp_unit_slot *s, enum fp_alu_op op, struct fp_opnd a,
                struct fp_opnd b, struct fp_opnd cc);
struct fp_alu_src_operand fp_alu_src_row(int index);
struct fp_alu_src_operand fp_alu_src_uniform(int index);
struct fp_alu_src_operand fp_alu_src_zero(void);
struct fp_alu_src_operand fp_alu_src_one(void);
struct fp_alu_src_operand fp_alu_src_special(int index);
struct fp_alu_instr fp_alu_nop(void);
void fp_mfu_bary_setup(struct fp_mfu_instr *mfu);
uint32_t fp_pseq_word(const struct fp_compiler *c, int pair);
uint32_t fp_tex_word(const struct fp_compiler *c, int n, int coord_pair);

/* fx10.c */
void fp_mark_fx10(struct fp_compiler *c);
void fp_reassociate(struct fp_compiler *c);
void fp_fold_scales(struct fp_compiler *c);
void fp_expand_blend_inv(struct fp_compiler *c);
void fp_fold_conditions(struct fp_compiler *c);
void fp_move_coords(struct fp_compiler *c);
bool fp_opnd_is_const(struct fp_compiler *c, const struct fp_opnd *o, float v);

/* preshader.c */
float fp20_to_float(uint32_t v);
void fp_preshade(struct fp_compiler *c);
void fp_specialize(struct fp_compiler *c);

/* linker.c */
int fp_tram_row(struct fp_compiler *c, int input, int comp, int column);
void fp_build_links(struct fp_compiler *c);

/* sched_rows.c */
bool fp_sched_rows(struct fp_compiler *c);
void fp_emit_rows(struct fp_compiler *c);
void fp_dump_rows(struct fp_compiler *c);

/* sched_seq.c */
bool fp_sched_seq(struct fp_compiler *c);
void fp_emit_seq(struct fp_compiler *c);

#endif
