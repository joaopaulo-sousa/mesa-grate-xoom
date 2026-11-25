/*
 * One-row sequential fragment programs (measured on Tegra20, see
 * gpu/tools/asmtri/README.md, "Special functions"):
 *
 * - The pixel packet has one row: r0..r3. The EXECs are passes in order;
 *   whatever an EXEC writes is there for the next one.
 * - An EXEC is: PSEQ (destination colour load into a register pair), MFU
 *   (either interpolation of varying components c into r(c), with the
 *   barycentric setup, or one SFU op working in place on a register), TEX
 *   (coordinates in a register pair, texel as fixed10 halves into the other
 *   pair), up to 3 ALU packets, and in the last EXEC the DW store.
 * - The colour is written as fixed10 halves to r2/r3 or, when the registers
 *   are tight, to r0/r1, as the NVIDIA compiler does: the DW store takes
 *   either pair.
 * - Within an EXEC the ALU sees what that EXEC's PSEQ/MFU/TEX loaded, and a
 *   packet reads the previous packet's results as alu0..3; its own register
 *   writes are only seen by the following EXECs. A later packet of the EXEC
 *   may see them all the same (a fixed10 half was, fstest fxlum), so no
 *   register is written while a later packet of its EXEC reads it.
 * - ex2/sin/cos take the pre-op on the same register in an earlier EXEC.
 *
 * Phase 1 assigns every unit, varying fetch, texture fetch and the
 * destination load an EXEC (and the ALU units a packet), keeping at most
 * "limit" register halves live; phase 2 gives the items one of the four
 * registers, or of their halves for fixed10 values (fx10.c). A failed phase 2
 * is retried with a lower limit.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grate_common.h"
#include "fp_compiler.h"

#include "util/u_math.h"
#include "util/u_memory.h"

#define SEQ_MAX_REGS 16
#define SEQ_REGS(c) (4 * (c)->seq_rows)
#define SEQ_ROW(c, e) ((e) % (c)->seq_rows)
/* the colour half h lives in the last row, which the DW of the last EXEC
 * stores */
#define SEQ_OUT_REG(c, h) (4 * ((c)->seq_rows - 1) + 2 * (c)->out_pair + (h) / 2)
#define T_PSEQ 0          /* stages of an EXEC, as times 8 * (4 * exec + stage) */
#define T_MFU 1
#define T_TEX 2
#define T_ALU 3           /* ... + 2 * packet in the EXEC, + 1 for its writes */
#define SEQ_T(e, stage) (8 * (4 * (e) + (stage)))
#define SEQ_FOREVER (1 << 20)
/* a schedule or allocation that cannot be: why, with GRATE_DEBUG=tgsi */
#define SEQ_FAIL(...) do { \
   if (grate_debug & GRATE_DEBUG_TGSI) \
      fprintf(stderr, "GRATE FRAG: " __VA_ARGS__); \
   return false; \
} while (0)

struct seq_state {
   int fetch_exec[FP_MAX_INPUTS][4];   /* EXEC interpolating input i comp c */
   int fetch_reg[FP_MAX_INPUTS][4];    /* ... into this register */
   int num_execs;
   int num_packets;
   int dst_exec;
   /* units, then texture fetches, in the order phase 1 considers them */
   bool by_channel;
   /* more rows: varyings fetched when needed, in any round, not all in the
    * first (seq_fetch_inputs) */
   bool late_ipl;
   int last_bary;                      /* EXEC of the last barycentric setup */
   int order[FP_MAX_UNITS + FP_MAX_TEXTURES];
   int rank[FP_MAX_UNITS + FP_MAX_TEXTURES];
};

/*
 * The order phase 1 takes work in: program order or, by_channel, by the
 * first colour write a unit or texture fetch leads to (by register, then in
 * program order), then program order. Vector code comes channel by channel this way, and
 * only one channel's varyings are fetched at a time, so that the few
 * registers hold one channel's chain instead of every channel's first step
 * (int division: all the truncated operands before any quotient).
 */
static void
seq_order(struct fp_compiler *c, struct seq_state *s)
{
   int n = c->num_units + c->num_tex;
   int *rank = s->rank;
   bool changed = s->by_channel;

   /* the colour channels of one register (blue and green, then red and
    * alpha, fp_output_half) together: the other register stays free for
    * temporaries until the last channels */
   for (int i = 0; i < n; i++)
      rank[i] = !s->by_channel ? 0 :
                i < c->num_units && c->units[i].output_chan >= 0 ?
                   fp_output_half(c->units[i].output_chan) / 2 * FP_MAX_UNITS + i :
                   INT_MAX;

   /* a value takes the lowest rank of its readers */
   while (changed) {
      changed = false;
      for (int i = n - 1; i >= 0; i--) {
         if (rank[i] == INT_MAX)
            continue;
         if (i < c->num_units) {
            struct fp_unit *u = &c->units[i];
            fp_unit_foreach_src(u, k, j) {
               struct fp_opnd *o = &u->slots[k].src[j];
               int d = o->kind == FP_OPND_VALUE ? o->index :
                       o->kind == FP_OPND_TEXEL ? c->num_units + o->index : -1;
               if (d >= 0 && rank[i] < rank[d]) {
                  rank[d] = rank[i];
                  changed = true;
               }
            }
         } else {
            int t = i - c->num_units;
            for (int k = 0; k < c->tex[t].num_coords; k++) {
               struct fp_opnd *o = &c->tex[t].coord[k];
               if (o->kind == FP_OPND_VALUE && rank[i] < rank[o->index]) {
                  rank[o->index] = rank[i];
                  changed = true;
               }
            }
         }
      }
   }

   /* insertion sort by (rank, index): stable, the lists are short */
   for (int i = 0; i < n; i++) {
      int k = i;
      while (k > 0 && rank[s->order[k - 1]] > rank[i])
         k--;
      memmove(&s->order[k + 1], &s->order[k], (i - k) * sizeof(int));
      s->order[k] = i;
   }
}

static bool
seq_is_alu(const struct fp_unit *u)
{
   return u->sfu == FP_SFU_OP_NOP;
}

/*
 * Condition registers (measured, asmtri exp-crrows, exp-crlim): cr0..cr4N-1
 * with N rows, one boolean per pixel, read as 0.0 or 1.0; a write of EXEC k
 * is seen from k + 1 with up to 2 rows, from k + 2 with 4 (k + 1 gives the
 * old value), so k + 2 from 3 rows on. The NVIDIA compiler keeps its
 * comparisons there for the CSELs that select by them (cgdump
 * shaders/toobig/atan2.dis: cr0..cr11 on 4 rows), which leaves r0..r15 to
 * the values.
 */
static int
seq_cr_count(const struct fp_compiler *c)
{
   return 4 * c->seq_rows;
}

static int
seq_cr_latency(const struct fp_compiler *c)
{
   return c->seq_rows > 2 ? 2 : 1;
}

/* Whether operand o can be read by an ALU unit in EXEC e, packet p */
static bool
seq_alu_can_read(struct fp_compiler *c, struct seq_state *s,
                 const struct fp_opnd *o, int e, int p)
{
   switch (o->kind) {
   case FP_OPND_VALUE: {
      struct fp_unit *d = &c->units[o->index];
      /* registers written in EXEC k: read from EXEC k + 1 with one row,
       * k + 2 with more (even and odd rows interleave over the 2-clock
       * ALU latency); the SFU result in its own EXEC as well */
      int lat = c->seq_rows > 1 ? 2 : 1;
      if (d->exec < 0)
         return false;
      /* a condition register: from a later EXEC only */
      if (d->cr)
         return d->exec + seq_cr_latency(c) <= e;
      if (!seq_is_alu(d))
         return d->exec == e || d->exec + lat <= e;
      return d->exec + lat <= e ||
             (d->exec == e && d->forward && d->packet == p - 1);
   }
   case FP_OPND_INPUT: {
      int f = s->fetch_exec[o->index][o->comp];
      return f >= 0 && f <= e;
   }
   case FP_OPND_TEXEL:
      return c->tex[o->index].exec >= 0 && c->tex[o->index].exec <= e;
   case FP_OPND_DST:
      return s->dst_exec >= 0 && s->dst_exec <= e;
   default:
      return true;
   }
}

/* Whether a register-held operand is there for the MFU/TEX stage of EXEC e:
 * ALU results of earlier EXECs, SFU results of this one on (TEX only) */
static bool
seq_reg_ready(struct fp_compiler *c, struct seq_state *s,
              const struct fp_opnd *o, int e, bool same_exec_sfu)
{
   int rows = c->seq_rows;

   if (o->kind == FP_OPND_INPUT) {
      int f = s->fetch_exec[o->index][o->comp];
      /* more rows: the SFU of a later round, on the row it went to; a TEX
       * of the EXEC that interpolated (as in multi-row programs) too */
      return f >= 0 && (rows == 1 ? f <= e :
                        (same_exec_sfu && f == e) || (f + rows <= e && (e - f) % rows == 0));
   }
   if (o->kind != FP_OPND_VALUE)
      return false;
   struct fp_unit *d = &c->units[o->index];
   if (d->exec < 0)
      return false;
   if (rows == 1)
      return d->exec < e || (!seq_is_alu(d) && same_exec_sfu && d->exec == e);
   if (!seq_is_alu(d) && same_exec_sfu && d->exec == e)
      return true;
   /* more rows (measured, asmtri exp-wrap5): the MFU of EXEC e sees an
    * ALU write of EXEC k from k + rows on; an SFU result stays on its row */
   return d->exec + rows <= e && (seq_is_alu(d) || (e - d->exec) % rows == 0);
}


/*
 * Register items of phase 1: values (unit results, flexible registers) and
 * the fixed ones: a varying component c in r(c), a texel or the destination
 * colour in a register pair.
 */
enum seq_item_kind { SEQ_NONE, SEQ_VALUE, SEQ_INPUT, SEQ_TEXEL, SEQ_DST, SEQ_OUT };

/* Register halves an item takes: one for a fixed10 value */
static int
seq_item_size(struct fp_compiler *c, enum seq_item_kind kind, int a)
{
   if (kind == SEQ_VALUE && c->units[a].cr)
      return 0;      /* not in r0..r15 */
   return kind == SEQ_VALUE && c->units[a].fx10 ? 1 : 2;
}

/* Condition registers in use after EXEC e (their readers to come) */
static int
seq_live_cr(struct fp_compiler *c, int e)
{
   int n = 0;
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || !u->cr || u->exec < 0)
         continue;
      for (int r = 0; r < c->num_units; r++) {
         struct fp_unit *x = &c->units[r];
         if (!x->live || (x->exec >= 0 && x->exec <= e))
            continue;
         bool reads = false;
         fp_unit_foreach_src(x, k, j)
            reads |= x->slots[k].src[j].kind == FP_OPND_VALUE && x->slots[k].src[j].index == i;
         if (reads) {
            n++;
            break;
         }
      }
   }
   return n;
}

static bool
seq_opnd_is(const struct fp_opnd *o, enum seq_item_kind kind, int a, int b)
{
   switch (kind) {
   case SEQ_VALUE: return o->kind == FP_OPND_VALUE && o->index == a;
   case SEQ_INPUT: return o->kind == FP_OPND_INPUT && o->index == a && o->comp == b;
   case SEQ_TEXEL: return o->kind == FP_OPND_TEXEL && o->index == a;
   case SEQ_DST:   return o->kind == FP_OPND_DST;
   default:        return false;
   }
}

/*
 * The readers of every value and varying component (the program does not
 * change while it is scheduled): units by index, texture fetch n as -1 - n.
 * Key: a unit's value, or FP_MAX_UNITS + 4 * input + component.
 */
#define SEQ_KEYS (FP_MAX_UNITS + 4 * FP_MAX_INPUTS)
struct seq_readers {
   int start[SEQ_KEYS + 1];
   int list[FP_MAX_UNITS * 12 + FP_MAX_TEXTURES * 4];
};

static int
seq_reader_key(const struct fp_opnd *o)
{
   return o->kind == FP_OPND_VALUE ? o->index :
          o->kind == FP_OPND_INPUT ? FP_MAX_UNITS + 4 * o->index + o->comp : -1;
}

static void
seq_build_readers(struct fp_compiler *c, struct seq_readers *rd)
{
   int count[SEQ_KEYS] = { 0 };

   for (int pass = 0; pass < 2; pass++) {
      int fill[SEQ_KEYS];
      if (pass) {
         rd->start[0] = 0;
         for (int k = 0; k < SEQ_KEYS; k++)
            rd->start[k + 1] = rd->start[k] + count[k];
         memcpy(fill, rd->start, sizeof(fill));
      }
      for (int i = 0; i < c->num_units + c->num_tex; i++) {
         const struct fp_opnd *ops[12];
         int n = 0, last[12];
         if (i < c->num_units) {
            fp_unit_foreach_src(&c->units[i], k, j)
               ops[n++] = &c->units[i].slots[k].src[j];
         } else {
            int t = i - c->num_units;
            for (int k = 0; k < c->tex[t].num_coords; k++)
               ops[n++] = &c->tex[t].coord[k];
         }
         int reader = i < c->num_units ? i : -1 - (i - c->num_units);
         for (int k = 0; k < n; k++) {
            int key = seq_reader_key(ops[k]);
            bool seen = false;
            for (int m = 0; m < k; m++)
               seen |= last[m] == key;
            last[k] = key;
            if (key < 0 || seen)
               continue;
            if (pass)
               rd->list[fill[key]++] = reader;
            else
               count[key]++;
         }
      }
   }
}

/* Whether an item is still read in EXEC e or later (or not yet scheduled);
 * "except": a unit not to count, -2: no colour write counts */
static bool
seq_item_needed(struct fp_compiler *c, enum seq_item_kind kind, int a, int b,
                int e, int except)
{
   struct seq_readers *rd = c->seq_readers;
   if (rd && (kind == SEQ_VALUE || kind == SEQ_INPUT)) {
      int key = kind == SEQ_VALUE ? a : FP_MAX_UNITS + 4 * a + b;
      for (int k = rd->start[key]; k < rd->start[key + 1]; k++) {
         int r = rd->list[k];
         if (r < 0) {
            int n = -1 - r;
            if (c->tex[n].live && !(c->tex[n].exec >= 0 && c->tex[n].exec < e))
               return true;
            continue;
         }
         struct fp_unit *u = &c->units[r];
         if (!u->live || r == except || (u->exec >= 0 && u->exec < e) ||
             (except == -2 && u->output_chan >= 0))
            continue;
         return true;
      }
      return false;
   }

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || i == except || (u->exec >= 0 && u->exec < e) ||
          (except == -2 && u->output_chan >= 0))
         continue;
      fp_unit_foreach_src(u, k, j)
         if (seq_opnd_is(&u->slots[k].src[j], kind, a, b))
            return true;
   }
   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live || (c->tex[n].exec >= 0 && c->tex[n].exec < e))
         continue;
      for (int k = 0; k < c->tex[n].num_coords; k++)
         if (seq_opnd_is(&c->tex[n].coord[k], kind, a, b))
            return true;
   }
   return false;
}

struct seq_owner {
   enum seq_item_kind kind;
   int a, b;
};

/* Whether a texel or destination half in register "reg" is still read in
 * EXEC e or later by a unit other than "except" */
static bool
seq_half_needed(struct fp_compiler *c, enum seq_item_kind kind, int a,
                int reg, int high, int e, int except)
{
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || i == except || (u->exec >= 0 && u->exec < e))
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (!seq_opnd_is(o, kind, a, 0))
            continue;
         int h = fp_output_half(o->comp);
         int pair = kind == SEQ_TEXEL ? 1 - c->tex[a].pair : c->dst_pair;
         int base = 4 * SEQ_ROW(c, kind == SEQ_TEXEL ? c->tex[a].exec : c->dst_exec);
         if (base + 2 * pair + h / 2 == reg && h % 2 == high)
            return true;
      }
   }
   return false;
}

/* Whether unit i is one of the coordinates of fetch n */
static bool
seq_coord_of(struct fp_compiler *c, int i, int n)
{
   for (int k = 0; k < c->tex[n].num_coords; k++)
      if (seq_opnd_is(&c->tex[n].coord[k], SEQ_VALUE, i, 0))
         return true;
   return false;
}

/*
 * Whether computing unit i, a texture coordinate, has to wait: with the
 * coordinates of other fetches computed and their TEXs still to come, it
 * may only take a register while there stays room for a texel pair ("room"
 * halves left after it), else the coordinates hold the registers and no TEX
 * finds a pair for its texel (one row: a 2D fetch and a cube map's three
 * coordinates; more: a blur's taps, all computed at once). The next ones
 * still go in the EXEC of such a TEX; those of a fetch begun do not wait.
 */
static bool
seq_coord_waits(struct fp_compiler *c, int i, int room)
{
   bool coord = false;
   for (int n = 0; n < c->num_tex && !coord; n++) {
      if (!c->tex[n].live || !seq_coord_of(c, i, n))
         continue;
      coord = true;
      /* the rest of a fetch begun goes on: it frees its registers sooner */
      for (int k = 0; k < c->tex[n].num_coords; k++) {
         struct fp_opnd *o = &c->tex[n].coord[k];
         if (o->kind == FP_OPND_VALUE && o->index != i && c->units[o->index].exec >= 0)
            return false;
      }
   }
   if (!coord)
      return false;
   int pending = 0;
   for (int m = 0; m < c->num_tex; m++) {
      if (!c->tex[m].live || c->tex[m].exec >= 0 || seq_coord_of(c, i, m))
         continue;
      for (int k = 0; k < c->tex[m].num_coords; k++) {
         struct fp_opnd *o = &c->tex[m].coord[k];
         if (o->kind == FP_OPND_VALUE && c->units[o->index].exec >= 0) {
            pending++;
            break;
         }
      }
   }
   return pending > 0 && room < 4;
}

/* Whether a cube map's r and a bias of fetch n are there for EXEC e */
static bool
seq_extra_coords_ready(struct fp_compiler *c, struct seq_state *s, int n, int e)
{
   for (int k = 2; k < c->tex[n].num_coords; k++)
      if (!seq_reg_ready(c, s, &c->tex[n].coord[k], e, true))
         return false;
   return true;
}

/*
 * Whether EXEC e of a program on more rows needs an ALU packet, an empty one
 * if it has none. Row 0 always (one there without kills the output, asmtri
 * exp-noalu). Elsewhere a lone EXEC without one is fine (exp-noalu3), but
 * the ALU latency counts packets, not EXECs: an ALU write of EXEC k read by
 * the ALU of k + 2 across an EXEC k + 1 without a packet gives all zeros
 * (exp-noalu4, 4 rows). So the packet stays where an ALU value written
 * before e is read after it by the ALU within 2 EXECs. An SFU op reading it
 * N EXECs on sees it across EXECs without one (exp-noalu5, N = 2..4); a
 * TEX, not measured, keeps the packet within rows + 1.
 */
static int seq_pad_why;   /* 0 row 0, 1 ALU read, 2 SFU/TEX read (debug) */
static bool
seq_needs_alu(struct fp_compiler *c, int e)
{
   seq_pad_why = 0;
   int rows = c->seq_rows;

   if (rows == 1)
      return false;
   if (e % rows == 0 || getenv("GRATE_FP_ALL_PACKETS"))
      return true;
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *r = &c->units[i];
      if (!r->live || r->exec <= e || !seq_is_alu(r))
         continue;
      int reach = 2;
      fp_unit_foreach_src(r, k, j) {
         const struct fp_opnd *o = &r->slots[k].src[j];
         if (o->kind != FP_OPND_VALUE || !seq_is_alu(&c->units[o->index]))
            continue;
         int w = c->units[o->index].exec;
         if (w < e && r->exec - w <= reach) {
            seq_pad_why = 1;
            return true;
         }
      }
   }
   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live || c->tex[n].exec <= e)
         continue;
      for (int k = 0; k < c->tex[n].num_coords; k++) {
         const struct fp_opnd *o = &c->tex[n].coord[k];
         if (o->kind != FP_OPND_VALUE || !seq_is_alu(&c->units[o->index]))
            continue;
         int w = c->units[o->index].exec;
         if (w < e && c->tex[n].exec - w <= rows + 1) {
            seq_pad_why = 2;
            return true;
         }
      }
   }
   return false;
}

/* Whether SFU unit i is in the in-place chain that ends in coordinate o */
static bool
seq_sfu_chain_has(struct fp_compiler *c, const struct fp_opnd *o, int i)
{
   if (o->kind != FP_OPND_VALUE)
      return false;
   for (int w = o->index; c->units[w].sfu; ) {
      if (w == i)
         return true;
      const struct fp_opnd *x = &c->units[w].slots[0].src[0];
      if (x->kind != FP_OPND_VALUE)
         break;
      w = x->index;
   }
   return false;
}

/*
 * Whether SFU unit i can go in EXEC e: an SFU result stays on its row, and
 * a TEX reads its coordinates from one row, so the SFU chains of a fetch's
 * coordinates go on the row of the first one begun (TXB of ex2(x).xy: s and
 * t on two rows, and no TEX could run, random.all_features.fragment.92)
 */
static bool
seq_sfu_row_ok(struct fp_compiler *c, int i, int e)
{
   int rows = c->seq_rows;

   if (rows == 1)
      return true;
   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live || c->tex[n].exec >= 0)
         continue;
      int mine = -1;
      for (int k = 0; k < c->tex[n].num_coords && mine < 0; k++)
         if (seq_sfu_chain_has(c, &c->tex[n].coord[k], i))
            mine = k;
      if (mine < 0)
         continue;
      for (int k = 0; k < c->tex[n].num_coords; k++) {
         const struct fp_opnd *o = &c->tex[n].coord[k];
         if (k == mine || o->kind != FP_OPND_VALUE)
            continue;
         for (int w = o->index; c->units[w].sfu; ) {
            if (c->units[w].exec >= 0 && (e - c->units[w].exec) % rows)
               return false;
            const struct fp_opnd *x = &c->units[w].slots[0].src[0];
            if (x->kind != FP_OPND_VALUE)
               break;
            w = x->index;
         }
      }
   }
   return true;
}

/* Whether the MFU of an EXEC has nothing to do yet */
static bool
seq_mfu_free(const struct fp_seq_exec *x)
{
   for (int k = 0; k < 4; k++)
      if (x->ipl_input[k] >= 0)
         return false;
   return x->sfu_unit < 0 && !x->cube_mfu;
}

/* Whether a fixed register can be loaded in EXEC e (by its PSEQ, MFU or
 * TEX: before that EXEC's ALU reads) */
static bool
seq_fixed_free(struct fp_compiler *c, const struct seq_owner *own, int e)
{
   if (own->kind == SEQ_OUT)
      return false;
   return own->kind == SEQ_NONE ||
          !seq_item_needed(c, own->kind, own->a, own->b, e, -1);
}

/* Whether the fixed item "own" is read after unit w writes: in a later EXEC,
 * or in a later packet of w's (those up to w's packet read before it
 * writes, as r = trunc(x) over x; the TEX and MFU of w's EXEC before) */
static bool
seq_read_after(struct fp_compiler *c, const struct seq_owner *own,
               const struct fp_unit *w)
{
   if (seq_item_needed(c, own->kind, own->a, own->b, w->exec + 1, -1))
      return true;
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || u->exec != w->exec || u->packet <= w->packet)
         continue;
      fp_unit_foreach_src(u, k, j)
         if (seq_opnd_is(&u->slots[k].src[j], own->kind, own->a, own->b))
            return true;
   }
   return false;
}

/*
 * Whether the computed coordinates of fetch n can take the registers pair p
 * puts them in (s, t in 2p, 2p + 1; r, a bias in 2q, 2q + 1): each value
 * owns its register from its own write on (seq_alloc), so a fixed item
 * there must not be read after that write.
 */
static bool
seq_coord_regs_free(struct fp_compiler *c, const struct seq_owner *owner,
                    int n, int base, int p)
{
   for (int k = 0; k < c->tex[n].num_coords; k++) {
      const struct fp_opnd *o = &c->tex[n].coord[k];
      if (o->kind != FP_OPND_VALUE)
         continue;
      int role = c->tex[n].role[k];
      int reg = base + (role < FP_TEX_R ? 2 * p + role : 2 * (1 - p) + role - FP_TEX_R);
      const struct seq_owner *own = &owner[reg];
      if (own->kind == SEQ_OUT ||
          (own->kind != SEQ_NONE && seq_read_after(c, own, &c->units[o->index])))
         return false;
   }
   return true;
}

/* Register halves live after EXEC e */
static int
seq_live_items(struct fp_compiler *c, const struct seq_owner *owner, int e)
{
   int n = 0;

   for (int r = 0; r < SEQ_REGS(c); r++) {
      if (seq_fixed_free(c, &owner[r], e + 1))
         continue;
      if (owner[r].kind == SEQ_TEXEL || owner[r].kind == SEQ_DST)
         for (int h = 0; h < 2; h++)
            n += seq_half_needed(c, owner[r].kind, owner[r].a, r, h, e + 1, -1);
      else
         n += 2;
   }
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (u->live && u->exec >= 0 && fp_unit_has_value(u) &&
          seq_item_needed(c, SEQ_VALUE, i, 0, e + 1, -1))
         n += seq_item_size(c, SEQ_VALUE, i);
   }
   return n;
}

/* Change of the live halves when unit i is scheduled in EXEC e */
static int
seq_unit_delta(struct fp_compiler *c, int i, int e)
{
   struct fp_unit *u = &c->units[i];
   int delta = fp_unit_has_value(u) ? seq_item_size(c, SEQ_VALUE, i) : 0;

   fp_unit_foreach_src(u, k, j) {
      struct fp_opnd *o = &u->slots[k].src[j];
      enum seq_item_kind kind = o->kind == FP_OPND_VALUE ? SEQ_VALUE :
                                o->kind == FP_OPND_INPUT ? SEQ_INPUT :
                                o->kind == FP_OPND_TEXEL ? SEQ_TEXEL :
                                o->kind == FP_OPND_DST ? SEQ_DST : SEQ_NONE;
      if (kind == SEQ_NONE)
         continue;
      /* each item once per unit */
      bool seen = false;
      fp_unit_foreach_src(u, k2, j2) {
         if (k2 * 3 + j2 >= k * 3 + j)
            break;
         if (seq_opnd_is(&u->slots[k2].src[j2], kind, o->index, o->comp))
            seen = true;
      }
      if (!seen && !seq_item_needed(c, kind, o->index, o->comp, e + 1, i))
         delta -= seq_item_size(c, kind, o->index);
   }
   return delta;
}

static bool
seq_reads_dst(struct fp_unit *u)
{
   fp_unit_foreach_src(u, k, j)
      if (u->slots[k].src[j].kind == FP_OPND_DST)
         return true;
   return false;
}

/* Whether the operands of unit u other than the varyings and the
 * destination colour can be read in EXEC e */
static bool
seq_others_ready(struct fp_compiler *c, struct seq_state *s,
                 struct fp_unit *u, int e, enum fp_opnd_kind skip)
{
   fp_unit_foreach_src(u, k, j) {
      struct fp_opnd *o = &u->slots[k].src[j];
      if (o->kind == skip)
         continue;
      if (seq_is_alu(u) ? !seq_alu_can_read(c, s, o, e, 0) :
                          !seq_reg_ready(c, s, o, e, false))
         return false;
   }
   return true;
}

/* Whether input i component comp is a texture coordinate: those must be
 * fetched into r(comp), the register pair the TEX reads */
static bool
seq_is_coord(struct fp_compiler *c, int i, int comp)
{
   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < c->tex[n].num_coords; k++)
         if (c->tex[n].live && seq_opnd_is(&c->tex[n].coord[k], SEQ_INPUT, i, comp))
            return true;
   return false;
}

/*
 * Whether the pair at register "pair" holds varying coordinates of a fetch
 * still to come other than n: the texel of either would then go over the
 * other's coordinates, and neither TEX could run (mpv's YUV planes: two
 * fetches of one varying's .xy and .zw)
 */
static bool
seq_pair_holds_coords(struct fp_compiler *c, const struct seq_owner *owner,
                      int pair, int n)
{
   for (int r = pair; r < pair + 2; r++) {
      if (owner[r].kind != SEQ_INPUT)
         continue;
      for (int m = 0; m < c->num_tex; m++) {
         if (m == n || !c->tex[m].live || c->tex[m].exec >= 0)
            continue;
         for (int k = 0; k < c->tex[m].num_coords; k++)
            if (seq_opnd_is(&c->tex[m].coord[k], SEQ_INPUT, owner[r].a, owner[r].b))
               return true;
      }
   }
   return false;
}

/*
 * A register the MFU of this EXEC can interpolate input i component comp
 * into, -1 if none: r(comp) preferably, the column the linker would give it
 * anyway, else any free one (the TRAM layout follows, fp/linker.c). A
 * texture coordinate goes where its TEX reads it: s in the first register
 * of a pair, t in the second, next to the other one if that is there
 * already (any component in any column: .yz or .zx as well as .xy).
 */
static int
seq_fetch_reg(struct fp_compiler *c, struct seq_state *s, const struct seq_owner *owner,
              const struct fp_seq_exec *x, int i, int comp, int e)
{
   int base = 4 * SEQ_ROW(c, e);
#define SEQ_COL_FREE(col) (x->ipl_input[col] < 0 && seq_fixed_free(c, &owner[base + (col)], e))

   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live)
         continue;
      for (int k = 0; k < 2 && k < c->tex[n].num_coords; k++) {
         if (!seq_opnd_is(&c->tex[n].coord[k], SEQ_INPUT, i, comp))
            continue;
         const struct fp_opnd *o = &c->tex[n].coord[1 - k];
         int other = o->kind == FP_OPND_INPUT ? s->fetch_reg[o->index][o->comp] : -1;
         bool other_in = o->kind == FP_OPND_INPUT;
         for (int p = 0; p < 2; p++) {
            int col = 2 * p + k, ocol = 2 * p + 1 - k;
            if (!SEQ_COL_FREE(col) || seq_pair_holds_coords(c, owner, base + 2 * (1 - p), n))
               continue;
            if (other_in && other >= 0 ? other == base + ocol :
                other_in ? SEQ_COL_FREE(ocol) : true)
               return base + col;
         }
         return -1;
      }
   }

   /* computed s, t and r (a cube map's, always) or q go to r0, r1, r2 with
    * pair 0 (below): other components keep out of them while such a fetch
    * is to come, as the NVIDIA compiler has them (cgdump shaders/cubebig:
    * x in r3 for r = trunc(x)) */
   bool triple = false;
   for (int n = 0; n < c->num_tex; n++)
      triple |= c->tex[n].live && c->tex[n].exec < 0 &&
                fp_tex_coord(c, n, FP_TEX_R) >= 0;
   for (int pass = triple ? 0 : 1; pass < 2; pass++)
      for (int k = 0; k < 4; k++) {
         int col = (comp + k) % 4;
         if ((pass || col == 3) && SEQ_COL_FREE(col))
            return base + col;
      }
   return -1;
#undef SEQ_COL_FREE
}

/* The lowest rank of the work left but the colour writes (constant ones
 * wait for the end): by_channel, only its varyings are fetched (whatever
 * it reads has a rank as low) */
static int
seq_pending_rank(struct fp_compiler *c, struct seq_state *s)
{
   for (int n = 0; n < c->num_units + c->num_tex; n++) {
      int i = s->order[n];
      bool pending = i < c->num_units ?
         c->units[i].live && c->units[i].exec < 0 && c->units[i].output_chan < 0 :
         c->tex[i - c->num_units].live && c->tex[i - c->num_units].exec < 0;
      if (pending)
         return s->rank[i];
   }
   return INT_MAX;
}

/* MFU interpolation: the varying components of the next units (in the
 * order of seq_order, by_channel those of the lowest rank left) whose
 * other operands are there, each into a free register */
static bool
seq_fetch_inputs(struct fp_compiler *c, struct seq_state *s,
                 struct seq_owner *owner, struct fp_seq_exec *x, int e,
                 int limit)
{
   bool any = false;
   int live = seq_live_items(c, owner, e - 1);
   int rank = seq_pending_rank(c, s);
   /*
    * An EXEC interpolates with the weights of a barycentric setup in it or
    * in an earlier EXEC of the same round (measured, asmtri exp-late: one
    * on the last row does not reach the next round's first): EXEC 0's
    * covers the first round. More rows interpolate everything that fits
    * in the first round first; late_ipl (when that does not fit) fetches
    * when needed instead, as one row does, with a setup of its own in an
    * EXEC whose SFU is free (as NVIDIA does: cgdump shaders/big/bump.dis).
    */
   bool just_in_time = c->seq_rows == 1 || s->late_ipl;
   bool by_rank = s->by_channel && just_in_time;
   bool covered = e < c->seq_rows ||
                  (s->last_bary >= 0 && s->last_bary / c->seq_rows == e / c->seq_rows);

   if (c->seq_rows > 1 && !covered && !s->late_ipl)
      return false;

   /* ... spread over the rows of that round, so that each keeps registers
    * for the values its SFU ops work on (in place, on their own row) */
   int cap = 4;
   if (c->seq_rows > 1 && !s->late_ipl) {
      bool used[FP_MAX_INPUTS][4] = { { false } };
      int total = 0;
      for (int i = 0; i < c->num_units; i++)
         if (c->units[i].live)
            fp_unit_foreach_src(&c->units[i], k, j) {
               struct fp_opnd *o = &c->units[i].slots[k].src[j];
               if (o->kind == FP_OPND_INPUT && !used[o->index][o->comp]) {
                  used[o->index][o->comp] = true;
                  total++;
               }
            }
      for (int n = 0; n < c->num_tex; n++)
         if (c->tex[n].live)
            for (int k = 0; k < c->tex[n].num_coords; k++) {
               struct fp_opnd *o = &c->tex[n].coord[k];
               if (o->kind == FP_OPND_INPUT && !used[o->index][o->comp]) {
                  used[o->index][o->comp] = true;
                  total++;
               }
            }
      cap = DIV_ROUND_UP(total, c->seq_rows);
   }
   int fetched = 0;

   for (int n_ = 0; n_ < c->num_units + c->num_tex; n_++) {
      int i = s->order[n_];
      struct fp_opnd *ops[12];
      int n = 0;

      if (by_rank && s->rank[i] < rank)
         continue;
      if (by_rank && s->rank[i] > rank)
         break;

      if (i < c->num_units) {
         struct fp_unit *u = &c->units[i];
         /* just in time: when the rest is there, to keep registers free;
          * else everything in the first round */
         if (!u->live || u->exec >= 0 ||
             (just_in_time && !seq_others_ready(c, s, u, e, FP_OPND_INPUT)))
            continue;
         /* colour writes wait for the rest (seq_unit_ready): their
          * varyings would only take registers meanwhile */
         if (u->output_chan >= 0 && !c->redirect_output && !s->by_channel &&
             just_in_time) {
            bool pending = false;
            for (int r = 0; r < c->num_units; r++)
               pending |= c->units[r].live && c->units[r].output_chan < 0 &&
                          c->units[r].exec < 0;
            if (pending)
               continue;
         }
         fp_unit_foreach_src(u, k, j)
            ops[n++] = &u->slots[k].src[j];
      } else {
         int t = i - c->num_units;
         if (!c->tex[t].live || c->tex[t].exec >= 0)
            continue;
         for (int k = 0; k < c->tex[t].num_coords; k++)
            ops[n++] = &c->tex[t].coord[k];
      }

      for (int k = 0; k < n; k++) {
         struct fp_opnd *o = ops[k];
         int comp = o->comp;
         /* (texture coordinates go together, to one pair: not capped) */
         if (o->kind != FP_OPND_INPUT || s->fetch_exec[o->index][comp] >= 0 ||
             live + 2 > limit ||
             (fetched >= cap && !seq_is_coord(c, o->index, comp)))
            continue;
         int r = seq_fetch_reg(c, s, owner, x, o->index, comp, e);
         if (r < 0)
            continue;
         x->ipl_input[r % 4] = o->index;
         x->ipl_comp[r % 4] = comp;
         s->fetch_exec[o->index][comp] = e;
         s->fetch_reg[o->index][comp] = r;
         owner[r] = (struct seq_owner) { SEQ_INPUT, o->index, comp };
         live += 2;
         fetched++;
         any = true;
      }
   }
   /* weights of its own (its SFU is free: seq_schedule) */
   if (any && !covered) {
      x->bary = true;
      s->last_bary = e;
   }
   return any;
}

/*
 * Whether ALU unit "index" can go in EXEC e, packet p. A colour write makes
 * r2 or r3 the colour's for good: only once nothing fixed is still needed
 * there and no varying component, texture fetch or destination load that
 * might want it is pending.
 */
static bool
seq_unit_ready(struct fp_compiler *c, struct seq_state *s,
               const struct seq_owner *owner, int index, int e, int p)
{
   struct fp_unit *u = &c->units[index];

   fp_unit_foreach_src(u, k, j)
      if (!seq_alu_can_read(c, s, &u->slots[k].src[j], e, p))
         return false;

   if (u->output_chan < 0)
      return true;

   /* without blending the colour is written last: the registers are
    * worth more as temporaries until then; by_channel a channel whose
    * write frees a value goes as soon as it is there, a fixed10 half
    * instead of a register held */
   if (!c->redirect_output && (!s->by_channel || seq_unit_delta(c, index, e) >= 0))
      for (int i = 0; i < c->num_units; i++)
         if (c->units[i].live && c->units[i].output_chan < 0 && c->units[i].exec < 0)
            return false;

   int half = fp_output_half(u->output_chan);
   int reg = SEQ_OUT_REG(c, half);
   switch (owner[reg].kind) {
   case SEQ_NONE:
   case SEQ_OUT:
      break;
   case SEQ_TEXEL:
   case SEQ_DST:
      /* fixed10 halves: only the half written must be free */
      if (seq_half_needed(c, owner[reg].kind, owner[reg].a, reg, half % 2, e + 1, index))
         return false;
      break;
   default:
      /* other colour writes go in this EXEC too (they are ready together)
       * and read the register before the writes take effect */
      if (seq_item_needed(c, owner[reg].kind, owner[reg].a, owner[reg].b, e + 1, -2))
         return false;
   }
   if (c->dst_used && s->dst_exec < 0)
      return false;
   for (int n = 0; n < c->num_tex; n++)
      if (c->tex[n].live && c->tex[n].exec < 0)
         return false;
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *r = &c->units[i];
      if (i == index || !r->live)
         continue;
      /* earlier writes of the channel first */
      if (r->output_chan == u->output_chan && i < index &&
          (r->exec < 0 || (r->exec == e && r->packet >= p)))
         return false;
      if (r->exec >= 0)
         continue;
      fp_unit_foreach_src(r, k, j) {
         struct fp_opnd *o = &r->slots[k].src[j];
         if (o->kind == FP_OPND_INPUT && o->comp == reg % 4 &&
             s->fetch_exec[o->index][o->comp] < 0 &&
             seq_is_coord(c, o->index, o->comp))
            return false;
      }
   }
   return true;
}

/*
 * Whether the result of unit i, put in packet "packet" (the p-th of EXEC e),
 * can go to its only reader by forwarding: the reader then fits in the next
 * packet of the same EXEC, its other operands being there by then. Such a
 * value takes no register.
 */
static bool
seq_forwardable(struct fp_compiler *c, struct seq_state *s, int i, int e,
                int packet, int p)
{
   int reader = -1;

   if (p + 1 >= FP_MAX_PACKETS_PER_EXEC || c->units[i].output_chan >= 0 || c->units[i].cr)
      return false;
   for (int r = 0; r < c->num_units; r++) {
      struct fp_unit *u = &c->units[r];
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j)
         if (seq_opnd_is(&u->slots[k].src[j], SEQ_VALUE, i, 0)) {
            if (reader >= 0 && reader != r)
               return false;
            reader = r;
         }
   }
   for (int n = 0; n < c->num_tex; n++)
      for (int k = 0; k < c->tex[n].num_coords; k++)
         if (c->tex[n].live && seq_opnd_is(&c->tex[n].coord[k], SEQ_VALUE, i, 0))
            return false;
   if (reader < 0 || !seq_is_alu(&c->units[reader]) || c->units[reader].exec >= 0)
      return false;

   struct fp_unit *r = &c->units[reader];
   fp_unit_foreach_src(r, k, j) {
      struct fp_opnd *o = &r->slots[k].src[j];
      if (seq_opnd_is(o, SEQ_VALUE, i, 0))
         continue;
      /* other results of this packet are forwarded too */
      if (o->kind == FP_OPND_VALUE && c->units[o->index].exec == e &&
          c->units[o->index].packet == packet && seq_is_alu(&c->units[o->index]))
         continue;
      if (!seq_alu_can_read(c, s, o, e, packet + 1))
         return false;
   }
   return true;
}

/* The units with their operands and where phase 1 put them, the register
 * owners: what a stuck schedule looks like */
static void
seq_dump(struct fp_compiler *c, struct seq_state *s, const struct seq_owner *owner)
{
   static const char kinds[] = "uvitd01s";   /* by enum fp_opnd_kind */

   for (int r = 0; r < SEQ_REGS(c); r++)
      fprintf(stderr, "GRATE FRAG:   r%d: owner kind %d (%d.%d)\n", r,
              owner[r].kind, owner[r].a, owner[r].b);
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live)
         continue;
      fprintf(stderr, "GRATE FRAG:   u%d rank %d e%d.%d %s%d%s:", i, s->rank[i],
              u->exec, u->packet, u->sfu ? "sfu" : "alu",
              u->sfu ? (int)u->sfu : (int)u->slots[0].op,
              u->output_chan >= 0 ? " out" : u->sfu_copy ? " copy" : "");
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE)
            fprintf(stderr, " u%d", o->index);
         else if (o->kind == FP_OPND_INPUT)
            fprintf(stderr, " in%d.%d", o->index, o->comp);
         else
            fprintf(stderr, " %c", o->kind < sizeof(kinds) - 1 ? kinds[o->kind] : '?');
      }
      fprintf(stderr, "\n");
   }
}

/* Phase 1 */
static bool
seq_schedule(struct fp_compiler *c, struct seq_state *s, int limit)
{
   struct seq_owner owner[SEQ_MAX_REGS] = { { SEQ_NONE, 0, 0 } };
   int remaining = 0, packet = 0, idle = 0;

   memset(s->fetch_exec, -1, sizeof(s->fetch_exec));
   memset(s->fetch_reg, -1, sizeof(s->fetch_reg));
   s->dst_exec = -1;
   s->last_bary = -1;
   seq_order(c, s);
   for (int i = 0; i < c->num_units; i++) {
      c->units[i].exec = c->units[i].packet = -1;
      remaining += c->units[i].live;
   }
   for (int n = 0; n < c->num_tex; n++) {
      c->tex[n].exec = -1;
      remaining += c->tex[n].live;
   }

   for (int e = 0; remaining; e++) {
      struct fp_seq_exec *x = &c->seq_exec[e];
      bool progress = false;

      if (e == FP_SEQ_MAX_EXECS)
         SEQ_FAIL("phase 1: out of EXECs (%d)\n", e);
      memset(x, 0, sizeof(*x));
      x->sfu_unit = x->tex = -1;
      for (int k = 0; k < 4; k++)
         x->ipl_input[k] = -1;
      x->first_packet = packet;

      /* MFU: an SFU op whose operand is in its register, else varyings */
      for (int n_ = 0; n_ < c->num_units + c->num_tex; n_++) {
         int i = s->order[n_];
         if (i >= c->num_units)
            continue;
         struct fp_unit *u = &c->units[i];
         if (!u->live || seq_is_alu(u) || u->exec >= 0 ||
             !seq_reg_ready(c, s, &u->slots[0].src[0], e, false) ||
             !seq_sfu_row_ok(c, i, e))
            continue;
         u->exec = e;
         x->sfu_unit = i;
         remaining--;
         progress = true;
         break;
      }
      if (x->sfu_unit < 0)
         progress |= seq_fetch_inputs(c, s, owner, x, e, limit);

      /* PSEQ: once a reader of the destination has its other operands,
       * into a free register pair */
      if (c->dst_used && s->dst_exec < 0 &&
          seq_live_items(c, owner, e - 1) + 4 <= limit) {
         for (int i = 0; i < c->num_units; i++) {
            struct fp_unit *u = &c->units[i];
            if (!u->live || u->exec >= 0 || !seq_reads_dst(u) ||
                !seq_others_ready(c, s, u, e, FP_OPND_DST))
               continue;
            int base = 4 * SEQ_ROW(c, e);   /* row-relative */
            for (int p = 0; p < 2; p++) {
               if (!seq_fixed_free(c, &owner[base + 2 * p], e) ||
                   !seq_fixed_free(c, &owner[base + 2 * p + 1], e))
                  continue;
               s->dst_exec = c->dst_exec = e;
               c->dst_pair = p;
               owner[base + 2 * p] = owner[base + 2 * p + 1] =
                  (struct seq_owner) { SEQ_DST, 0, 0 };
               x->pseq = true;
               progress = true;
               break;
            }
            break;
         }
      }

      /* TEX: coordinates in pair p (a varying .xy/.zw keeps its pair),
       * texel into the other one */
      for (int n = 0; n < c->num_tex && x->tex < 0; n++) {
         if (!c->tex[n].live || c->tex[n].exec >= 0 ||
             !seq_reg_ready(c, s, &c->tex[n].coord[0], e, true) ||
             !seq_reg_ready(c, s, &c->tex[n].coord[1], e, true) ||
             !seq_extra_coords_ready(c, s, n, e) ||
             (fp_tex_coord(c, n, FP_TEX_R) >= 0 && !seq_mfu_free(x)) ||
             /* the texel takes a pair; r and a bias are already in it */
             seq_live_items(c, owner, e - 1) + 4 -
                2 * (c->tex[n].num_coords - 2) > limit)
            continue;
         /* the TEX works on the pairs of its EXEC's row */
         int base = 4 * SEQ_ROW(c, e);
         for (int p = 0; p < 2; p++) {
            struct fp_opnd *s0 = &c->tex[n].coord[0], *s1 = &c->tex[n].coord[1];
            /* a cube map's computed coordinates: pair 0 only, r in r2, as
             * the NVIDIA compiler does (rcp r10); with pair 1, r in r0 and
             * rcp r8, random.texture.fragment.38 renders wrong */
            if (p == 1 && c->tex[n].cube && fp_tex_coord(c, n, FP_TEX_R) >= 0)
               continue;
            if ((s0->kind == FP_OPND_INPUT &&
                 s->fetch_reg[s0->index][s0->comp] != base + 2 * p) ||
                (s1->kind == FP_OPND_INPUT &&
                 s->fetch_reg[s1->index][s1->comp] != base + 2 * p + 1))
               continue;
            int q = 1 - p;
            if (!seq_fixed_free(c, &owner[base + 2 * q], e) ||
                !seq_fixed_free(c, &owner[base + 2 * q + 1], e))
               continue;
            /* computed coordinates: no fixed item may be in their registers
             * while they are there */
            if (!seq_coord_regs_free(c, owner, n, base, p))
               continue;
            c->tex[n].exec = e;
            c->tex[n].pair = p;
            owner[base + 2 * q] = owner[base + 2 * q + 1] =
               (struct seq_owner) { SEQ_TEXEL, n, 0 };
            x->tex = n;
            x->cube_mfu = fp_tex_coord(c, n, FP_TEX_R) >= 0;
            remaining--;
            progress = true;
            break;
         }
      }

      /* ALU packets: closing items first, at most "limit" live; room is
       * kept for the destination colour load unless that blocks everything */
      bool reserve = c->dst_used && s->dst_exec < 0;
      for (int p = 0; p < FP_MAX_PACKETS_PER_EXEC && remaining; p++) {
         unsigned mask = 0;
         for (;;) {
            int best = -1, best_delta = 0;
            for (int i = 0; i < c->num_units; i++) {
               struct fp_unit *u = &c->units[i];
               if (!u->live || !seq_is_alu(u) || u->exec >= 0 ||
                   !seq_unit_ready(c, s, owner, i, e, packet))
                  continue;
               if (u->num_slots > 1 ? (mask & ((1u << u->num_slots) - 1)) != 0 :
                                      mask == 0xf)
                  continue;
               /* a comparison into a condition register while one is free */
               if (u->cr && seq_live_cr(c, e) >= seq_cr_count(c))
                  continue;
               int delta = seq_unit_delta(c, i, e);
               int room = limit - (reserve && s->dst_exec < 0 ? 4 : 0);
               if (seq_coord_waits(c, i, room - seq_live_items(c, owner, e) - MAX2(delta, 0)))
                  continue;
               if (delta > 0 && seq_live_items(c, owner, e) + delta > room &&
                   !seq_forwardable(c, s, i, e, packet, p))
                  continue;
               if (best < 0 || delta < best_delta ||
                   (delta == best_delta && s->rank[i] < s->rank[best])) {
                  best = i;
                  best_delta = delta;
               }
            }
            if (best < 0 && reserve && !mask && !progress) {
               reserve = false;
               continue;
            }
            if (best < 0)
               break;
            struct fp_unit *u = &c->units[best];
            u->exec = e;
            u->packet = packet;
            u->slot = u->num_slots > 1 ? 0 : ffs(~mask & 0xf) - 1;
            mask |= ((1u << u->num_slots) - 1) << u->slot;
            remaining--;
            if (u->output_chan >= 0)
               owner[SEQ_OUT_REG(c, fp_output_half(u->output_chan))] =
                  (struct seq_owner) { SEQ_OUT, 0, 0 };
         }
         if (!mask)
            break;
         if (packet == FP_MAX_PACKETS)
            SEQ_FAIL("phase 1: out of packets at EXEC %d\n", e);
         packet++;
         x->num_packets++;
         progress = true;
      }

      /* nothing at all happened: the limit or a missing rule blocks it
       * (more rows: an SFU may wait that many EXECs for its operand) */
      if (!progress && ++idle > c->seq_rows) {
         if (grate_debug & GRATE_DEBUG_TGSI) {
            fprintf(stderr, "GRATE FRAG: phase 1: stuck at EXEC %d, live %d:", e,
                    seq_live_items(c, owner, e));
            for (int i = 0; i < c->num_units; i++)
               if (c->units[i].live && c->units[i].exec < 0)
                  fprintf(stderr, " u%d%s", i, c->units[i].sfu ? "(sfu)" : "");
            for (int n = 0; n < c->num_tex; n++)
               if (c->tex[n].live && c->tex[n].exec < 0)
                  fprintf(stderr, " tex%d", n);
            fprintf(stderr, "\n");
            seq_dump(c, s, owner);
         }
         return false;
      }
      if (progress)
         idle = 0;
      s->num_execs = e + 1;
   }
   s->num_packets = packet;

   /* EXECs in whole rounds of the rows (other counts hang the channel,
    * asmtri exp-rows3): empty ones to fill up */
   while (remaining == 0 && s->num_execs % c->seq_rows) {
      if (s->num_execs == FP_SEQ_MAX_EXECS)
         SEQ_FAIL("phase 1: out of EXECs filling the rows\n");
      struct fp_seq_exec *x = &c->seq_exec[s->num_execs++];
      memset(x, 0, sizeof(*x));
      x->sfu_unit = x->tex = -1;
      for (int k = 0; k < 4; k++)
         x->ipl_input[k] = -1;
      x->first_packet = packet;
   }

   /* more rows: each EXEC without a packet gets an empty one (emission),
    * and they count against the program's packets too */
   int total = packet;
   for (int e = 0; c->seq_rows > 1 && e < s->num_execs; e++)
      total += !c->seq_exec[e].num_packets && seq_needs_alu(c, e);
   if (total > FP_MAX_PACKETS)
      SEQ_FAIL("phase 1: %d packets with the empty ones (%d EXECs, %d packets of work)\n",
               total, s->num_execs, packet);
   return true;
}

/*
 * Phase 2: register items. Values an SFU works on in place share one item
 * (the chain of copy, pre-op and op); an item lives from the stage that
 * writes it to the last stage reading it. Texels, the destination colour and
 * the colour output are fixed10 halves, one item each, so that a half freed
 * by its last read can take a fixed10 value or a colour channel.
 */
struct seq_item {
   int def, last;
   int reg;               /* fixed register or -1 */
   int half;              /* fixed half of reg, -1: the whole register */
   int row;               /* the row an SFU works on it in, -1: any */
   bool fx10;             /* a fixed10 value: any one half */
};

static int
seq_alu_time(struct fp_compiler *c, struct fp_unit *u, bool write)
{
   return SEQ_T(u->exec, T_ALU) + 2 * (u->packet - c->seq_exec[u->exec].first_packet) + write;
}

static int
seq_read_time(struct fp_compiler *c, struct fp_unit *r)
{
   return seq_is_alu(r) ? seq_alu_time(c, r, false) : SEQ_T(r->exec, T_MFU);
}

static bool
seq_overlap(const struct seq_item *a, const struct seq_item *b)
{
   return a->def <= b->last && b->def <= a->last;
}

/* Whether a fixed item other than "self" needs register r (half h, -1: any)
 * while item "it" lives */
static bool
seq_fixed_clash(const struct seq_item *items, int num, int self,
                const struct seq_item *it, int r, int h)
{
   for (int f = 0; f < num; f++) {
      const struct seq_item *x = &items[f];
      if (f != self && x->reg == r && x->last >= x->def &&
          (h < 0 || x->half < 0 || x->half == h) && seq_overlap(x, it))
         return true;
   }
   return false;
}

/* Whether register r (half h, -1: both) holds an assigned item while "it"
 * lives */
static bool
seq_busy(const struct seq_item *items, int num, const int *assigned,
         const int *assigned_half, const struct seq_item *it, int r, int h)
{
   for (int k = 0; k < num; k++)
      if (assigned[k] == r && (h < 0 || assigned_half[k] < 0 || assigned_half[k] == h) &&
          seq_overlap(&items[k], it))
         return true;
   return false;
}

/* Whether an ALU unit of EXEC e reads the value or varying component o */
static bool
seq_read_in_exec(struct fp_compiler *c, const struct fp_opnd *o, int e)
{
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *r = &c->units[i];
      if (!r->live || r->exec != e || !seq_is_alu(r))
         continue;
      fp_unit_foreach_src(r, k, j) {
         const struct fp_opnd *x = &r->slots[k].src[j];
         if (x->kind == o->kind && x->index == o->index &&
             (o->kind != FP_OPND_INPUT || x->comp == o->comp))
            return true;
      }
   }
   return false;
}

/* The register items of phase 2 and their registers: on the heap (with
 * FP_MAX_UNITS of them, too much for a thread's stack) */
#define SEQ_MAX_ITEMS (FP_MAX_UNITS + FP_MAX_INPUTS * 4 + 4 * FP_MAX_TEXTURES + 8)
struct seq_alloc_mem {
   struct seq_item items[SEQ_MAX_ITEMS];
   int order[SEQ_MAX_ITEMS];
   int assigned[SEQ_MAX_ITEMS], assigned_half[SEQ_MAX_ITEMS];
   int value_item[FP_MAX_UNITS];
};

static bool
seq_alloc(struct fp_compiler *c, struct seq_state *s)
{
   struct seq_alloc_mem *m = c->seq_mem;
   struct seq_item *items = m->items;
   int *value_item = m->value_item;
   int input_item[FP_MAX_INPUTS][4];
   int texel_item[FP_MAX_TEXTURES][4], dst_item[4], out_item[4];
   int num = 0;

   memset(value_item, -1, sizeof(m->value_item));
   memset(input_item, -1, sizeof(input_item));
   memset(texel_item, -1, sizeof(texel_item));
   memset(dst_item, -1, sizeof(dst_item));
   memset(out_item, -1, sizeof(out_item));

   /* values that need a register: read in a later EXEC, by an SFU or TEX */
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || u->output_chan >= 0 || u->cr)
         continue;
      if (!seq_is_alu(u))
         continue;   /* in place: below, once the inputs have items */
      items[num] = (struct seq_item) { .def = seq_alu_time(c, u, true), .last = -1, .reg = -1, .half = -1, .row = -1, .fx10 = u->fx10 };
      value_item[i] = num++;
   }
#define SEQ_MFU_DEF(e, stage) (c->seq_rows > 1 && (e) > 0 ? SEQ_T((e) - 1, T_ALU) : SEQ_T(e, stage))
   /*
    * The MFU of an EXEC writes before the ALU of the one before reads
    * (asmtri exp-late h*: an ipl in EXEC k + 1 over a register the ALU of
    * EXEC k reads gives it the new value; from k + 2 on the old one): with
    * more rows, a component fetched in EXEC e owns its register from that
    * ALU on.
    */
   for (int i = 0; i < FP_MAX_INPUTS; i++)
      for (int k = 0; k < 4; k++)
         if (s->fetch_exec[i][k] >= 0) {
            int e = s->fetch_exec[i][k];
            int def = SEQ_MFU_DEF(e, T_MFU);
            items[num] = (struct seq_item) { .def = def, .last = -1, .reg = s->fetch_reg[i][k], .half = -1, .row = -1 };
            input_item[i][k] = num++;
         }
   /* an SFU works in place: the item of its operand, a value or a varying
    * component it was folded onto (fp_fold_sfu_copies) */
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || u->output_chan >= 0 || seq_is_alu(u))
         continue;
      const struct fp_opnd *x = &u->slots[0].src[0];
      value_item[i] = x->kind == FP_OPND_VALUE ? value_item[x->index] :
                      x->kind == FP_OPND_INPUT ? input_item[x->index][x->comp] : -1;
      /* the SFU register is row-relative: the whole chain on its row */
      int it = value_item[i], row = SEQ_ROW(c, u->exec);
      if (it < 0 || c->seq_rows == 1)
         continue;
      if ((items[it].row >= 0 && items[it].row != row) ||
          (items[it].reg >= 0 && items[it].reg / 4 != row))
         SEQ_FAIL("phase 2: SFU u%d on row %d, its operand on row %d\n", i, row,
                  items[it].row >= 0 ? items[it].row : items[it].reg / 4);
      items[it].row = row;
   }

   /*
    * The TEX and the PSEQ load write all four halves of their pair, read
    * or not, also before the ALU of the EXEC before reads (with more rows,
    * as the ipl: asmtri exp-late w*, x*). Except where the fetch's own r or
    * bias is (the texel pair): the TEX reads that first, so the texel
    * starts there with the TEX, and no ALU of the EXEC before may read that
    * coordinate (the texel would be there already).
    */
   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live)
         continue;
      int e = c->tex[n].exec;
      int q = 1 - c->tex[n].pair, t = SEQ_T(e, T_TEX);
      int base = 4 * SEQ_ROW(c, e);
      bool own[2] = { false, false };
      for (int k = 0; k < c->tex[n].num_coords; k++) {
         int role = c->tex[n].role[k];
         if (role < FP_TEX_R)
            continue;
         own[role - FP_TEX_R] = true;
         if (c->seq_rows > 1 && e > 0 && seq_read_in_exec(c, &c->tex[n].coord[k], e - 1))
            SEQ_FAIL("phase 2: r/bias of tex%d read by the ALU of EXEC %d\n", n, e - 1);
      }
      for (int h = 0; h < 4; h++) {
         int def = own[h / 2] ? t : SEQ_MFU_DEF(e, T_TEX);
         items[num] = (struct seq_item) { .def = def, .last = t, .reg = base + 2 * q + h / 2, .half = h % 2, .row = -1 };
         texel_item[n][h] = num++;
      }
   }
   if (c->dst_used) {
      int q = c->dst_pair, t = SEQ_T(s->dst_exec, T_PSEQ);
      int def = SEQ_MFU_DEF(s->dst_exec, T_PSEQ);
      int base = 4 * SEQ_ROW(c, s->dst_exec);
      for (int h = 0; h < 4; h++) {
         items[num] = (struct seq_item) { .def = def, .last = t, .reg = base + 2 * q + h / 2, .half = h % 2, .row = -1 };
         dst_item[h] = num++;
      }
   }

   /* last reads */
#define SEQ_READ(it, t) do { if ((it) >= 0) items[it].last = MAX2(items[it].last, t); } while (0)
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *r = &c->units[i];
      if (!r->live)
         continue;
      int t = seq_read_time(c, r);
      fp_unit_foreach_src(r, k, j) {
         struct fp_opnd *o = &r->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE) {
            struct fp_unit *d = &c->units[o->index];
            /* forwarded to the next packet: no register read */
            if (seq_is_alu(d) && seq_is_alu(r) && d->exec == r->exec)
               continue;
            SEQ_READ(value_item[o->index], t);
         } else if (o->kind == FP_OPND_INPUT)
            SEQ_READ(input_item[o->index][o->comp], t);
         else if (o->kind == FP_OPND_TEXEL)
            SEQ_READ(texel_item[o->index][fp_output_half(o->comp)], t);
         else if (o->kind == FP_OPND_DST)
            SEQ_READ(dst_item[fp_output_half(o->comp)], t);
      }
   }
   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live)
         continue;
      int p = c->tex[n].pair;
      for (int k = 0; k < c->tex[n].num_coords; k++) {
         struct fp_opnd *o = &c->tex[n].coord[k];
         int it = o->kind == FP_OPND_VALUE ? value_item[o->index] :
                  input_item[o->index][o->comp];
         if (it < 0)
            SEQ_FAIL("phase 2: coordinate %d of tex%d not in a register\n", k, n);
         /* the coordinates are read from pair p: s in 2p, t in 2p+1, as
          * fp20 values; a cube map's r (by the MFU) and a bias from the
          * texel pair, 2q and 2q+1, before the texel lands there */
         int role = c->tex[n].role[k];
         int want = 4 * SEQ_ROW(c, c->tex[n].exec) +
                    (role < FP_TEX_R ? 2 * p + role : 2 * (1 - p) + (role - FP_TEX_R));
         SEQ_READ(it, SEQ_T(c->tex[n].exec, T_TEX) - (role >= FP_TEX_R));
         if (items[it].reg >= 0 && items[it].reg != want)
            SEQ_FAIL("phase 2: coordinate %d of tex%d (%s%d) in r%d, wanted in r%d\n", k, n,
                     o->kind == FP_OPND_VALUE ? "u" : "in", o->index, items[it].reg, want);
         items[it].reg = want;
         items[it].fx10 = false;
      }
   }
#undef SEQ_READ
#undef SEQ_MFU_DEF

   /* colour channels own their half from the first write on */
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || u->output_chan < 0)
         continue;
      int h = fp_output_half(u->output_chan);
      int t = seq_alu_time(c, u, true);
      if (out_item[h] < 0) {
         items[num] = (struct seq_item) { .def = t, .last = SEQ_FOREVER, .reg = SEQ_OUT_REG(c, h), .half = h % 2, .row = -1 };
         out_item[h] = num++;
      }
      items[out_item[h]].def = MIN2(items[out_item[h]].def, t);
   }

   /* items without readers (results only forwarded) take no register */
   for (int i = 0; i < num; i++)
      if (items[i].last < 0)
         items[i].last = items[i].def - 1;

   /* greedy in order of definition, fixed registers first at equal times;
    * more rows: the items bound to a row (SFU operands) before the free
    * ones, which fit around them */
   int *order = m->order;
   for (int i = 0; i < num; i++)
      order[i] = i;
#define SEQ_FREE(it) (c->seq_rows > 1 && (it)->reg < 0 && (it)->row < 0)
   for (int i = 1; i < num; i++)
      for (int k = i; k > 0; k--) {
         struct seq_item *a = &items[order[k - 1]], *b = &items[order[k]];
         if (SEQ_FREE(a) != SEQ_FREE(b) ? SEQ_FREE(b) :
             a->def < b->def || (a->def == b->def && (a->reg >= 0 || b->reg < 0)))
            break;
         int tmp = order[k]; order[k] = order[k - 1]; order[k - 1] = tmp;
      }

   int *assigned = m->assigned, *assigned_half = m->assigned_half;
   memset(assigned, -1, sizeof(m->assigned));
   memset(assigned_half, -1, sizeof(m->assigned_half));
   for (int oi = 0; oi < num; oi++) {
      int i = order[oi];
      struct seq_item *it = &items[i];
      if (it->last < it->def)
         continue;

      int r = it->reg, h = it->half;
      if (r < 0) {
         /* a free register, or half for a fixed10 value, that no fixed
          * item needs meanwhile */
         for (r = 0; r < SEQ_REGS(c); r++) {
            if (it->row >= 0 && r / 4 != it->row)
               continue;
            if (it->fx10) {
               for (h = 0; h < 2; h++)
                  if (!seq_busy(items, num, assigned, assigned_half, it, r, h) &&
                      !seq_fixed_clash(items, num, i, it, r, h))
                     break;
               if (h < 2)
                  break;
            } else if (!seq_busy(items, num, assigned, assigned_half, it, r, -1) &&
                       !seq_fixed_clash(items, num, i, it, r, -1))
               break;
         }
         if (r == SEQ_REGS(c)) {
            if (grate_debug & GRATE_DEBUG_TGSI)
               fprintf(stderr, "GRATE FRAG: phase 2: no register for item %d [%d, %d]%s row %d\n",
                       i, it->def, it->last, it->fx10 ? " (fixed10)" : "", it->row);
            if (grate_debug & GRATE_DEBUG_TGSI) {
               for (int k = 0; k < num; k++)
                  if (assigned[k] >= 0 && items[k].last >= it->def && items[k].def <= it->last)
                     fprintf(stderr, "GRATE FRAG:   r%d%s item %d [%d, %d]\n", assigned[k],
                             assigned_half[k] < 0 ? "" : assigned_half[k] ? ".h" : ".l",
                             k, items[k].def, items[k].last);
            }
            return false;
         }
         if (!it->fx10)
            h = -1;
      } else if (seq_busy(items, num, assigned, assigned_half, it, r, h)) {
         if (grate_debug & GRATE_DEBUG_TGSI)
            fprintf(stderr, "GRATE FRAG: phase 2: fixed r%d%s for item %d [%d, %d] busy\n",
                    r, h < 0 ? "" : h ? ".h" : ".l", i, it->def, it->last);
         return false;
      }
      assigned[i] = r;
      assigned_half[i] = h;
   }

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      bool held = u->live && value_item[i] >= 0;
      u->reg = held ? assigned[value_item[i]] : -1;
      u->half = held ? assigned_half[value_item[i]] : -1;
   }

   /* condition registers: from the write to the last CSEL reading it,
    * greedy in the order of the writes */
   int cr_last[16], cr_def[FP_MAX_UNITS], cr_end[FP_MAX_UNITS];
   for (int k = 0; k < 16; k++)
      cr_last[k] = -1;
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || !u->cr)
         continue;
      cr_def[i] = seq_alu_time(c, u, true);
      cr_end[i] = cr_def[i];
      for (int r = 0; r < c->num_units; r++) {
         struct fp_unit *x = &c->units[r];
         if (!x->live)
            continue;
         fp_unit_foreach_src(x, k, j)
            if (x->slots[k].src[j].kind == FP_OPND_VALUE && x->slots[k].src[j].index == i)
               cr_end[i] = MAX2(cr_end[i], seq_read_time(c, x));
      }
   }
   for (;;) {
      int next = -1;
      for (int i = 0; i < c->num_units; i++)
         if (c->units[i].live && c->units[i].cr && c->units[i].reg < 0 &&
             (next < 0 || cr_def[i] < cr_def[next]))
            next = i;
      if (next < 0)
         break;
      int k;
      for (k = 0; k < seq_cr_count(c) && cr_last[k] >= cr_def[next]; k++)
         ;
      if (k == seq_cr_count(c))
         SEQ_FAIL("phase 2: no condition register for u%d\n", next);
      cr_last[k] = cr_end[next];
      c->units[next].reg = 64 + k / 2;
      c->units[next].half = k % 2;
   }
   return true;
}

static bool
seq_try(struct fp_compiler *c, int step)
{
   struct seq_state s;

   /* program order first: channel by channel takes more EXECs; more rows:
    * varyings in the first round first, then fetched late */
   for (int late = 0; late < (c->seq_rows > 1 ? 2 : 1); late++)
   for (int by_channel = 0; by_channel < 2; by_channel++)
   for (c->out_pair = 1; c->out_pair >= 0; c->out_pair--)
   for (int limit = 2 * SEQ_REGS(c); limit > 0; limit -= step) {
      s.by_channel = by_channel;
      s.late_ipl = late;
      if (!seq_schedule(c, &s, limit)) {
         if (grate_debug & GRATE_DEBUG_TGSI)
            fprintf(stderr, "GRATE FRAG: sequential schedule failed (%d rows, limit %d%s%s%s)\n",
                    c->seq_rows, limit, by_channel ? ", by channel" : "",
                    late ? ", late" : "", c->units[0].fx10 || step == 1 ? ", fx10" : "");
         continue;
      }
      if (grate_debug & GRATE_DEBUG_TGSI)
         fprintf(stderr, "GRATE FRAG: sequential schedule ok (%d rows, limit %d, "
                 "%d EXECs, colour in r%d/r%d), allocating\n", c->seq_rows, limit,
                 s.num_execs, SEQ_OUT_REG(c, 0), SEQ_OUT_REG(c, 2));
      if (seq_alloc(c, &s)) {
         c->num_execs = s.num_execs;
         c->num_packets = s.num_packets;
         memcpy(c->fetch_exec, s.fetch_exec, sizeof(s.fetch_exec));
         memcpy(c->fetch_reg, s.fetch_reg, sizeof(s.fetch_reg));
         c->dst_exec = s.dst_exec;
         c->dw.src_regs = c->out_pair ? FP_DW_REGS_R2_R3 : FP_DW_REGS_R0_R1;
         return true;
      }
      if (grate_debug & GRATE_DEBUG_TGSI)
         fprintf(stderr, "GRATE FRAG: sequential registers failed (limit %d)\n", limit);
   }
   return false;
}

/*
 * Rows of a sequential program (measured, gpu/tools/asmtri/README.md "Long
 * multi-row programs"): more than one gives 4 registers more each, at the
 * cost of pixels in flight. The TEX and the PSEQ load of EXEC e work on the
 * pairs of row e % rows, as in multi-row programs (fstest and dEQP texture,
 * blend and depth_stencil with every program on 1 and on 3 rows,
 * GRATE_FP_SEQ / GRATE_FP_SEQ_ROWS: the same pixels). So does the MFU
 * division of computed cube map and projective coordinates: its registers,
 * rcp r(8 + c) of a cube map's too, are row-relative (asmtri exp-cube).
 *
 * One row, fp20 values first; fixed10 halves for those that may (fx10.c)
 * only when the program does not fit otherwise (GRATE_FP_FX10=0 turns them
 * off); then more rows, the same way.
 */
/*
 * CSEL(c, 1, 0) or CSEL(c, 0, 1) (a comparison on a value fp_fold_conditions
 * could not take, as a varying): the same test of c as a tested MAD, which
 * can write a condition register
 */
static void
seq_csel_to_test(struct fp_compiler *c, struct fp_unit *u)
{
   struct fp_unit_slot *s = &u->slots[0];
   float x, y;

   if (!u->live || !seq_is_alu(u) || u->num_slots != 1 || s->op != FP_ALU_OP_CSEL ||
       u->condition || u->scale || u->output_chan >= 0 || u->kill ||
       !fp_opnd_known(c, &s->src[1], &x) || !fp_opnd_known(c, &s->src[2], &y) ||
       !((x == 1.0f && y == 0.0f) || (x == 0.0f && y == 1.0f)))
      return;
   struct fp_opnd cond = s->src[0];
   if (cond.minus_one || cond.twice)
      return;
   /* as fp_fold_conditions: c < 0 ? 1 : 0 is -c > 0, c < 0 ? 0 : 1 is
    * c >= 0, -|c| < 0 ? 0 : 1 is c == 0 */
   bool lt_one = x == 1.0f, negate;
   enum fp_condition test;
   if (cond.absolute) {
      if (!cond.negate || lt_one)
         return;
      negate = false;
      test = FP_CONDITION_EQUAL;
   } else {
      negate = cond.negate ^ lt_one;
      test = lt_one ? FP_CONDITION_GREATER : FP_CONDITION_GEQUAL;
   }
   cond.negate = negate;
   cond.absolute = false;
   fp_unit_op(s, FP_ALU_OP_MAD, cond, fp_opnd_one(c, false), fp_opnd_zero(c, false));
   u->condition = test;
}

/*
 * Comparisons a condition register may hold: a tested MAD (1.0 or 0.0) the
 * ALU reads. Narrow (wide = false): only CSELs read it, as their selector -c
 * or -|c|, as the NVIDIA compiler has them. Wide: any ALU operand (a cr
 * reads as 0.0 or 1.0, asmtri exp-imm-cr), and the CSELs choosing 1 or 0
 * become tested MADs first (the program does not fit otherwise when this
 * runs). Neither fits every program the other does
 * (random.all_features.fragment.88 narrow, atan2 vec4 wide). Returns how
 * many.
 */
static int
seq_mark_cr(struct fp_compiler *c, bool on, bool wide)
{
   int n = 0;
   for (int i = 0; on && wide && i < c->num_units; i++)
      seq_csel_to_test(c, &c->units[i]);
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      u->cr = false;
      if (!on || !u->live || !seq_is_alu(u) || u->num_slots != 1 || !u->condition ||
          u->output_chan >= 0 || u->kill || u->sfu_copy)
         continue;
      bool ok = true, read = false;
      for (int r = 0; r < c->num_units && ok; r++) {
         struct fp_unit *x = &c->units[r];
         if (!x->live)
            continue;
         fp_unit_foreach_src(x, k, j) {
            const struct fp_opnd *o = &x->slots[k].src[j];
            if (o->kind != FP_OPND_VALUE || o->index != i)
               continue;
            read = true;
            ok &= seq_is_alu(x) &&
                  (wide || (x->num_slots == 1 && x->slots[0].op == FP_ALU_OP_CSEL &&
                            k == 0 && j == 0 && o->negate && !o->twice && !o->minus_one));
         }
      }
      for (int t = 0; t < c->num_tex && ok; t++)
         for (int k = 0; k < c->tex[t].num_coords; k++)
            ok &= !(c->tex[t].coord[k].kind == FP_OPND_VALUE && c->tex[t].coord[k].index == i);
      if (ok && read) {
         u->cr = true;
         n++;
      }
   }
   return n;
}

static bool seq_sched_rows_pass(struct fp_compiler *c);

/*
 * The search as before; when nothing fits, again with the comparisons in
 * condition registers (GRATE_FP_CR=0 turns that off)
 */
static bool
seq_sched_rows(struct fp_compiler *c)
{
   if (seq_sched_rows_pass(c))
      return true;
   if (getenv("GRATE_FP_CR") && !atoi(getenv("GRATE_FP_CR")))
      return false;
   for (int wide = 0; wide < 2; wide++) {
      int n = seq_mark_cr(c, true, wide);
      if (grate_debug & GRATE_DEBUG_TGSI)
         fprintf(stderr, "GRATE FRAG: retrying with %d comparisons in condition registers%s\n",
                 n, wide ? " (wide)" : "");
      if (n && seq_sched_rows_pass(c))
         return true;
   }
   seq_mark_cr(c, false, false);
   return false;
}

static bool
seq_sched_rows_pass(struct fp_compiler *c)
{
   bool fx10_ok = !getenv("GRATE_FP_FX10") || atoi(getenv("GRATE_FP_FX10"));
   /* GRATE_FP_SEQ_ROWS: at least that many rows (to try them out) */
   int min_rows = getenv("GRATE_FP_SEQ_ROWS") ? CLAMP(atoi(getenv("GRATE_FP_SEQ_ROWS")), 1, 4) : 1;

   for (c->seq_rows = min_rows; c->seq_rows <= 4; c->seq_rows++) {
      for (int i = 0; i < c->num_units; i++)
         c->units[i].fx10 = false;
      if (seq_try(c, 2))
         return true;
      if (!fx10_ok)
         continue;

      fp_mark_fx10(c);
      int n = 0;
      for (int i = 0; i < c->num_units; i++)
         n += c->units[i].fx10;
      if (grate_debug & GRATE_DEBUG_TGSI)
         fprintf(stderr, "GRATE FRAG: retrying with %d fixed10 values\n", n);
      if (n && seq_try(c, 1))
         return true;
   }
   c->seq_rows = 1;
   return false;
}

bool
fp_sched_seq(struct fp_compiler *c)
{
   struct seq_readers *rd = MALLOC_STRUCT(seq_readers);
   struct seq_alloc_mem *mem = MALLOC_STRUCT(seq_alloc_mem);
   if (!rd || !mem) {
      FREE(rd);
      FREE(mem);
      return false;
   }
   c->seq_mem = mem;
   seq_build_readers(c, rd);
   c->seq_readers = rd;
   bool ok = seq_sched_rows(c);
   c->seq_readers = NULL;
   c->seq_mem = NULL;
   FREE(rd);
   FREE(mem);
   return ok;
}

/* Register an operand of a unit in EXEC e, packet p is read from */
static struct fp_alu_src_operand
seq_resolve(struct fp_compiler *c, const struct fp_opnd *o, int e, int p)
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
   case FP_OPND_SPECIAL:
      ret = fp_alu_src_special(o->index);
      break;
   case FP_OPND_INPUT:
      ret = fp_alu_src_row(c->fetch_reg[o->index][o->comp]);
      break;
   case FP_OPND_TEXEL:
   case FP_OPND_DST: {
      int pair = o->kind == FP_OPND_TEXEL ? 1 - c->tex[o->index].pair : c->dst_pair;
      int exec = o->kind == FP_OPND_TEXEL ? c->tex[o->index].exec : c->dst_exec;
      ret = fp_alu_src_row(4 * SEQ_ROW(c, exec) + 2 * pair + fp_output_half(o->comp) / 2);
      ret.datatype = FP_DATATYPE_FIXED10;
      ret.sub_reg_select_high = fp_output_half(o->comp) % 2;
      break;
   }
   case FP_OPND_VALUE: {
      struct fp_unit *d = &c->units[o->index];
      if (seq_is_alu(d) && d->exec == e && d->packet == p - 1)
         ret.index = FP_ALU_RESULT_REG(d->slot);
      else {
         assert(d->reg >= 0);
         ret.index = d->reg;
         if (d->half >= 0) {
            ret.datatype = FP_DATATYPE_FIXED10;
            ret.sub_reg_select_high = d->half;
         }
      }
      break;
   }
   }
   ret.negate = o->negate;
   ret.absolute_value = o->absolute;
   ret.scale_by_two = o->twice;
   ret.minus_one = o->minus_one;
   return ret;
}

static void
seq_emit_unit(struct fp_compiler *c, struct fp_unit *u,
              struct fp_alu_instr_packet *pkt)
{
   for (int k = 0; k < u->num_slots; k++) {
      struct fp_unit_slot *s = &u->slots[k];
      struct fp_alu_instr *a = &pkt->slots[u->slot + k];
      struct fp_alu_src_operand c2 = seq_resolve(c, &s->src[2], u->exec, u->packet);

      /* rD is left disabled: result = op(rA * rB, rC) */
      *a = (struct fp_alu_instr) {
         .op = s->op,
         .accumulate_this = s->accumulate_this,
         .accumulate_other = s->accumulate_other,
         .src = {
            seq_resolve(c, &s->src[0], u->exec, u->packet),
            seq_resolve(c, &s->src[1], u->exec, u->packet),
            c2, c2
         },
      };
      if (s->square_c) {
         /* rC * rD with rD = rC: the sign drops out of the square */
         a->src[2].negate = a->src[3].negate = false;
         a->rd_enable = true;
      }

      if (k > 0) {
         a->dst.index = 31;     /* DP partial product, added to slot 0 */
         continue;
      }

      a->scale = u->scale;
      a->condition = u->condition;
      a->dst.saturate = u->saturate;
      if (u->output_chan >= 0) {
         int o = fp_output_half(u->output_chan);
         a->dst.index = SEQ_OUT_REG(c, o);
         a->dst.write_low_sub_reg = (o % 2) == 0;
         a->dst.write_high_sub_reg = (o % 2) != 0;
      } else {
         a->dst.index = u->kill ? FP_KILL_REG : u->reg >= 0 ? u->reg : 31;
         a->dst.write_low_sub_reg = u->half != 1;
         a->dst.write_high_sub_reg = u->half != 0;
      }
   }
}

void
fp_emit_seq(struct fp_compiler *c)
{
   struct grate_fp_shader *fp = c->fp;

   /* more rows: the weights are set up in EXEC 0 for the first round (the
    * later interpolating EXECs carry them over) and where seq_fetch_inputs
    * asked for them again (late_ipl) */
   bool any_ipl = false;
   for (int e = 0; e < c->num_execs; e++)
      for (int k = 0; k < 4; k++)
         any_ipl |= c->seq_exec[e].ipl_input[k] >= 0;

   for (int e = 0; e < c->num_execs; e++) {
      struct fp_seq_exec *x = &c->seq_exec[e];
      struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
      list_inithead(&inst->link);

      if (x->pseq)
         inst->pseq = fp_pseq_word(c, c->dst_pair);

      bool ipl = false;
      for (int k = 0; k < 4; k++)
         ipl |= x->ipl_input[k] >= 0;
      bool bary = c->seq_rows == 1 ? ipl : (e == 0 && any_ipl) || x->bary;
      if (ipl || bary || x->sfu_unit >= 0) {
         struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
         list_inithead(&mfu->link);
         if (bary) {
            /* one row: every interpolating EXEC sets up the weights (as
             * the NVIDIA compiler does in one-row programs) */
            fp_mfu_bary_setup(mfu);
         }
         if (ipl || bary) {
            for (int k = 0; k < 4; k++) {
               if (x->ipl_input[k] < 0)
                  continue;
               mfu->var[k].op = FP_VAR_OP_FP20;
               mfu->var[k].tram_row = fp_tram_row(c, x->ipl_input[k],
                                                  x->ipl_comp[k], k);
            }
         } else {
            struct fp_unit *u = &c->units[x->sfu_unit];
            mfu->sfu.op = u->sfu;
            mfu->sfu.reg = u->reg - 4 * SEQ_ROW(c, e);   /* row-relative */
         }
         inst->mfu_sched.num_instructions = 1;
         inst->mfu_sched.address = list_length(&fp->mfu_instructions);
         list_addtail(&mfu->link, &fp->mfu_instructions);
      }

      if (x->cube_mfu) {
         /*
          * Cube or projective coordinates computed by the ALU, as the
          * NVIDIA compiler does: rcp of register 8 + R's column (the major
          * axis) or of q into that register, then S and T scaled by it.
          */
         int p = c->tex[x->tex].pair, r = 2 * (1 - p);
         struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
         list_inithead(&mfu->link);
         mfu->sfu.op = FP_SFU_OP_RCP;
         mfu->sfu.reg = c->tex[x->tex].cube ? 8 + r : r;
         for (int k = 0; k < 2; k++) {
            mfu->mul[k].dst = FP_MFU_MUL_DST_ROW_REG_0 + 2 * p + k;
            mfu->mul[k].src[0] = FP_MFU_MUL_SRC_ROW_REG_0 + r;
            mfu->mul[k].src[1] = FP_MFU_MUL_SRC_ROW_REG_0 + 2 * p + k;
         }
         inst->mfu_sched.num_instructions = 1;
         inst->mfu_sched.address = list_length(&fp->mfu_instructions);
         list_addtail(&mfu->link, &fp->mfu_instructions);
      }

      if (x->tex >= 0) {
         inst->tex = fp_tex_word(c, x->tex, c->tex[x->tex].pair);
         fp->info.sampler_mask |= 1u << c->tex[x->tex].sampler;
      }

      inst->alu_sched.address = list_length(&fp->alu_instructions);
      /* more rows: an empty packet where one is needed (seq_needs_alu) */
      if (!x->num_packets && c->seq_rows > 1 && (grate_debug & GRATE_DEBUG_TGSI)) {
         bool pad = seq_needs_alu(c, e);
         fprintf(stderr, "GRATE FRAG: EXEC %d without ALU: %s\n", e,
                 !pad ? "no packet" : seq_pad_why == 0 ? "empty packet (row 0)" :
                 seq_pad_why == 1 ? "empty packet (ALU read across)" : "empty packet (TEX read across)");
      }
      if (!x->num_packets && seq_needs_alu(c, e)) {
         struct fp_alu_instr_packet *pkt = CALLOC_STRUCT(fp_alu_instr_packet);
         list_inithead(&pkt->link);
         for (int k = 0; k < 4; k++)
            pkt->slots[k] = fp_alu_nop();
         list_addtail(&pkt->link, &fp->alu_instructions);
         inst->alu_sched.num_instructions++;
      }
      for (int p = x->first_packet; p < x->first_packet + x->num_packets; p++) {
         struct fp_alu_instr_packet *pkt = CALLOC_STRUCT(fp_alu_instr_packet);
         list_inithead(&pkt->link);
         for (int k = 0; k < 4; k++)
            pkt->slots[k] = fp_alu_nop();
         for (int i = 0; i < c->num_units; i++) {
            struct fp_unit *u = &c->units[i];
            if (u->live && seq_is_alu(u) && u->packet == p)
               seq_emit_unit(c, u, pkt);
         }
         list_addtail(&pkt->link, &fp->alu_instructions);
         inst->alu_sched.num_instructions++;
      }
      /* none: address 0, as the NVIDIA compiler has it (after the 64th
       * packet the next address would not fit in the 6 bits) */
      if (!inst->alu_sched.num_instructions)
         inst->alu_sched.address = 0;

      /* the store runs in the last pass */
      if (e == c->num_execs - 1)
         inst->dw = c->dw;

      list_addtail(&inst->link, &fp->fp_instructions);
   }

   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "GRATE FRAG: sequential, %d rows, %d EXECs:", c->seq_rows,
              c->num_execs);
      for (int e = 0; e < c->num_execs; e++) {
         struct fp_seq_exec *x = &c->seq_exec[e];
         fprintf(stderr, " [e%d%s%s%s %d alu]", e,
                 x->sfu_unit >= 0 ? " sfu" : "", x->tex >= 0 ? " tex" : "",
                 x->pseq ? " pseq" : "", x->num_packets);
      }
      fprintf(stderr, "\n");
   }
}
