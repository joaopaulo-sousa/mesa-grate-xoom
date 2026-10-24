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

#define SEQ_REGS 4
#define SEQ_HALVES (2 * SEQ_REGS)
#define T_PSEQ 0          /* stages of an EXEC, as times 8 * (4 * exec + stage) */
#define T_MFU 1
#define T_TEX 2
#define T_ALU 3           /* ... + 2 * packet in the EXEC, + 1 for its writes */
#define SEQ_T(e, stage) (8 * (4 * (e) + (stage)))
#define SEQ_FOREVER (1 << 20)

struct seq_state {
   int fetch_exec[FP_MAX_INPUTS][4];   /* EXEC interpolating input i comp c */
   int fetch_reg[FP_MAX_INPUTS][4];    /* ... into this register */
   int num_execs;
   int num_packets;
   int dst_exec;
};

static bool
seq_is_alu(const struct fp_unit *u)
{
   return u->sfu == FP_SFU_OP_NOP;
}

/* Whether operand o can be read by an ALU unit in EXEC e, packet p */
static bool
seq_alu_can_read(struct fp_compiler *c, struct seq_state *s,
                 const struct fp_opnd *o, int e, int p)
{
   switch (o->kind) {
   case FP_OPND_VALUE: {
      struct fp_unit *d = &c->units[o->index];
      if (d->exec < 0)
         return false;
      if (!seq_is_alu(d))
         return d->exec <= e;
      return d->exec < e ||
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
   if (o->kind == FP_OPND_INPUT) {
      int f = s->fetch_exec[o->index][o->comp];
      return f >= 0 && f <= e;
   }
   if (o->kind != FP_OPND_VALUE)
      return false;
   struct fp_unit *d = &c->units[o->index];
   if (d->exec < 0)
      return false;
   if (!seq_is_alu(d) && same_exec_sfu)
      return d->exec <= e;
   return d->exec < e;
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
   return kind == SEQ_VALUE && c->units[a].fx10 ? 1 : 2;
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

/* Whether an item is still read in EXEC e or later (or not yet scheduled);
 * "except": a unit not to count, -2: no colour write counts */
static bool
seq_item_needed(struct fp_compiler *c, enum seq_item_kind kind, int a, int b,
                int e, int except)
{
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
      for (int k = 0; k < 2; k++)
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
         if (2 * pair + h / 2 == reg && h % 2 == high)
            return true;
      }
   }
   return false;
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

/* Register halves live after EXEC e */
static int
seq_live_items(struct fp_compiler *c, const struct seq_owner *owner, int e)
{
   int n = 0;

   for (int r = 0; r < SEQ_REGS; r++) {
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
      for (int k = 0; k < 2; k++)
         if (c->tex[n].live && seq_opnd_is(&c->tex[n].coord[k], SEQ_INPUT, i, comp))
            return true;
   return false;
}

/* A register the MFU of this EXEC can interpolate input i component comp
 * into, -1 if none: r(comp) preferably, the column the linker would give it
 * anyway, else any free one (the TRAM layout follows, fp/linker.c) */
static int
seq_fetch_reg(struct fp_compiler *c, const struct seq_owner *owner,
              const struct fp_seq_exec *x, int i, int comp, int e)
{
   bool coord = seq_is_coord(c, i, comp);

   for (int k = 0; k < SEQ_REGS; k++) {
      int r = (comp + k) % SEQ_REGS;
      if (coord && r != comp)
         break;
      if (x->ipl_input[r] < 0 && seq_fixed_free(c, &owner[r], e))
         return r;
   }
   return -1;
}

/* MFU interpolation: the varying components of the next units in program
 * order whose other operands are there, each into a free register */
static bool
seq_fetch_inputs(struct fp_compiler *c, struct seq_state *s,
                 struct seq_owner *owner, struct fp_seq_exec *x, int e,
                 int limit)
{
   bool any = false;
   int live = seq_live_items(c, owner, e - 1);

   for (int i = 0; i < c->num_units + c->num_tex; i++) {
      struct fp_opnd *ops[12];
      int n = 0;

      if (i < c->num_units) {
         struct fp_unit *u = &c->units[i];
         if (!u->live || u->exec >= 0 || !seq_others_ready(c, s, u, e, FP_OPND_INPUT))
            continue;
         /* colour writes wait for the rest (seq_unit_ready): their
          * varyings would only take registers meanwhile */
         if (u->output_chan >= 0 && !c->redirect_output) {
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
         ops[n++] = &c->tex[t].coord[0];
         ops[n++] = &c->tex[t].coord[1];
      }

      for (int k = 0; k < n; k++) {
         struct fp_opnd *o = ops[k];
         int comp = o->comp;
         if (o->kind != FP_OPND_INPUT || s->fetch_exec[o->index][comp] >= 0 ||
             live + 2 > limit)
            continue;
         int r = seq_fetch_reg(c, owner, x, o->index, comp, e);
         if (r < 0)
            continue;
         x->ipl_input[r] = o->index;
         x->ipl_comp[r] = comp;
         s->fetch_exec[o->index][comp] = e;
         s->fetch_reg[o->index][comp] = r;
         owner[r] = (struct seq_owner) { SEQ_INPUT, o->index, comp };
         live += 2;
         any = true;
      }
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
    * worth more as temporaries until then */
   if (!c->redirect_output)
      for (int i = 0; i < c->num_units; i++)
         if (c->units[i].live && c->units[i].output_chan < 0 && c->units[i].exec < 0)
            return false;

   int half = fp_output_half(u->output_chan);
   int reg = 2 * c->out_pair + half / 2;
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
         if (o->kind == FP_OPND_INPUT && o->comp == reg &&
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

   if (p + 1 >= FP_MAX_PACKETS_PER_EXEC || c->units[i].output_chan >= 0)
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
      for (int k = 0; k < 2; k++)
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

/* Phase 1 */
static bool
seq_schedule(struct fp_compiler *c, struct seq_state *s, int limit)
{
   struct seq_owner owner[SEQ_REGS] = { { SEQ_NONE, 0, 0 } };
   int remaining = 0, packet = 0, idle = 0;

   memset(s->fetch_exec, -1, sizeof(s->fetch_exec));
   memset(s->fetch_reg, -1, sizeof(s->fetch_reg));
   s->dst_exec = -1;
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
         return false;
      memset(x, 0, sizeof(*x));
      x->sfu_unit = x->tex = -1;
      for (int k = 0; k < 4; k++)
         x->ipl_input[k] = -1;
      x->first_packet = packet;

      /* MFU: an SFU op whose operand is in its register, else varyings */
      for (int i = 0; i < c->num_units; i++) {
         struct fp_unit *u = &c->units[i];
         if (!u->live || seq_is_alu(u) || u->exec >= 0 ||
             !seq_reg_ready(c, s, &u->slots[0].src[0], e, false))
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
            for (int p = 0; p < 2; p++) {
               if (!seq_fixed_free(c, &owner[2 * p], e) ||
                   !seq_fixed_free(c, &owner[2 * p + 1], e))
                  continue;
               s->dst_exec = e;
               c->dst_pair = p;
               owner[2 * p] = owner[2 * p + 1] = (struct seq_owner) { SEQ_DST, 0, 0 };
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
             seq_live_items(c, owner, e - 1) + 4 > limit)
            continue;
         for (int p = 0; p < 2; p++) {
            struct fp_opnd *s0 = &c->tex[n].coord[0], *s1 = &c->tex[n].coord[1];
            if ((s0->kind == FP_OPND_INPUT && s0->comp != 2 * p) ||
                (s1->kind == FP_OPND_INPUT && s1->comp != 2 * p + 1))
               continue;
            int q = 1 - p;
            if (!seq_fixed_free(c, &owner[2 * q], e) ||
                !seq_fixed_free(c, &owner[2 * q + 1], e))
               continue;
            /* computed coordinates go to pair p: no fixed item may be there */
            if ((s0->kind == FP_OPND_VALUE && owner[2 * p].kind != SEQ_NONE &&
                 !seq_fixed_free(c, &owner[2 * p], e)) ||
                (s1->kind == FP_OPND_VALUE && owner[2 * p + 1].kind != SEQ_NONE &&
                 !seq_fixed_free(c, &owner[2 * p + 1], e)))
               continue;
            c->tex[n].exec = e;
            c->tex[n].pair = p;
            owner[2 * q] = owner[2 * q + 1] = (struct seq_owner) { SEQ_TEXEL, n, 0 };
            x->tex = n;
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
               int delta = seq_unit_delta(c, i, e);
               int room = limit - (reserve && s->dst_exec < 0 ? 4 : 0);
               if (delta > 0 && seq_live_items(c, owner, e) + delta > room &&
                   !seq_forwardable(c, s, i, e, packet, p))
                  continue;
               if (best < 0 || delta < best_delta) {
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
               owner[2 * c->out_pair + fp_output_half(u->output_chan) / 2] =
                  (struct seq_owner) { SEQ_OUT, 0, 0 };
         }
         if (!mask)
            break;
         if (packet == FP_MAX_PACKETS)
            return false;
         packet++;
         x->num_packets++;
         progress = true;
      }

      /* nothing at all happened: the limit or a missing rule blocks it */
      if (!progress && ++idle > 1) {
         if (grate_debug & GRATE_DEBUG_TGSI) {
            fprintf(stderr, "GRATE FRAG: stuck at EXEC %d, live %d:", e,
                    seq_live_items(c, owner, e));
            for (int i = 0; i < c->num_units; i++)
               if (c->units[i].live && c->units[i].exec < 0)
                  fprintf(stderr, " u%d%s", i, c->units[i].sfu ? "(sfu)" : "");
            for (int n = 0; n < c->num_tex; n++)
               if (c->tex[n].live && c->tex[n].exec < 0)
                  fprintf(stderr, " tex%d", n);
            fprintf(stderr, "\n");
         }
         return false;
      }
      if (progress)
         idle = 0;
      s->num_execs = e + 1;
   }
   s->num_packets = packet;
   return remaining == 0;
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

static bool
seq_alloc(struct fp_compiler *c, struct seq_state *s)
{
   struct seq_item items[FP_MAX_UNITS + FP_MAX_INPUTS * 4 + 4 * FP_MAX_TEXTURES + 8];
   int value_item[FP_MAX_UNITS], input_item[FP_MAX_INPUTS][4];
   int texel_item[FP_MAX_TEXTURES][4], dst_item[4], out_item[4];
   int num = 0;

   memset(value_item, -1, sizeof(value_item));
   memset(input_item, -1, sizeof(input_item));
   memset(texel_item, -1, sizeof(texel_item));
   memset(dst_item, -1, sizeof(dst_item));
   memset(out_item, -1, sizeof(out_item));

   /* values that need a register: read in a later EXEC, by an SFU or TEX */
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || u->output_chan >= 0)
         continue;
      if (!seq_is_alu(u)) {
         /* in place: the item of the operand */
         value_item[i] = value_item[u->slots[0].src[0].index];
         continue;
      }
      items[num] = (struct seq_item) { .def = seq_alu_time(c, u, true), .last = -1, .reg = -1, .half = -1, .fx10 = u->fx10 };
      value_item[i] = num++;
   }
   for (int i = 0; i < FP_MAX_INPUTS; i++)
      for (int k = 0; k < 4; k++)
         if (s->fetch_exec[i][k] >= 0) {
            items[num] = (struct seq_item) { .def = SEQ_T(s->fetch_exec[i][k], T_MFU), .last = -1, .reg = s->fetch_reg[i][k], .half = -1 };
            input_item[i][k] = num++;
         }
   /* the TEX and the PSEQ load write all four halves of their pair, read
    * or not */
   for (int n = 0; n < c->num_tex; n++) {
      if (!c->tex[n].live)
         continue;
      int q = 1 - c->tex[n].pair, t = SEQ_T(c->tex[n].exec, T_TEX);
      for (int h = 0; h < 4; h++) {
         items[num] = (struct seq_item) { .def = t, .last = t, .reg = 2 * q + h / 2, .half = h % 2 };
         texel_item[n][h] = num++;
      }
   }
   if (c->dst_used) {
      int q = c->dst_pair, t = SEQ_T(s->dst_exec, T_PSEQ);
      for (int h = 0; h < 4; h++) {
         items[num] = (struct seq_item) { .def = t, .last = t, .reg = 2 * q + h / 2, .half = h % 2 };
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
      for (int k = 0; k < 2; k++) {
         struct fp_opnd *o = &c->tex[n].coord[k];
         int it = o->kind == FP_OPND_VALUE ? value_item[o->index] :
                  input_item[o->index][o->comp];
         if (it < 0)
            return false;
         SEQ_READ(it, SEQ_T(c->tex[n].exec, T_TEX));
         /* the coordinates are read from pair p: s in 2p, t in 2p+1, as
          * fp20 values */
         int want = 2 * p + k;
         if (items[it].reg >= 0 && items[it].reg != want)
            return false;
         items[it].reg = want;
         items[it].fx10 = false;
      }
   }
#undef SEQ_READ

   /* colour channels own their half from the first write on */
   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || u->output_chan < 0)
         continue;
      int h = fp_output_half(u->output_chan);
      int t = seq_alu_time(c, u, true);
      if (out_item[h] < 0) {
         items[num] = (struct seq_item) { .def = t, .last = SEQ_FOREVER, .reg = 2 * c->out_pair + h / 2, .half = h % 2 };
         out_item[h] = num++;
      }
      items[out_item[h]].def = MIN2(items[out_item[h]].def, t);
   }

   /* items without readers (results only forwarded) take no register */
   for (int i = 0; i < num; i++)
      if (items[i].last < 0)
         items[i].last = items[i].def - 1;

   /* greedy in order of definition, fixed registers first at equal times */
   int order[ARRAY_SIZE(items)];
   for (int i = 0; i < num; i++)
      order[i] = i;
   for (int i = 1; i < num; i++)
      for (int k = i; k > 0; k--) {
         struct seq_item *a = &items[order[k - 1]], *b = &items[order[k]];
         if (a->def < b->def || (a->def == b->def && (a->reg >= 0 || b->reg < 0)))
            break;
         int tmp = order[k]; order[k] = order[k - 1]; order[k - 1] = tmp;
      }

   int until[SEQ_REGS][2];
   memset(until, -1, sizeof(until));
   int assigned[ARRAY_SIZE(items)], assigned_half[ARRAY_SIZE(items)];
   for (int oi = 0; oi < num; oi++) {
      int i = order[oi];
      struct seq_item *it = &items[i];
      assigned[i] = assigned_half[i] = -1;
      if (it->last < it->def)
         continue;

      int r = it->reg, h = it->half;
      if (r < 0) {
         /* a free register, or half for a fixed10 value, that no fixed
          * item needs meanwhile */
         for (r = 0; r < SEQ_REGS; r++) {
            if (it->fx10) {
               for (h = 0; h < 2; h++)
                  if (until[r][h] < it->def && !seq_fixed_clash(items, num, i, it, r, h))
                     break;
               if (h < 2)
                  break;
            } else if (until[r][0] < it->def && until[r][1] < it->def &&
                       !seq_fixed_clash(items, num, i, it, r, -1))
               break;
         }
         if (r == SEQ_REGS) {
            if (grate_debug & GRATE_DEBUG_TGSI)
               fprintf(stderr, "GRATE FRAG: no register for item %d [%d, %d]%s\n",
                       i, it->def, it->last, it->fx10 ? " (fixed10)" : "");
            return false;
         }
         if (!it->fx10)
            h = -1;
      } else if (h >= 0 ? until[r][h] >= it->def :
                          until[r][0] >= it->def || until[r][1] >= it->def) {
         if (grate_debug & GRATE_DEBUG_TGSI)
            fprintf(stderr, "GRATE FRAG: fixed r%d%s for item %d [%d, %d] busy until %d/%d\n",
                    r, h < 0 ? "" : h ? ".h" : ".l", i, it->def, it->last,
                    until[r][0], until[r][1]);
         return false;
      }
      assigned[i] = r;
      assigned_half[i] = h;
      for (int k = 0; k < 2; k++)
         if (h < 0 || h == k)
            until[r][k] = it->last;
   }

   for (int i = 0; i < c->num_units; i++) {
      struct fp_unit *u = &c->units[i];
      bool held = u->live && value_item[i] >= 0;
      u->reg = held ? assigned[value_item[i]] : -1;
      u->half = held ? assigned_half[value_item[i]] : -1;
   }
   return true;
}

static bool
seq_try(struct fp_compiler *c, int step)
{
   struct seq_state s;

   for (c->out_pair = 1; c->out_pair >= 0; c->out_pair--)
   for (int limit = SEQ_HALVES; limit > 0; limit -= step) {
      if (!seq_schedule(c, &s, limit)) {
         if (grate_debug & GRATE_DEBUG_TGSI)
            fprintf(stderr, "GRATE FRAG: sequential schedule failed (limit %d)\n", limit);
         continue;
      }
      if (grate_debug & GRATE_DEBUG_TGSI)
         fprintf(stderr, "GRATE FRAG: sequential schedule ok (limit %d, %d EXECs, "
                 "colour in r%d/r%d), allocating\n", limit, s.num_execs,
                 2 * c->out_pair, 2 * c->out_pair + 1);
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

/* fp20 values first; fixed10 halves for those that may (fx10.c) only when
 * the program does not fit otherwise. GRATE_FP_FX10=0 turns them off. */
bool
fp_sched_seq(struct fp_compiler *c)
{
   for (int i = 0; i < c->num_units; i++)
      c->units[i].fx10 = false;
   if (seq_try(c, 2))
      return true;
   if (getenv("GRATE_FP_FX10") && !atoi(getenv("GRATE_FP_FX10")))
      return false;

   fp_mark_fx10(c);
   int n = 0;
   for (int i = 0; i < c->num_units; i++)
      n += c->units[i].fx10;
   if (grate_debug & GRATE_DEBUG_TGSI)
      fprintf(stderr, "GRATE FRAG: retrying with %d fixed10 values\n", n);
   return n && seq_try(c, 1);
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
      ret = fp_alu_src_row(2 * pair + fp_output_half(o->comp) / 2);
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
      a->dst.saturate = u->saturate;
      if (u->output_chan >= 0) {
         int o = fp_output_half(u->output_chan);
         a->dst.index = 2 * c->out_pair + o / 2;
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

   for (int e = 0; e < c->num_execs; e++) {
      struct fp_seq_exec *x = &c->seq_exec[e];
      struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
      list_inithead(&inst->link);

      if (x->pseq)
         inst->pseq = fp_pseq_word(c, c->dst_pair);

      bool ipl = false;
      for (int k = 0; k < 4; k++)
         ipl |= x->ipl_input[k] >= 0;
      if (ipl || x->sfu_unit >= 0) {
         struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
         list_inithead(&mfu->link);
         if (ipl) {
            /* every interpolating EXEC sets up the weights (as the NVIDIA
             * compiler does in one-row programs) */
            fp_mfu_bary_setup(mfu);
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
            mfu->sfu.reg = u->reg;
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

      /* the store runs in the last pass */
      if (e == c->num_execs - 1)
         inst->dw = c->dw;

      list_addtail(&inst->link, &fp->fp_instructions);
   }

   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "GRATE FRAG: sequential, %d EXECs:", c->num_execs);
      for (int e = 0; e < c->num_execs; e++) {
         struct fp_seq_exec *x = &c->seq_exec[e];
         fprintf(stderr, " [e%d%s%s%s %d alu]", e,
                 x->sfu_unit >= 0 ? " sfu" : "", x->tex >= 0 ? " tex" : "",
                 x->pseq ? " pseq" : "", x->num_packets);
      }
      fprintf(stderr, "\n");
   }
}
