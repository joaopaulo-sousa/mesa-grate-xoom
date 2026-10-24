#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grate_common.h"
#include "fp_compiler.h"

#include "util/u_math.h"
#include "util/u_memory.h"

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
 * The units are list-scheduled into packets and EXECs honouring the latencies
 * above, and the values that have to outlive the next packet get a register.
 * The colour is written as fixed10 to r2/r3.
 */

/* Give every varying component read an MFU of the earliest EXEC that has its
 * component slot free, in order of first use */
static void
fp_assign_inputs(struct fp_compiler *c, bool packed)
{
   memset(c->fetch_exec, -1, sizeof(c->fetch_exec));
   memset(c->fetch_col, -1, sizeof(c->fetch_col));
   memset(c->exec_input, -1, sizeof(c->exec_input));
   memset(c->exec_comp, -1, sizeof(c->exec_comp));

   /*
    * Texture fetches first, one EXEC each: the coordinate pair is fetched by
    * that EXEC's MFU, the other pair receives the texel (-2: no varying).
    * The texels with the longest chain of operations after them come first,
    * their chains then start earliest.
    */
   int height[FP_MAX_UNITS], tex_height[FP_MAX_TEXTURES] = { 0 };
   for (int i = c->num_units - 1; i >= 0; i--)
      height[i] = c->units[i].live && !fp_unit_has_value(&c->units[i]) ? 1 : 0;
   for (int i = c->num_units - 1; i >= 0; i--) {
      struct fp_unit *u = &c->units[i];
      if (!u->live || !height[i])
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE)
            height[o->index] = MAX2(height[o->index], height[i] + 1);
         else if (o->kind == FP_OPND_TEXEL)
            tex_height[o->index] = MAX2(tex_height[o->index], height[i]);
      }
   }
   int order[FP_MAX_TEXTURES], num = 0;
   for (int n = 0; n < c->num_tex; n++) {
      int at = num++;
      while (at > 0 && tex_height[order[at - 1]] < tex_height[n]) {
         order[at] = order[at - 1];
         at--;
      }
      order[at] = n;
   }

   int e = 0;
   for (int t = 0; t < num; t++) {
      int n = order[t];
      if (!c->tex[n].live)
         continue;
      int p = c->tex[n].pair, in = c->tex[n].input;
      c->tex[n].exec = e;
      for (int k = 0; k < 2; k++) {
         int comp = c->tex[n].coord[k].comp;
         c->exec_input[e][2 * p + k] = in;
         c->exec_comp[e][2 * p + k] = comp;
         if (c->fetch_exec[in][comp] < 0) {
            c->fetch_exec[in][comp] = e;
            c->fetch_col[in][comp] = 2 * p + k;
         }
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
         /* the earliest EXEC with the component's own column free; or,
          * packed, the earliest with any: one beside a taken one first,
          * keeping free pairs for the destination colour */
         int col = -1;
         for (e = 0; e < FP_MAX_EXECS && col < 0; e++) {
            int *x = c->exec_input[e];
            if (x[o->comp] == -1)
               col = o->comp;
            for (int k = 0; k < 4 && col < 0 && packed; k++)
               if (x[k] == -1 && x[k ^ 1] != -1)
                  col = k;
            for (int k = 0; k < 4 && col < 0 && packed; k++)
               if (x[k] == -1)
                  col = k;
         }
         if (col < 0) {
            if (packed)
               fp_fail(c, "too many varyings");
            else
               c->failed = true;
            return;
         }
         e--;
         c->exec_input[e][col] = o->index;
         c->exec_comp[e][col] = o->comp;
         c->fetch_exec[o->index][o->comp] = e;
         c->fetch_col[o->index][o->comp] = col;
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
      if (packed)
         fp_fail(c, "no registers left for the destination colour");
      else
         c->failed = true;
   }
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
      return 4 * c->fetch_exec[o->index][o->comp] + c->fetch_col[o->index][o->comp];
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
   int delta = fp_unit_has_value(u) ? 1 : 0;

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
   int uses_left[FP_MAX_UNITS] = { 0 }, height[FP_MAX_UNITS];
   int remaining = 0, packet = 0, exec = 0, in_exec = 0, stalls = 0, open = 0;

   /* operations on the longest path from a unit to the end of the program:
    * the critical units go first */
   for (int i = c->num_units - 1; i >= 0; i--)
      height[i] = 1;
   for (int i = c->num_units - 1; i >= 0; i--) {
      struct fp_unit *u = &c->units[i];
      if (!u->live)
         continue;
      fp_unit_foreach_src(u, k, j) {
         struct fp_opnd *o = &u->slots[k].src[j];
         if (o->kind == FP_OPND_VALUE)
            height[o->index] = MAX2(height[o->index], height[i] + 1);
      }
   }

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
            if (best < 0 || height[i] > height[best] ||
                (height[i] == height[best] && delta < best_delta)) {
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
   case FP_OPND_SPECIAL:
      ret = fp_alu_src_special(o->index);
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
   ret.scale_by_two = o->twice;
   ret.minus_one = o->minus_one;
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
         a->dst.index = 2 + o / 2;
         a->dst.write_low_sub_reg = (o % 2) == 0;
         a->dst.write_high_sub_reg = (o % 2) != 0;
      } else {
         /* fp20 value: both halves; only forwarded ones go to r31 */
         a->dst.index = u->kill ? FP_KILL_REG : u->reg >= 0 ? u->reg : 31;
         a->dst.write_low_sub_reg = true;
         a->dst.write_high_sub_reg = true;
      }
   }
}

void
fp_emit_rows(struct fp_compiler *c)
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
            mfu->var[k].tram_row = fp_tram_row(c, c->exec_input[e][k],
                                               c->exec_comp[e][k], k);
         }
         if (e == 0)
            fp_mfu_bary_setup(mfu);
         inst->mfu_sched.num_instructions = 1;
         inst->mfu_sched.address = list_length(&fp->mfu_instructions);
         list_addtail(&mfu->link, &fp->mfu_instructions);
      }

      if (c->dst_used && c->dst_exec == e)
         inst->pseq = fp_pseq_word(c, c->dst_pair);

      for (int n = 0; n < c->num_tex; n++) {
         if (!c->tex[n].live || c->tex[n].exec != e)
            continue;
         inst->tex = fp_tex_word(c, n, c->tex[n].pair);
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

void
fp_dump_rows(struct fp_compiler *c)
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

/*
 * Schedule for parallelism first, then with fewer values open at a time
 * until the registers suffice. More rows give more registers;
 * GRATE_FP_ROWS forces a minimum.
 */
bool
fp_sched_rows(struct fp_compiler *c)
{
   int min_rows = 1;

   if (getenv("GRATE_FP_ROWS"))
      min_rows = CLAMP(atoi(getenv("GRATE_FP_ROWS")), 1, FP_MAX_EXECS);

   /* varyings in their own columns; packed into earlier EXECs if that
    * does not fit (the destination colour and what reads them get room) */
   for (int packed = 0; packed < 2; packed++) {
      fp_assign_inputs(c, packed);
      if (c->failed && !packed) {
         c->failed = false;
         continue;
      }
      if (c->failed)
         return false;

      for (int limit = 16; limit > 0; limit--) {
         if (!fp_schedule(c, limit))
            continue;
         fp_pad_execs(c, min_rows);
         for (;;) {
            if (fp_alloc_regs(c))
               return true;
            if (c->num_execs == FP_MAX_EXECS)
               break;
            fp_pad_execs(c, c->num_execs + 1);
         }
      }
   }
   return false;
}
