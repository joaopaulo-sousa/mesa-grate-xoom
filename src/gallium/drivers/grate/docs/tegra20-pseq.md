# Tegra 2 (Tegra20) GR3D: the program sequencer and multi-pass fragment programs

This note describes how the Tegra20 3D engine runs fragment programs longer
than its instruction memories hold, by running them as several passes driven
by the **program sequencer (PSEQ)** from memory. None of this was documented
before: the grate wiki (`docs/fragment-shader-isa.md` of grate-driver/grate)
says the feature exists, cites the patents, and states that "the BLOB driver
gives up on large shaders". **That last part is wrong.** NVIDIA's GLES driver
uses multi-pass routinely (the RetroArch `crt-easymode` shader is 3 passes of
64 + 64 + 44 EXECs), and the mechanism is simple once its command table is
known.

Everything here was found by static analysis of NVIDIA's L4T/Android
binaries (`libGLESv2_tegra.so`, `libcgdrv.so`, addresses below are for the
copies in this project) and **verified on hardware** (Motorola Xoom, Tegra
250) with the `asmtri` test program and with mesa-grate, where it is now used
(commit "grate: run fragment programs longer than 64 EXECs as several PSEQ
passes"). Things that were seen but not verified are marked as such.

Contents:

1. [How a fragment program runs](#1-how-a-fragment-program-runs)
2. [The PSEQ command table](#2-the-pseq-command-table)
3. [What goes in the buffer](#3-what-goes-in-the-buffer)
4. [The spill surface](#4-the-spill-surface)
5. [State across passes](#5-state-across-passes)
6. [A complete example](#6-a-complete-example)
7. [How NVIDIA's driver does it](#7-how-nvidias-driver-does-it)
8. [NVIDIA's fragment program binary](#8-nvidias-fragment-program-binary)
9. [How mesa-grate does it](#9-how-mesa-grate-does-it)
10. [Pitfalls (things that hang the GPU)](#10-pitfalls-things-that-hang-the-gpu)
11. [Other findings](#11-other-findings)
12. [Open questions](#12-open-questions)

---

## 1. How a fragment program runs

The fragment pipeline has five units, each with its own instruction memory:

| Unit | Does | Instruction memory (per pass) | Upload register |
|------|------|------------------------------|-----------------|
| PSEQ | sequencing, reads registers from render targets | 64 words | `PSEQ_INST_DATA` 0x541 |
| MFU (AT) | varying interpolation, SFU (rcp, rsq, log, exp, sin, cos) | 64 schedules + 64 instructions (2 words each) | `AT_REMAP_DATA` 0x601, `AT_INST_DATA_LO` 0x604 |
| TEX | texture fetches | 64 words | `TEX_INST_DATA` 0x701 |
| ALU | 4 scalar ALUs, as packets of up to 4 instructions | 64 schedules + 64 packets (8 words each) | `ALU_REMAP_DATA` 0x801, `ALU_INST_DATA` 0x804 |
| DW | writes the pixel to the render targets | 64 words | `DW_INST_DATA` 0x901 |

(plus `ALU_P2CX_DATA` 0x806, one word per EXEC, zero in every program seen.)

One **EXEC** is one step of the whole pipeline: entry *k* of the PSEQ, MFU
schedule, TEX, ALU schedule and DW memories together. A pixel quad goes
through the program EXEC by EXEC. The *schedule* words (the "remap"
memories) say which MFU instructions and ALU packets an EXEC runs (address
and count, see the grate ISA doc). So every memory holds **at most 64 EXECs**,
and the MFU and ALU hold at most 64 instructions / packets: that is the limit
of a single pass.

The ALU works on a buffer of 1 to 4 **rows** of registers per quad
(`GLOBAL_PIX_ATTR` 0xe20 `NUM_ROWS`, bits 0..1 = rows - 1). With more rows
more quads are in flight and the EXECs take turns over the rows (see the DW
note in section 5). A program on *N* rows must have a multiple of *N* EXECs.

Until now every open driver ran the program as the only pass, with one
command in the PSEQ command table and the code uploaded inline in the command
stream:

```
IMM  GLOBAL_INST_OFFSET (0xe22) = 0
IMM  AT_INST_OFFSET     (0x603) = 0
IMM  ALU_INST_OFFSET    (0x803) = 0
INCR PSEQ_COMMAND_EVEN(0) (0x520) = 0x20006000 | EXECs
INCR PSEQ_DWR_IF_STATE  (0x546) = EXECs << 6
IMM  PSEQ_FLUSH         (0x500) = 0
NONINCR PSEQ_INST_DATA ... DW_INST_DATA      (the code)
```

`0x20006000 | n` is "run *n* EXECs, this is the first and the last pass".

## 2. The PSEQ command table

`PSEQ_COMMAND_EVEN(i)` / `PSEQ_COMMAND_ODD(i)`, registers 0x520 + 2*i and
0x521 + 2*i, i = 0..15, hold **16 commands** of two words: the command in
EVEN, its argument (an address or 0) in ODD. `PSEQ_PC` (0x503) is the index
of the command to start at; writing it starts the sequencer on the table.

Command words (bits 28..31 = kind):

| Command (EVEN) | Argument (ODD) | Meaning | Verified |
|----------------|----------------|---------|----------|
| `0x10000000 \| n` | address | **fetch** *n* words from memory and execute them as host1x opcodes on the 3D class (register writes: the code of a pass) | yes |
| `0x00000000 \| n` | address | fetch *n* words too; NVIDIA uses it for the uniforms of a pass (`INCR ALU_GLOBALS(0) (0x820), 32` + 32 values = 33 words) | yes (as a uniform reload) |
| `0x20002000 \| e` | 0 | **run** *e* EXECs (bits 0..7, up to 64), more passes follow | yes |
| `0x20004000 \| e` | 0 | run *e* EXECs, the **last** pass | yes |
| `0x20006000 \| e` | 0 | run *e* EXECs, the only pass (first and last) | yes |
| `0x40000000 \| d` | 0 | **jump** back: d = (-n) & 31, n = how many commands back. NVIDIA always ends its lists with it, back to command 0, for the next batch of quads | yes |

So bit 13 of a run command seems to mean "not the last pass" and bit 14 "the
last pass" (store the pixel after it).

The jump is **relative** (`libcgdrv.so` 0xdce50 computes `0x40000000 |
(-index & 31)`). Every small list of NVIDIA's has its jump at index 6, so it
always reads `0x4000001a` (back 6), which is easy to mistake for a fixed "end
of list" word. If the jump does not land on command 0, the next batch of
quads starts at the wrong command: pixels go missing, without a hang.

A fetched block is a plain host1x opcode stream (`INCR`, `NONINCR`, `IMM`,
`MASK` on the 3D class). The PSEQ executes it as if it were in the command
stream, so a pass's code goes in with the same `NONINCR ..._INST_DATA`
uploads a driver uses inline.

## 3. What goes in the buffer

For each pass the buffer holds a block of register writes that first resets
the upload offsets, then uploads that pass's code (only its own EXECs, with
the MFU and ALU addresses in its schedules counted from the first instruction
of the pass):

```
INCR GLOBAL_INST_OFFSET (0xe22), 1: 0
INCR AT_INST_OFFSET     (0x603), 1: 0
INCR ALU_INST_OFFSET    (0x803), 1: 0
INCR PSEQ_FLUSH         (0x500), 1: 0
NONINCR PSEQ_INST_DATA  (0x541) x EXECs
NONINCR AT_REMAP_DATA   (0x601) x EXECs
NONINCR AT_INST_DATA_LO (0x604) x 2 * MFU instructions
NONINCR TEX_INST_DATA   (0x701) x EXECs
NONINCR ALU_REMAP_DATA  (0x801) x EXECs
NONINCR ALU_INST_DATA   (0x804) x 8 * ALU packets
NONINCR ALU_P2CX_DATA   (0x806) x EXECs
NONINCR DW_INST_DATA    (0x901) x EXECs
```

NVIDIA's blocks write `PSEQ_FLUSH` (0x500) after the PSEQ instructions rather
than before; both orders work. The command table goes in the command stream:

```
INCR GLOBAL_SPILLSURFADDR (0xe2a), 1: <spill buffer address>      (section 4)
INCR PSEQ_COMMAND_EVEN(0) (0x520), 2 * (2 * passes + 1):
     0x10000000 | words(pass 0),  address(pass 0)
     0x20002000 | EXECs(pass 0),  0
     0x10000000 | words(pass 1),  address(pass 1)
     ...
     0x20004000 | EXECs(last),    0
     0x40000000 | (-(2 * passes) & 31), 0
IMM  PSEQ_DWR_IF_STATE (0x546) = EXECs(last pass) << 6
IMM  PSEQ_PC (0x503) = 0
```

With uniforms reloaded per pass a `0x00000021, address` command goes before
each fetch (`crt-geom` has shorter ones, 2 and 4 words, presumably an
`INCR ALU_GLOBALS(k)` of only the slots that change; the constant area is
filled at draw time, so its contents were not seen). Sixteen commands allow 7 passes (14 commands + the jump).

**Longer lists** (seen in NVIDIA's binaries, *not tried on hardware*): the
last command of a full table is a fetch of the next part of the image
itself, which is again `INCR PSEQ_COMMAND_EVEN(0), 2k` + commands + `IMM
PSEQ_PC = 0`: it rewrites the table and restarts the sequencer at command 0.
Instead of the jump, the last part ends with a fetch of the first part,
which sets the first table back up for the next quads. `crt-geom` (8 passes):

```
table at image word 0, commands 0..15:
  cmd  0: 00000002 00000000  fetch 2 words (uniforms)
  cmd  1: 10000372 00000010  fetch 882 words (pass 0)
  cmd  2: 20002040 00000000  run 64 EXECs (more passes follow)
  ...                                       (passes 1..5, uniforms between)
  cmd 14: 20002040 00000000  run 64 EXECs (more passes follow)
  cmd 15: 1000000f 0000008c  fetch 15 words at image +0x8c (the next table)
table at image word 35, commands 0..5:
  cmd  0: 10000254 00003d08  fetch 596 words (pass 6)
  cmd  1: 20002040 00000000  run 64 EXECs (more passes follow)
  cmd  2: 00000004 00000018  fetch 4 words (uniforms)
  cmd  3: 10000216 00004668  fetch 534 words (pass 7)
  cmd  4: 2000402c 00000000  run 44 EXECs (last pass)
  cmd  5: 10000023 00000000  fetch 35 words at image +0 (the first table)
```

`xbrz-freescale` chains three tables this way for 14 passes, so the number
of passes has no fixed limit.

`PSEQ_DWR_IF_STATE` bits 6..12 ("PSEQ to DW exec number") is the number of
EXECs before the pixel is stored; with several passes it is the number of
EXECs of the last pass.

## 4. The spill surface

**`GLOBAL_SPILLSURFADDR` (0xe2a) must point to a buffer before a multi-pass
draw, or the GPU hangs** (host1x `cdma_timeout`, confirmed by leaving it
out). The quads finished with a pass wait there while the pipeline runs
other quads, before they recirculate for the next pass (this is the
"recirculation" of the patents US8411096 and US8659601).

NVIDIA allocates 0x13d0 bytes (317 x 16 bytes, one 16-byte entry per quad
ID). mesa-grate uses one 8 KiB buffer per context.

## 5. State across passes

All verified with `asmtri`, comparing the image of a program split into
2, 3 and 4 passes on 1 and 4 rows with the same program in one pass:

* **ALU registers carry over** from one pass to the next: pass 1 can compute
  and pass 2 only store.
* **Uniforms carry over** too: a pass does not need to reload them, unless
  it uses other values in the same slots (that is what NVIDIA's reload
  commands are for: a program with more uniforms than the 32 slots).
* **Interpolation:** the first EXEC that interpolates in each pass after the
  first must set up the barycentric weights again (the bary flag of its
  MFU instruction).
* **Rows:** every pass is a whole number of rounds of the rows (a pass starts
  on row 0), so each pass has a multiple of *rows* EXECs.
* **DW is row-relative:** the DW instruction of EXEC *k* reads the row of
  EXEC *k - 1* (mod rows): with 4 rows the store in the last EXEC reads
  r14/r15, not r2/r3.
* The ALU forwards results only within an EXEC, so any EXEC boundary is a
  valid cut point.

## 6. A complete example

`crt-easymode` (RetroArch), as compiled by NVIDIA's `cgc` (4 rows, 172
EXECs), decoded with `gpu/tools/cgdump/fpblob.py`:

```
cmd  0: 100002c2 00000010  fetch 706 words at +0x10
            INCR GLOBAL_INST_OFFSET x1: 0
            INCR AT_INST_OFFSET x1: 0
            INCR ALU_INST_OFFSET x1: 0
            NONINCR PSEQ_INST x64
            INCR PSEQ_FLUSH x1: 0
            NONINCR AT_REMAP x64
            NONINCR AT_INST_LO x58
            NONINCR TEX_INST x64
            NONINCR ALU_REMAP x64
            NONINCR ALU_INST x248
            NONINCR ALU_P2CX x64
            NONINCR DW_INST x64
cmd  1: 20002040 00000000  run 64 EXECs (more passes follow)
cmd  2: 10000322 00000b28  fetch 802 words at +0xb28
cmd  3: 20002040 00000000  run 64 EXECs (more passes follow)
cmd  4: 10000226 000017c0  fetch 550 words at +0x17c0
cmd  5: 2000402c 00000000  run 44 EXECs (last pass)
cmd  6: 4000001a 00000000  jump 6 back, to command 0 (next quads)
```

and a program with too many uniforms (16 `vec4`s), 2 passes of 4 EXECs each
with its uniforms in between:

```
cmd  0: 00000021 00000000  fetch 33 words at +0x0 (constant area: uniforms)
cmd  1: 1000006a 00000010  fetch 106 words at +0x10
cmd  2: 20002004 00000000  run 4 EXECs (more passes follow)
cmd  3: 00000021 00000084  fetch 33 words at +0x84 (constant area: uniforms)
cmd  4: 10000067 000001c8  fetch 103 words at +0x1c8
cmd  5: 20004004 00000000  run 4 EXECs (last pass)
cmd  6: 4000001a 00000000  jump 6 back, to command 0 (next quads)
```

## 7. How NVIDIA's driver does it

`libGLESv2_tegra.so` (draw at 0x21904):

* One pass (`[fs+216]` = 1): 0x305f4 writes `GLOBAL_PIX_ATTR`,
  `PSEQ_QUAD_ID` = 0, `PSEQ_CTL`, zeroes 0xe22/0x603/0x803, sets
  `PSEQ_COMMAND_EVEN(0)` = `0x20006000 | EXECs` and copies the code inline.
* More (`[fs+216]` > 1): 0x304c4 writes
  * `GLOBAL_PIX_ATTR` = (rows - 1) | MAX_QID << 24, MAX_QID from a table at
    0x492bc: 0x4a, 0x25, 0x18, 0x12 for 1..4 rows;
  * `PSEQ_QUAD_ID` (0x545) = 0;
  * `PSEQ_CTL` (0x501) with MAX_OUT = 300 (bits 4..13) and MIN_OUT = 0
    (bits 14..23; a one-pass program gets 200, `0x3212c0`), bit 3 set when the
    program has no register spill (section 11);
  * the 2-word bootstrap `PSEQ_COMMAND_EVEN(0)` = `0x10000000 | words`,
    `ODD(0)` = the buffer's address, then `PSEQ_PC` = 0: the PSEQ fetches the
    rest of the command table from the buffer and runs it;
  * the spill surface (0x3040e, `[ctx+1472]`).

The buffer (`[sl+32]`) is made when the program is linked: a copy of the
binary's *image* (section 8) relocated to its addresses (0x1f614), followed
by the constant area that the uniform fetches read, rewritten on each draw
whose uniforms changed.

mesa-grate's single-pass setup (MAX_OUT 300 and its usual `PSEQ_CTL`) also
works for multi-pass; MIN_OUT = 0 was not needed.

## 8. NVIDIA's fragment program binary

What `libGLESv2_tegra.so` loads (0x1ed44) is the program binary the `cgc`
compiler library returns (`cg->stream`; the `FPBLOB` line of our `cgdump`
tool), not its "BINHEX" listing:

| Word / offset | Contents |
|---------------|----------|
| word 0 | sum of all the other words, mod 2^32 |
| word 1 | magic `0x26836d2f` (fragment; vertex `0x26836d1f`) |
| word 2 | version `0x10006` |
| words 0..72 | header; at the byte offsets below, pairs {offset in words, size in bytes} |
| +184 | info, 80 bytes, copied to `[fs+212..]`: [1] passes, [4] rows, [5] EXECs, [7] register spill size, [12] bootstrap size |
| +232 / +236 | code: one AR20-BIN block per pass (16-byte header: rows, EXECs; then the host1x register writes of section 3) |
| +248 / +252 | image: host1x words, `INCR PSEQ_COMMAND_EVEN(0), 2k` + the table + `IMM PSEQ_PC = 0` |
| +264 / +268 | relocations: {word index in the image, type}: 0 = constant area, 1 = the code, 2 = the image itself |

The image is built by the compiler (`libcgdrv.so` 0xdcc80): a fetch is
`0x10000000 | (bytes / 4 - 4)` at the block's offset + 16 (skipping the
AR20-BIN header), a run is `0x20002000` / `0x20004000` / `0x20006000` | EXECs,
a uniform block `bytes / 4` at its offset in the constant area.

## 9. How mesa-grate does it

* `fp/sched_seq.c` `seq_chunk()`: a sequential schedule that fits in no
  single pass is cut greedily into passes of at most 64 EXECs, 64 ALU packets
  and 64 MFU instructions, each a whole number of rounds of the rows, up to
  7 passes (`GRATE_FP_MAX_CHUNKS`, one table of 16 commands). This is the **last** retry of
  `grate_fs_compile()`: one pass on more rows is faster than several on fewer.
* `grate_program.c` `grate_fp_chunks()`: the blocks of section 3, uploaded
  once to a BO per shader variant.
* `grate_state.c` `emit_fs_passes()`: the spill surface, the command table,
  `PSEQ_PC` = 0. A one-pass program drawn after a multi-pass one clears what
  the table holds past command 0.
* `GRATE_FP_CHUNK_EXECS=n` cuts every sequential program into passes of at
  most *n* EXECs: a test of the passes with programs that fit in one.

Results: dEQP-GLES2 mustpass unchanged (no regression in 17165 cases); with
`GRATE_FP_CHUNK_EXECS=8` on all the `shaders.*` and `texture.filtering`
groups no image changes (3 programs no longer fit in 7 passes of 8). The
fragment programs that still fail to compile are now short of **registers**,
not of EXECs (section 11).

## 10. Pitfalls (things that hang the GPU)

* A table with a fetch and no run or jump command: the PSEQ runs off the end
  of the table.
* No `GLOBAL_SPILLSURFADDR` with more than one pass.
* A pass, or a one-pass program, whose number of EXECs is not a multiple of
  its rows.
* A GPU hang on the Xoom can be followed by `spi-tegra20-slink` timeouts on
  the CPCAP PMIC and a division by zero in `tegra_slink_transfer_one()`'s
  late IRQ handling (bits_per_word = 0 after a timed-out transfer), after
  which voltage scaling fails and the system crawls until a reboot. host1x
  itself recovers from the hang. This is a separate kernel bug.

## 11. Other findings

* **Register spill.** NVIDIA's compiler also spills ALU registers to a
  second surface: info[7] (`[fs+240]`) x `[fs+244]` bytes (0x4c0 per program
  seen), registered by 0x2fa45 with index 4. Programs that need more
  registers than the rows hold (RetroArch `crt-geom`, dEQP
  `random.all_features.fragment.5`) depend on it. Not implemented yet.
* **Texture descriptor** (`libGLESv2_tegra.so` 0x24810): a cube map gets
  `LO |= 0x83` (CUBEMAP + clamp S/T); an NPOT texture `HI = W << 20 | H << 8 |
  NORMALIZE | NPOT | ARRAY_MAX` and `LO |= 3`; a POT texture without mipmaps
  `LOD_MAX = 1/16` + `BASE_LEVEL_ONLY`; `ARRAY_MAX = [tex+80] - 1`.
* **Default register state.** NVIDIA's driver initialises the 3D class from
  a table of {reg << 16 | count, value} pairs at 0x492c0 of
  `libGLESv2_tegra.so` (`gpu/tools/regstate.py` compares a mesa-grate
  command stream with it, from `GRATE_DUMP_STREAM=file`).

## 12. Open questions

* The bits of a run command other than 13 and 14 (8..12 and 15 are 0 in
  every list seen), and what kind 0 does differently from kind 1 besides
  being used for uniforms.
* Chained tables for more than 7 passes (section 3) work as NVIDIA's
  binaries suggest; mesa-grate stops at 7 passes for now.
* What MIN_OUT = 0 and bit 3 of `PSEQ_CTL` are for.
* The register spill surface (section 11).
* Whether the PSEQ instructions' render target reads (render target 4 =
  "spill buffer" in the grate ISA doc) are how the spill surface is read
  back.

## Tools

These live in the Xoom port's workspace, not in this repository (`gpu/tools`):

* `gpu/tools/cgdump`: compiles GLSL with NVIDIA's `libcgdrv.so` on the device
  and dumps the binary; `fpblob.py` decodes it (sections 6 and 8).
* `gpu/tools/asmtri`: grate's `asmtri` test with `GRATE_PSEQ_MEM=3` (one pass
  from memory) and `=4` (N passes, `ASM_FS`..`ASM_FS8`, `GRATE_PSEQ_RELOAD=1`
  for uniform reloads); experiments in `exp-pseq2`, `exp-pseq3`.
