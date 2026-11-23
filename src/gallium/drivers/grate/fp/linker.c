/*
 * TRAM layout and linker instructions. The linker (SU) copies components of
 * the VS exports to columns of TRAM rows; the fragment MFU interpolates
 * column k of a row into register k of its pixel packet row. So a varying
 * component can be brought into any register by putting it in that column
 * of some row, as the NVIDIA compiler does: the schedulers ask for the row
 * holding (input, component) in a column (fp_tram_row), and the linker
 * instructions follow from the resulting layout.
 */
#include "fp_compiler.h"
#include "tgr_3d.xml.h"

#include "pipe/p_shader_tokens.h"
#include "util/macros.h"

#define LINK_DST(index, comp, type) (((comp) | (type) << 2) << ((index) * 4))
#define LINK_DST_FP20      3

/* Row holding TGSI input "input" component "comp" in column "column" (a new
 * one if needed), -1 if the TRAM is full */
int
fp_tram_row(struct fp_compiler *c, int input, int comp, int column)
{
   int want = input * 4 + comp;

   for (int r = 0; r < FP_TRAM_ROWS; r++)
      if (c->tram[r][column] == want)
         return r;
   for (int r = 0; r < FP_TRAM_ROWS; r++)
      if (c->tram[r][column] < 0) {
         c->tram[r][column] = want;
         return r;
      }
   fp_fail(c, "too many varying components");
   return -1;
}

/*
 * One linker instruction per row and input, with the components that input
 * gives the row. An export component goes to a single column per
 * instruction: the same component twice in a row takes a second one.
 */
void
fp_build_links(struct fp_compiler *c)
{
   struct grate_fp_info *info = &c->fp->info;

   info->num_links = 0;
   info->max_tram_row = 1;

   for (int r = 0; r < FP_TRAM_ROWS; r++) {
      bool placed[4] = { false };

      for (int col = 0; col < 4; col++) {
         if (c->tram[r][col] < 0 || placed[col])
            continue;
         int input = c->tram[r][col] / 4;
         uint32_t dst = 0, used = 0;

         for (int k = col; k < 4; k++) {
            int comp = c->tram[r][k] % 4;
            if (c->tram[r][k] < 0 || placed[k] || c->tram[r][k] / 4 != input ||
                (used & (1u << comp)))
               continue;
            dst |= LINK_DST(comp, k, LINK_DST_FP20);
            used |= 1u << comp;
            placed[k] = true;
         }

         if (info->num_links == GRATE_FP_MAX_LINKS) {
            fp_fail(c, "too many linker instructions");
            return;
         }
         /* gl_PointCoord: the setup unit puts the point sprite's S and T in
          * place of components x and y of whatever export the link reads */
         uint32_t point = 0;
         if (c->input_semantic[input] >> 8 == TGSI_SEMANTIC_PCOORD)
            point = (used & 1 ? TGR3D_SU_INST_EVEN_P0_POINT(TGR3D_POINT_S) : 0) |
                    (used & 2 ? TGR3D_SU_INST_EVEN_P1_POINT(TGR3D_POINT_T) : 0);
         info->links[info->num_links++] = (struct grate_fp_link) {
            .dst = dst,
            .point = point,
            .semantic = c->input_semantic[input],
            .tram_row = r,
            .input = input,
         };
         info->max_tram_row = MAX2(info->max_tram_row, r + 1);
      }
   }
}
