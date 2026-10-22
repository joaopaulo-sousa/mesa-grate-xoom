#ifndef GRATE_TWOPASS_H
#define GRATE_TWOPASS_H

#include "pipe/p_state.h"

struct grate_context;
struct grate_twopass;

/* Draw blended in two passes when the fragment program only fits unblended
 * (grate_twopass.c); false when the draw does not qualify */
bool
grate_twopass_draw(struct grate_context *context, const struct pipe_draw_info *info,
                   unsigned drawid_offset, const struct pipe_draw_start_count_bias *draws,
                   unsigned num_draws);

void
grate_twopass_destroy(struct grate_context *context, struct grate_twopass *tp);

#endif
