/*
 * VC-1 frame threading statistics. Diagnostic only, not meant for upstream.
 */
#ifndef AVCODEC_VC1_STATS_H
#define AVCODEC_VC1_STATS_H

#include <stdatomic.h>
#include <stdint.h>

#include "libavutil/time.h"
#include "threadprogress.h"

enum {
    VC1_ST_WAIT_MC,        ///< blocked on reference rows in motion compensation
    VC1_ST_WAIT_DIRECT,    ///< blocked on co-located motion of the next anchor
    VC1_ST_WAIT_SKIP,      ///< blocked on rows copied by a skipped picture
    VC1_ST_WAIT_OUTPUT,    ///< blocked on a picture about to be output
    VC1_ST_TOTAL,          ///< time spent in vc1_decode_frame()
    VC1_ST_SETUP_NOW,      ///< entry until hand-over right after the header
    VC1_ST_SETUP_FIELD,    ///< entry until hand-over after the 2nd field header
    VC1_ST_SETUP_END,      ///< pictures handed over only after decoding
    VC1_ST_PIC_I, VC1_ST_PIC_P, VC1_ST_PIC_B, VC1_ST_PIC_FIELD,
    VC1_ST_NB
};

extern atomic_llong ff_vc1_st_time[VC1_ST_NB];   ///< microseconds
extern atomic_llong ff_vc1_st_count[VC1_ST_NB];

static inline void ff_vc1_st_add(int kind, int64_t us)
{
    atomic_fetch_add_explicit(&ff_vc1_st_time[kind], us, memory_order_relaxed);
    atomic_fetch_add_explicit(&ff_vc1_st_count[kind], 1, memory_order_relaxed);
}

/** ff_thread_progress_await() that accounts for the time it blocks. */
static inline void ff_vc1_st_await(const ThreadProgress *pro, int n, int kind)
{
    int64_t t;

    if (atomic_load_explicit(&pro->progress, memory_order_acquire) >= n)
        return;
    t = av_gettime_relative();
    ff_thread_progress_await(pro, n);
    ff_vc1_st_add(kind, av_gettime_relative() - t);
}

#endif /* AVCODEC_VC1_STATS_H */
