/*
 * hdr10_to_sdr_lut.c
 *
 * Analyses an HDR10 video, extracts static HDR10 metadata, then generates
 * a 3D LUT (.cube) using libplacebo BT.2390 tone-mapping on CPU.
 *
 * Validation + auto-tune pass:
 *   1. Decodes sample frames, extracts PQ RGB pixels into memory.
 *   2. Binary-searches peak_nits so white clipping < TUNE_WHITE_CLIP_TARGET.
 *   3. Adjusts sdr_white (SDR normalization point) so mean output luma
 *      falls in a reasonable range.
 *   4. Re-checks and re-tunes if needed after the sdr_white change.
 *   5. Reports a full validation histogram + stats before writing the LUT.
 *
 * Usage:
 *   hdr10_to_sdr_lut <input_hdr10_video> [output.cube] [lut_size=65]
 *
 * Apply with FFmpeg (no GPU needed):
 *   ffmpeg -i input.mkv -vf "lut3d=output.cube" -c:v libx264 -crf 18 out.mp4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
#include <libavutil/opt.h>

#include <libplacebo/colorspace.h>
#include <libplacebo/tone_mapping.h>

/* ── LUT / signal constants ─────────────────────────────────────────────── */
#define DEFAULT_LUT_SIZE        65
#define PQ_MAX_NITS          10000.0f
#define SDR_REF_WHITE          203.0f
#define FALLBACK_PEAK_NITS    1000.0f
#define FALLBACK_MIN_NITS      0.005f
#define MAX_SCAN_FRAMES         500

/* ── Sampling constants ─────────────────────────────────────────────────── */
#define VAL_FRAMES              20       /* spread frames to sample          */
#define VAL_PIXEL_STEP          16       /* sample every Nth pixel (x & y)   */
#define VAL_MAX_SAMPLES    3000000L      /* max PQ pixels kept in memory     */

/* ── Auto-tune targets ──────────────────────────────────────────────────── */
#define TUNE_WHITE_CLIP_TARGET   0.50    /* % pixels allowed to white-clip   */
#define TUNE_MEAN_LOW            0.18    /* re-adjust if mean luma below this */
#define TUNE_MEAN_HIGH           0.58    /* re-adjust if mean luma above this */
#define TUNE_MEAN_TARGET         0.35    /* desired mean output luma          */
#define TUNE_ITERS               24      /* binary-search iterations          */

/* ── Reporting ──────────────────────────────────────────────────────────── */
#define HIST_BINS               12

/* ═══════════════════════════════════════════════════════════════════════════
 * HDR metadata
 * ═══════════════════════════════════════════════════════════════════════════*/
typedef struct {
    float peak_nits;
    float min_nits;
    float max_cll;
    float max_fall;
    float sdr_white;      /* SDR normalisation point (default SDR_REF_WHITE) */
    int   have_mastering;
    int   have_cll;
} HDRMeta;

/* ═══════════════════════════════════════════════════════════════════════════
 * PQ RGB pixel sample (signal [0,1])
 * ═══════════════════════════════════════════════════════════════════════════*/
typedef struct { float r, g, b; } PQSample;

/* ═══════════════════════════════════════════════════════════════════════════
 * Evaluation result from running a sample buffer through the LUT pipeline
 * ═══════════════════════════════════════════════════════════════════════════*/
typedef struct {
    double white_clip_pct;
    double black_clip_pct;
    double mean_luma_out;
} EvalResult;

/* ═══════════════════════════════════════════════════════════════════════════
 * Transfer functions
 * ═══════════════════════════════════════════════════════════════════════════*/
static float pq_eotf(float x)
{
    if (x <= 0.0f) return 0.0f;
    const float m1i = 16384.0f / 2610.0f;
    const float m2i = 4096.0f  / (2523.0f * 128.0f);
    const float c1  = 3424.0f  / 4096.0f;
    const float c2  = 2413.0f  * 32.0f / 4096.0f;
    const float c3  = 2392.0f  * 32.0f / 4096.0f;
    float xm  = powf(x, m2i);
    float num = fmaxf(xm - c1, 0.0f);
    float den = c2 - c3 * xm;
    return (den <= 0.0f) ? 0.0f : PQ_MAX_NITS * powf(num / den, m1i);
}

static float srgb_oetf(float x)
{
    x = fmaxf(x, 0.0f);
    return (x <= 0.0031308f) ? 12.92f * x
                              : 1.055f * powf(x, 1.0f/2.4f) - 0.055f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BT.2020 → BT.709 matrix
 * ═══════════════════════════════════════════════════════════════════════════*/
static const float M_2020_709[3][3] = {
    { 1.6604910f, -0.5876411f, -0.0728499f },
    {-0.1245505f,  1.1328999f, -0.0083494f },
    {-0.0181508f, -0.1005789f,  1.1187297f },
};
static void mat3_mul(const float M[3][3], float r, float g, float b,
                     float *or, float *og, float *ob)
{
    *or = M[0][0]*r + M[0][1]*g + M[0][2]*b;
    *og = M[1][0]*r + M[1][1]*g + M[1][2]*b;
    *ob = M[2][0]*r + M[2][1]*g + M[2][2]*b;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 10-bit YCbCr → PQ-encoded RGB [0,1]
 * Handles YUV420P10LE / YUV422P10LE / YUV444P10LE.
 * Returns 0 on success, -1 if format unsupported.
 * ═══════════════════════════════════════════════════════════════════════════*/
static int frame_get_pq_rgb(const AVFrame *f, int x, int y,
                             float *r, float *g, float *b)
{
    int fmt = f->format;
    if (fmt != AV_PIX_FMT_YUV420P10LE &&
        fmt != AV_PIX_FMT_YUV422P10LE &&
        fmt != AV_PIX_FMT_YUV444P10LE) return -1;

    int Ys  = f->linesize[0] / 2;
    int UVs = f->linesize[1] / 2;
    int cx  = x, cy = y;
    if      (fmt == AV_PIX_FMT_YUV420P10LE) { cx = x>>1; cy = y>>1; }
    else if (fmt == AV_PIX_FMT_YUV422P10LE) { cx = x>>1; }

    const uint16_t *Yp  = (const uint16_t *)f->data[0];
    const uint16_t *Cbp = (const uint16_t *)f->data[1];
    const uint16_t *Crp = (const uint16_t *)f->data[2];

    float Yn  = ((float)Yp [y  * Ys  + x ] -  64.0f) / 876.0f;
    float Pbn = ((float)Cbp[cy * UVs + cx] - 512.0f) / 896.0f;
    float Prn = ((float)Crp[cy * UVs + cx] - 512.0f) / 896.0f;

    /* BT.2020 YCbCr → PQ R'G'B' */
    float R = Yn + 1.4746f * Prn;
    float B = Yn + 1.8814f * Pbn;
    float G = (Yn - 0.2627f * R - 0.0593f * B) / 0.6780f;
    *r = fmaxf(0.0f, fminf(1.0f, R));
    *g = fmaxf(0.0f, fminf(1.0f, G));
    *b = fmaxf(0.0f, fminf(1.0f, B));
    return 0;
}

static int cmp_float_asc(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * libplacebo params builder
 * ═══════════════════════════════════════════════════════════════════════════*/
static struct pl_tone_map_params build_tm_params(const HDRMeta *hdr)
{
    struct pl_tone_map_params p = {0};
    p.function   = &pl_tone_map_bt2390;
    p.input_max  = hdr->peak_nits;
    p.input_min  = hdr->min_nits;
    p.output_max = hdr->sdr_white;
    p.output_min = 0.0f;
    p.hdr.max_luma = hdr->peak_nits;
    p.hdr.min_luma = hdr->min_nits;
    p.hdr.max_cll  = hdr->max_cll;
    p.hdr.max_fall = hdr->max_fall;
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Core colour pipeline: PQ/BT.2020 [0,1] → sRGB/BT.709 [0,1]
 * ═══════════════════════════════════════════════════════════════════════════*/
static void convert_sample(float ri, float gi, float bi,
                           float *ro, float *go, float *bo,
                           const struct pl_tone_map_params *tm)
{
    float lr = pq_eotf(ri), lg = pq_eotf(gi), lb = pq_eotf(bi);
    float r9, g9, b9;
    mat3_mul(M_2020_709, lr, lg, lb, &r9, &g9, &b9);

    float Y = 0.2126f*r9 + 0.7152f*g9 + 0.0722f*b9;
    float Y_sdr = (Y > 0.001f) ? pl_tone_map_sample(Y, tm) : 0.0f;
    float scale = (Y > 0.001f) ? (Y_sdr / Y) : 0.0f;

    float sln = Y_sdr / tm->output_max;
    float desat = fmaxf(0.0f, (sln - 0.8f) / 0.2f);

    float rn = r9 * scale / tm->output_max;
    float gn = g9 * scale / tm->output_max;
    float bn = b9 * scale / tm->output_max;
    rn += desat * (sln - rn);
    gn += desat * (sln - gn);
    bn += desat * (sln - bn);

    *ro = srgb_oetf(fminf(fmaxf(rn, 0.0f), 1.0f));
    *go = srgb_oetf(fminf(fmaxf(gn, 0.0f), 1.0f));
    *bo = srgb_oetf(fminf(fmaxf(bn, 0.0f), 1.0f));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Evaluate tone-map params against a pre-sampled pixel buffer (pure CPU math,
 * no video I/O — suitable for rapid iteration during binary search).
 * ═══════════════════════════════════════════════════════════════════════════*/
static EvalResult eval_lut(const PQSample *px, long n,
                            const struct pl_tone_map_params *tm)
{
    long   white_clip = 0, black_clip = 0;
    double sum_luma   = 0.0;

    for (long i = 0; i < n; i++) {
        float ro, go, bo;
        convert_sample(px[i].r, px[i].g, px[i].b, &ro, &go, &bo, tm);
        float Yo = 0.2126f*ro + 0.7152f*go + 0.0722f*bo;
        sum_luma += Yo;
        if (ro > 0.997f || go > 0.997f || bo > 0.997f) white_clip++;
        if (ro < 0.003f && go < 0.003f && bo < 0.003f) black_clip++;
    }

    EvalResult e;
    e.white_clip_pct = 100.0 * white_clip / n;
    e.black_clip_pct = 100.0 * black_clip / n;
    e.mean_luma_out  = sum_luma / n;
    return e;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Decode VAL_FRAMES spread across the video and collect PQSample pixels.
 * Also fills luma_buf[] (scene nits, for percentile stats) and *luma_n.
 * Returns heap-allocated sample buffer (caller frees), or NULL on error.
 * ═══════════════════════════════════════════════════════════════════════════*/
static PQSample *collect_samples(const char *path,
                                 long *out_n,
                                 float *luma_buf, long *luma_n,
                                 int *frames_done)
{
    *out_n = 0; *luma_n = 0; *frames_done = 0;

    AVFormatContext *fctx = NULL;
    if (avformat_open_input(&fctx, path, NULL, NULL) < 0) return NULL;
    avformat_find_stream_info(fctx, NULL);
    int vidx = av_find_best_stream(fctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vidx < 0) { avformat_close_input(&fctx); return NULL; }
    AVStream *vst = fctx->streams[vidx];

    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) { avformat_close_input(&fctx); return NULL; }
    AVCodecContext *cctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(cctx, vst->codecpar);
    cctx->thread_count = 4;
    cctx->skip_frame   = AVDISCARD_NONKEY;
    if (avcodec_open2(cctx, codec, NULL) < 0) {
        avcodec_free_context(&cctx); avformat_close_input(&fctx); return NULL;
    }

    PQSample *buf = malloc(VAL_MAX_SAMPLES * sizeof(PQSample));
    if (!buf) {
        avcodec_free_context(&cctx); avformat_close_input(&fctx); return NULL;
    }

    int warned_fmt = 0;
    int64_t dur    = fctx->duration;
    AVPacket *pkt  = av_packet_alloc();
    AVFrame  *frm  = av_frame_alloc();

    for (int fi = 0; fi < VAL_FRAMES && *out_n < VAL_MAX_SAMPLES; fi++) {
        if (dur > 0) {
            double frac = (VAL_FRAMES > 1) ? (double)fi / (VAL_FRAMES-1) : 0.0;
            int64_t ts  = av_rescale_q((int64_t)(frac * dur),
                                       AV_TIME_BASE_Q, vst->time_base);
            av_seek_frame(fctx, vidx, ts, AVSEEK_FLAG_BACKWARD);
            avcodec_flush_buffers(cctx);
        }
        int got = 0;
        while (!got && av_read_frame(fctx, pkt) >= 0) {
            if (pkt->stream_index != vidx) { av_packet_unref(pkt); continue; }
            if (avcodec_send_packet(cctx, pkt) == 0) {
                while (avcodec_receive_frame(cctx, frm) == 0 && !got) {
                    int fmt = frm->format;
                    if (fmt != AV_PIX_FMT_YUV420P10LE &&
                        fmt != AV_PIX_FMT_YUV422P10LE &&
                        fmt != AV_PIX_FMT_YUV444P10LE) {
                        if (!warned_fmt) {
                            fprintf(stderr,
                                "[validate] Unsupported pixel format '%s' "
                                "(need 10-bit YUV) — skipping validation.\n",
                                av_get_pix_fmt_name(fmt));
                            warned_fmt = 1;
                        }
                        av_frame_unref(frm); continue;
                    }
                    int w = frm->width, h = frm->height;
                    for (int y = 0; y < h && *out_n < VAL_MAX_SAMPLES;
                         y += VAL_PIXEL_STEP) {
                        for (int x = 0; x < w && *out_n < VAL_MAX_SAMPLES;
                             x += VAL_PIXEL_STEP) {
                            float r, g, b;
                            if (frame_get_pq_rgb(frm, x, y, &r, &g, &b) < 0)
                                continue;
                            buf[*out_n].r = r;
                            buf[*out_n].g = g;
                            buf[*out_n].b = b;
                            (*out_n)++;

                            /* Scene luminance in nits for percentile stats */
                            float lr = pq_eotf(r), lg = pq_eotf(g),
                                  lb = pq_eotf(b);
                            float r9, g9, b9;
                            mat3_mul(M_2020_709, lr, lg, lb, &r9, &g9, &b9);
                            float Yi = 0.2126f*r9 + 0.7152f*g9 + 0.0722f*b9;
                            if (*luma_n < VAL_MAX_SAMPLES)
                                luma_buf[(*luma_n)++] = Yi;
                        }
                    }
                    got = 1; (*frames_done)++;
                    av_frame_unref(frm);
                }
            }
            av_packet_unref(pkt);
        }
        printf("\r[validate] Sampling frame %2d / %d ...", fi+1, VAL_FRAMES);
        fflush(stdout);
    }
    printf("\n");

    av_frame_free(&frm);
    av_packet_free(&pkt);
    avcodec_free_context(&cctx);
    avformat_close_input(&fctx);
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Binary-search auto-tuner
 *
 * Phase 1 – find minimum peak_nits that satisfies white-clip target.
 *            (Higher peak → more headroom → less clipping but darker image.)
 * Phase 2 – adjust sdr_white so mean output luma hits TUNE_MEAN_TARGET.
 *            (Lower sdr_white → brighter; higher → darker.)
 * Phase 3 – if adjusting sdr_white reintroduced clipping, re-run Phase 1.
 * ═══════════════════════════════════════════════════════════════════════════*/
static void autotune_params(const PQSample *px, long n,
                             float p90, float p999, HDRMeta *meta)
{
    printf("\n[tune] Starting binary-search auto-tune "
           "(target: <%.1f%% white clip)...\n", TUNE_WHITE_CLIP_TARGET);

    /* ── Phase 1 ── */
    float lo = fmaxf(p90, 10.0f);
    float hi = 10000.0f;

    /* Ensure invariant: hi already satisfies the clip target */
    {
        HDRMeta t = *meta; t.peak_nits = hi;
        struct pl_tone_map_params tm = build_tm_params(&t);
        EvalResult e = eval_lut(px, n, &tm);
        if (e.white_clip_pct > TUNE_WHITE_CLIP_TARGET) {
            fprintf(stderr, "[tune] WARNING: white clipping persists even "
                    "at 10000 nits — content may have out-of-range values.\n");
            hi = 10000.0f;   /* best we can do */
        }
    }

    float last_wc = 0.0;
    for (int iter = 0; iter < TUNE_ITERS; iter++) {
        float mid = (lo + hi) * 0.5f;
        HDRMeta t = *meta; t.peak_nits = mid;
        struct pl_tone_map_params tm = build_tm_params(&t);
        EvalResult e = eval_lut(px, n, &tm);
        last_wc = (float)e.white_clip_pct;
        if (e.white_clip_pct > TUNE_WHITE_CLIP_TARGET)
            lo = mid;    /* too much clipping → need higher peak */
        else
            hi = mid;    /* fits → try lower (brighter) peak */
        printf("\r[tune] iter %2d/%d  peak=%7.1f nits  white_clip=%5.2f%%",
               iter+1, TUNE_ITERS, (double)mid, e.white_clip_pct);
        fflush(stdout);
    }
    printf("\n");
    meta->peak_nits = hi;
    (void)last_wc;

    /* ── Phase 2: adjust sdr_white for mean luma ── */
    {
        struct pl_tone_map_params tm = build_tm_params(meta);
        EvalResult e = eval_lut(px, n, &tm);
        printf("[tune] After peak tune: mean_luma=%.3f  white_clip=%.2f%%\n",
               e.mean_luma_out, e.white_clip_pct);

        if (e.mean_luma_out < TUNE_MEAN_LOW || e.mean_luma_out > TUNE_MEAN_HIGH) {
            /* sdr_white_new = sdr_white_old * (mean_actual / target)
             * because output ∝ 1/sdr_white, so halving sdr_white doubles luma */
            float new_sw = meta->sdr_white
                           * (float)(e.mean_luma_out / TUNE_MEAN_TARGET);
            new_sw = fmaxf(40.0f, fminf(600.0f, new_sw));
            printf("[tune] Mean luma %.3f outside [%.2f,%.2f]. "
                   "Adjusting SDR white: %.0f → %.0f nits.\n",
                   e.mean_luma_out, TUNE_MEAN_LOW, TUNE_MEAN_HIGH,
                   (double)meta->sdr_white, (double)new_sw);
            meta->sdr_white = new_sw;

            /* ── Phase 3: re-tune peak if clipping reintroduced ── */
            tm = build_tm_params(meta);
            e  = eval_lut(px, n, &tm);
            if (e.white_clip_pct > TUNE_WHITE_CLIP_TARGET) {
                printf("[tune] Re-tuning peak after SDR white adjustment...\n");
                lo = fmaxf(p90, 10.0f); hi = 10000.0f;
                for (int iter = 0; iter < TUNE_ITERS; iter++) {
                    float mid = (lo + hi) * 0.5f;
                    HDRMeta t = *meta; t.peak_nits = mid;
                    struct pl_tone_map_params tm2 = build_tm_params(&t);
                    EvalResult e2 = eval_lut(px, n, &tm2);
                    if (e2.white_clip_pct > TUNE_WHITE_CLIP_TARGET) lo = mid;
                    else                                             hi = mid;
                    printf("\r[tune] re-iter %2d/%d  peak=%7.1f nits  "
                           "white_clip=%5.2f%%",
                           iter+1, TUNE_ITERS, (double)mid, e2.white_clip_pct);
                    fflush(stdout);
                }
                printf("\n");
                meta->peak_nits = hi;
            }
        }
    }

    /* ── Final summary ── */
    struct pl_tone_map_params tm_f = build_tm_params(meta);
    EvalResult ef = eval_lut(px, n, &tm_f);
    printf("[tune] ✓ Tuned:  peak=%.0f nits  SDR white=%.0f nits\n",
           (double)meta->peak_nits, (double)meta->sdr_white);
    printf("[tune]   mean_luma=%.3f  white_clip=%.2f%%  black_clip=%.2f%%\n",
           ef.mean_luma_out, ef.white_clip_pct, ef.black_clip_pct);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Validation report  (printed after tuning, before LUT write)
 * ═══════════════════════════════════════════════════════════════════════════*/
static void print_validation_report(const PQSample *px, long n,
                                    const float *luma_buf, long luma_n,
                                    int frames_done,
                                    const HDRMeta *meta)
{
    struct pl_tone_map_params tm = build_tm_params(meta);

    long   white_clip = 0, black_clip = 0;
    double sum_out = 0.0;
    long   hist[HIST_BINS] = {0};

    for (long i = 0; i < n; i++) {
        float ro, go, bo;
        convert_sample(px[i].r, px[i].g, px[i].b, &ro, &go, &bo, &tm);
        float Yo = 0.2126f*ro + 0.7152f*go + 0.0722f*bo;
        sum_out += Yo;
        if (ro > 0.997f || go > 0.997f || bo > 0.997f) white_clip++;
        if (ro < 0.003f && go < 0.003f && bo < 0.003f) black_clip++;
        int bin = (int)(Yo * HIST_BINS);
        if (bin < 0) bin = 0;
        if (bin >= HIST_BINS) bin = HIST_BINS - 1;
        hist[bin]++;
    }

    /* Luma percentiles */
    /* luma_buf is already filled; we need a sorted copy */
    float *sorted = malloc(luma_n * sizeof(float));
    if (sorted) {
        memcpy(sorted, luma_buf, luma_n * sizeof(float));
        qsort(sorted, luma_n, sizeof(float), cmp_float_asc);
    }
    float p50  = sorted ? sorted[(long)(luma_n * 0.500)] : 0.0f;
    float p90  = sorted ? sorted[(long)(luma_n * 0.900)] : 0.0f;
    float p99  = sorted ? sorted[(long)(luma_n * 0.990)] : 0.0f;
    float p999 = sorted ? sorted[(long)(luma_n * 0.999)] : 0.0f;
    float pmax = sorted ? sorted[luma_n - 1]             : 0.0f;
    free(sorted);

    double mean_out   = sum_out / n;
    double pct_white  = 100.0 * white_clip / n;
    double pct_black  = 100.0 * black_clip / n;

    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│                  Final Validation Report                     │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│  Frames sampled : %-4d   Pixels sampled : %-14ld   │\n",
           frames_done, n);
    printf("│  Tuned peak     : %7.0f nits   SDR white: %5.0f nits      │\n",
           (double)meta->peak_nits, (double)meta->sdr_white);
    printf("│                                                              │\n");
    printf("│  Scene luminance (nits, BT.709 linear):                     │\n");
    printf("│    p50  = %8.2f   p90  = %8.2f                       │\n",
           (double)p50, (double)p90);
    printf("│    p99  = %8.2f   p99.9= %8.2f   max= %8.2f         │\n",
           (double)p99, (double)p999, (double)pmax);
    printf("│                                                              │\n");
    printf("│  LUT output (sRGB):                                         │\n");
    printf("│    Mean luma    : %.4f  (%.1f%% of SDR range)           │\n",
           mean_out, mean_out * 100.0);
    printf("│    White clip   : %6.2f%%                                   │\n",
           pct_white);
    printf("│    Black clip   : %6.2f%%                                   │\n",
           pct_black);
    printf("│                                                              │\n");
    printf("│  Output luma histogram (sRGB gamma):                        │\n");
    for (int b = 0; b < HIST_BINS; b++) {
        float pct = 100.0f * (float)hist[b] / (float)n;
        int bar = (int)(pct / 2.0f); if (bar > 24) bar = 24;
        printf("│   %3.0f%%–%3.0f%%  [",
               100.0f * b / HIST_BINS, 100.0f * (b+1) / HIST_BINS);
        for (int k = 0; k < bar; k++) printf("█");
        for (int k = bar; k < 24; k++) printf(" ");
        printf("] %5.1f%%  │\n", (double)pct);
    }
    printf("└──────────────────────────────────────────────────────────────┘\n\n");

    if (pct_white > TUNE_WHITE_CLIP_TARGET)
        fprintf(stderr, "[validate] WARNING: %.2f%% white clipping remains.\n",
                pct_white);
    if (pct_black > 15.0)
        fprintf(stderr, "[validate] WARNING: %.2f%% black clipping — shadows may be crushed.\n",
                pct_black);
    if (pct_white <= TUNE_WHITE_CLIP_TARGET && pct_black <= 15.0)
        printf("[validate] ✓ No significant clipping issues.\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Combined validate + auto-tune entry point
 * ═══════════════════════════════════════════════════════════════════════════*/
static void validate_and_adjust(const char *path, HDRMeta *meta)
{
    printf("\n[validate] Collecting pixel samples for auto-tune...\n");

    float *luma_buf = malloc(VAL_MAX_SAMPLES * sizeof(float));
    if (!luma_buf) { fprintf(stderr, "[validate] OOM\n"); return; }

    long n = 0, luma_n = 0;
    int  frames_done = 0;
    PQSample *px = collect_samples(path, &n, luma_buf, &luma_n, &frames_done);

    if (!px || n == 0) {
        fprintf(stderr, "[validate] No pixels sampled — skipping.\n");
        free(luma_buf); free(px); return;
    }
    printf("[validate] Collected %ld pixels from %d frame(s).\n", n, frames_done);

    /* Sort luma buffer once; extract percentiles for tune bounds */
    qsort(luma_buf, luma_n, sizeof(float), cmp_float_asc);
    float p90  = luma_buf[(long)(luma_n * 0.900)];
    float p999 = luma_buf[(long)(luma_n * 0.999)];

    /* If metadata dramatically overstates peak, snap it down first */
    if (p999 > 1.0f && p999 < meta->peak_nits * 0.80f) {
        printf("[validate] Metadata peak=%.0f nits but p99.9=%.0f nits — "
               "pre-clamping peak.\n", (double)meta->peak_nits, (double)p999);
        meta->peak_nits = p999 * 1.05f;
        if (meta->max_cll > meta->peak_nits) meta->max_cll = meta->peak_nits;
    }

    /* Binary-search auto-tune */
    autotune_params(px, n, p90, p999, meta);

    /* Print final validation report using tuned params */
    /* Pass the original (unsorted) luma buffer for percentile display
     * — but we already sorted it, that's fine, qsort is idempotent. */
    print_validation_report(px, n, luma_buf, luma_n, frames_done, meta);

    free(px);
    free(luma_buf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HDR metadata extraction
 * ═══════════════════════════════════════════════════════════════════════════*/
static void extract_from_stream(AVStream *st, HDRMeta *meta)
{
    const AVPacketSideData *sd = av_packet_side_data_get(
        st->codecpar->coded_side_data, st->codecpar->nb_coded_side_data,
        AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
    if (sd) {
        const AVMasteringDisplayMetadata *mdm =
            (const AVMasteringDisplayMetadata *)sd->data;
        if (mdm->has_luminance) {
            meta->peak_nits     = (float)av_q2d(mdm->max_luminance);
            meta->min_nits      = (float)av_q2d(mdm->min_luminance);
            meta->have_mastering = 1;
        }
    }
    sd = av_packet_side_data_get(st->codecpar->coded_side_data,
                                  st->codecpar->nb_coded_side_data,
                                  AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
    if (sd) {
        const AVContentLightMetadata *clm =
            (const AVContentLightMetadata *)sd->data;
        meta->max_cll  = (float)clm->MaxCLL;
        meta->max_fall = (float)clm->MaxFALL;
        meta->have_cll  = 1;
    }
}

static void extract_from_frame(AVFrame *frame, HDRMeta *meta)
{
    AVFrameSideData *sd = av_frame_get_side_data(
        frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd && !meta->have_mastering) {
        AVMasteringDisplayMetadata *mdm = (AVMasteringDisplayMetadata *)sd->data;
        if (mdm->has_luminance) {
            meta->peak_nits     = (float)av_q2d(mdm->max_luminance);
            meta->min_nits      = (float)av_q2d(mdm->min_luminance);
            meta->have_mastering = 1;
        }
    }
    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (sd && !meta->have_cll) {
        AVContentLightMetadata *clm = (AVContentLightMetadata *)sd->data;
        meta->max_cll  = (float)clm->MaxCLL;
        meta->max_fall = (float)clm->MaxFALL;
        meta->have_cll  = 1;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Scan video for HDR metadata
 * ═══════════════════════════════════════════════════════════════════════════*/
static int scan_video(const char *path, HDRMeta *meta)
{
    AVFormatContext *fctx = NULL;
    if (avformat_open_input(&fctx, path, NULL, NULL) < 0) {
        fprintf(stderr, "Cannot open '%s'\n", path); return -1;
    }
    if (avformat_find_stream_info(fctx, NULL) < 0) {
        fprintf(stderr, "Cannot find stream info\n");
        avformat_close_input(&fctx); return -1;
    }
    int vidx = av_find_best_stream(fctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vidx < 0) {
        fprintf(stderr, "No video stream\n");
        avformat_close_input(&fctx); return -1;
    }
    AVStream *vst = fctx->streams[vidx];
    printf("[info] Video: %s, %s, %dx%d\n",
           avcodec_get_name(vst->codecpar->codec_id),
           av_get_pix_fmt_name(vst->codecpar->format),
           vst->codecpar->width, vst->codecpar->height);

    extract_from_stream(vst, meta);

    if (!meta->have_mastering || !meta->have_cll) {
        const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
        if (!codec) { fprintf(stderr, "No decoder\n"); goto done; }
        AVCodecContext *cctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(cctx, vst->codecpar);
        cctx->thread_count = 2;
        if (avcodec_open2(cctx, codec, NULL) < 0) {
            fprintf(stderr, "Cannot open codec\n");
            avcodec_free_context(&cctx); goto done;
        }
        AVPacket *pkt = av_packet_alloc();
        AVFrame  *frm = av_frame_alloc();
        int frames = 0;
        while (av_read_frame(fctx, pkt) >= 0) {
            if (pkt->stream_index != vidx) { av_packet_unref(pkt); continue; }
            if (avcodec_send_packet(cctx, pkt) == 0) {
                while (avcodec_receive_frame(cctx, frm) == 0) {
                    extract_from_frame(frm, meta);
                    frames++;
                    av_frame_unref(frm);
                    if ((meta->have_mastering && meta->have_cll)
                        || frames >= MAX_SCAN_FRAMES) goto decode_done;
                }
            }
            av_packet_unref(pkt);
        }
decode_done:
        av_frame_free(&frm); av_packet_free(&pkt);
        avcodec_free_context(&cctx);
        printf("[info] Scanned %d frame(s) for HDR metadata\n", frames);
    }
done:
    avformat_close_input(&fctx);

    if (!meta->have_mastering) {
        fprintf(stderr, "[warn] No mastering metadata — using fallback "
                "%.0f / %.4f nits\n",
                (double)FALLBACK_PEAK_NITS, (double)FALLBACK_MIN_NITS);
        meta->peak_nits = FALLBACK_PEAK_NITS;
        meta->min_nits  = FALLBACK_MIN_NITS;
    }
    if (!meta->have_cll) {
        meta->max_cll  = meta->peak_nits;
        meta->max_fall = meta->peak_nits * 0.5f;
    }
    if (meta->max_cll > 1.0f && meta->max_cll < meta->peak_nits)
        meta->peak_nits = meta->max_cll;

    meta->sdr_white = SDR_REF_WHITE;   /* tuner may override this */
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3D LUT generation (.cube)
 * ═══════════════════════════════════════════════════════════════════════════*/
static int write_cube(const char *outpath, int N, const HDRMeta *meta)
{
    struct pl_tone_map_params tm = build_tm_params(meta);

    printf("[info] Peak luminance  : %.1f nits\n", (double)meta->peak_nits);
    printf("[info] Min  luminance  : %.4f nits\n", (double)meta->min_nits);
    printf("[info] MaxCLL/MaxFALL  : %.0f / %.0f nits\n",
           (double)meta->max_cll, (double)meta->max_fall);
    printf("[info] SDR white point : %.0f nits\n", (double)meta->sdr_white);
    printf("[info] LUT size        : %d³ (%d entries)\n", N, N*N*N);
    printf("[lut] Writing '%s' ...\n", outpath);

    FILE *f = fopen(outpath, "w");
    if (!f) { perror("fopen"); return -1; }

    fprintf(f, "# 3D LUT – HDR10 (BT.2020/PQ) → SDR (BT.709/sRGB)\n");
    fprintf(f, "# Generated by hdr10_to_sdr_lut (auto-tuned)\n");
    fprintf(f, "# Source peak : %.1f nits  SDR white: %.0f nits\n",
            (double)meta->peak_nits, (double)meta->sdr_white);
    fprintf(f, "# Tone mapping: BT.2390 EETF (libplacebo)\n");
    fprintf(f, "TITLE \"HDR10 to SDR\"\n");
    fprintf(f, "LUT_3D_SIZE %d\n", N);
    fprintf(f, "DOMAIN_MIN 0.0 0.0 0.0\n");
    fprintf(f, "DOMAIN_MAX 1.0 1.0 1.0\n\n");

    float step = 1.0f / (float)(N - 1);
    for (int bi = 0; bi < N; bi++) {
        for (int gi = 0; gi < N; gi++) {
            for (int ri = 0; ri < N; ri++) {
                float ro, go, bo;
                convert_sample((float)ri * step,
                               (float)gi * step,
                               (float)bi * step,
                               &ro, &go, &bo, &tm);
                fprintf(f, "%.6f %.6f %.6f\n",
                        (double)ro, (double)go, (double)bo);
            }
        }
        if ((bi & 7) == 0)
            printf("\r  progress: %3d / %d slices ...", bi+1, N);
        fflush(stdout);
    }
    printf("\r  progress: %3d / %d slices ... done\n", N, N);
    fclose(f);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════*/
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <input_hdr10_video> [output.cube] [lut_size=%d]\n"
            "\n"
            "  input_hdr10_video  – HDR10 video to analyse\n"
            "  output.cube        – output LUT file (default: output.cube)\n"
            "  lut_size           – grid size per axis (default: %d, max: 128)\n"
            "\n"
            "Apply with FFmpeg:\n"
            "  ffmpeg -i input.mkv -vf \"lut3d=output.cube\" \\\n"
            "         -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4\n",
            argv[0], DEFAULT_LUT_SIZE, DEFAULT_LUT_SIZE);
        return 1;
    }
    const char *infile  = argv[1];
    const char *outfile = (argc >= 3) ? argv[2] : "output.cube";
    int lut_size        = (argc >= 4) ? atoi(argv[3]) : DEFAULT_LUT_SIZE;
    if (lut_size < 2 || lut_size > 128) {
        fprintf(stderr, "lut_size must be 2–128\n"); return 1;
    }

    printf("═══════════════════════════════════════════\n");
    printf("  HDR10 → SDR 3D LUT generator\n");
    printf("  input  : %s\n", infile);
    printf("  output : %s\n", outfile);
    printf("═══════════════════════════════════════════\n");

    /* 1. Extract static HDR10 metadata */
    HDRMeta meta = {0};
    if (scan_video(infile, &meta) < 0) return 1;

    printf("\n[info] Metadata: peak=%.0f nits  min=%.3f nits  "
           "MaxCLL=%.0f  MaxFALL=%.0f\n",
           (double)meta.peak_nits, (double)meta.min_nits,
           (double)meta.max_cll,   (double)meta.max_fall);

    /* 2. Sample real pixels → binary-search auto-tune → validation report */
    validate_and_adjust(infile, &meta);

    /* 3. Write LUT with tuned parameters */
    if (write_cube(outfile, lut_size, &meta) < 0) return 1;

    printf("\n✓ Done.  Apply with:\n");
    printf("  ffmpeg -i \"%s\" \\\n", infile);
    printf("    -vf \"lut3d=%s\" \\\n", outfile);
    printf("    -c:v libx264 -crf 18 -pix_fmt yuv420p out_sdr.mp4\n\n");
    return 0;
}
