/*
 * hdr10_to_sdr_lut.c
 *
 * Analyses an HDR10 video with FFmpeg/libav, extracts static HDR10 metadata
 * (MaxCLL, MaxFALL, mastering display), then generates a 3D LUT (.cube file)
 * using libplacebo's tone-mapping evaluated per LUT grid point entirely on CPU.
 *
 * The .cube file can be used on any machine (no GPU) with:
 *   ffmpeg -i input.mkv -vf "lut3d=output.cube" -c:v libx264 -crf 18 out.mp4
 *
 * Usage:
 *   hdr10_to_sdr_lut <input_hdr10_video> [output.cube] [lut_size=65]
 *
 * Build: see Makefile
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── libav ─────────────────────────────────────────────────────────────── */
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
#include <libavutil/opt.h>

/* ── libplacebo ─────────────────────────────────────────────────────────── */
#include <libplacebo/colorspace.h>
#include <libplacebo/tone_mapping.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════*/
#define DEFAULT_LUT_SIZE      65
#define PQ_MAX_NITS        10000.0f   /* ST.2084 reference peak            */
#define SDR_REF_WHITE        203.0f   /* PQ reference white (nits)         */
#define SDR_PEAK_NITS        100.0f   /* Consumer SDR display peak         */
#define FALLBACK_PEAK_NITS  1000.0f   /* Used when metadata is missing     */
#define FALLBACK_MIN_NITS    0.005f
#define MAX_SCAN_FRAMES       500     /* Frames decoded to hunt for metadata*/

/* ── Validation constants ── */
#define VAL_FRAMES          20        /* Number of frames to sample         */
#define VAL_PIXEL_STEP      16        /* Sample every Nth pixel (x and y)   */
#define VAL_HIST_BINS       12        /* Histogram buckets for output luma  */
#define VAL_MAX_SAMPLES   2000000     /* Max luma samples for percentiles   */
#define VAL_WHITE_CLIP_THRESH 0.997f
#define VAL_BLACK_CLIP_THRESH 0.003f

/* ═══════════════════════════════════════════════════════════════════════════
 * HDR10 metadata bag
 * ═══════════════════════════════════════════════════════════════════════════*/
typedef struct {
    float peak_nits;          /* MaxCLL, or mastering display max          */
    float min_nits;           /* Mastering display min luminance           */
    float max_cll;
    float max_fall;
    int   have_mastering;
    int   have_cll;
} HDRMeta;

/* ═══════════════════════════════════════════════════════════════════════════
 * Transfer functions  (all inline, no external dep)
 * ═══════════════════════════════════════════════════════════════════════════*/

/* PQ EOTF: signal [0,1] → absolute luminance [0, 10000] nits */
static float pq_eotf(float x)
{
    if (x <= 0.0f) return 0.0f;
    const float m1i = 16384.0f / 2610.0f;
    const float m2i = 4096.0f  / (2523.0f * 128.0f);
    const float c1  = 3424.0f  / 4096.0f;
    const float c2  = 2413.0f  * 32.0f / 4096.0f;
    const float c3  = 2392.0f  * 32.0f / 4096.0f;
    float xm = powf(x, m2i);
    float num = fmaxf(xm - c1, 0.0f);
    float den = c2 - c3 * xm;
    return (den <= 0.0f) ? 0.0f : PQ_MAX_NITS * powf(num / den, m1i);
}

/* PQ OETF: absolute luminance [0, 10000] nits → signal [0,1]
 * (kept for reference; not called during LUT generation) */
__attribute__((unused))
static float pq_oetf(float y)
{
    if (y <= 0.0f) return 0.0f;
    const float m1 = 2610.0f  / 16384.0f;
    const float m2 = 2523.0f  * 128.0f / 4096.0f;
    const float c1 = 3424.0f  / 4096.0f;
    const float c2 = 2413.0f  * 32.0f  / 4096.0f;
    const float c3 = 2392.0f  * 32.0f  / 4096.0f;
    float ym = powf(y / PQ_MAX_NITS, m1);
    return powf((c1 + c2 * ym) / (1.0f + c3 * ym), m2);
}

/* sRGB OETF: linear [0,1] → gamma-encoded [0,1] */
static float srgb_oetf(float x)
{
    x = fmaxf(x, 0.0f);
    if (x <= 0.0031308f) return 12.92f * x;
    return 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BT.2020 → BT.709 linear colour matrix
 * (derived via XYZ intermediate, D65 white point)
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
 * Pixel extraction: 10-bit YCbCr → PQ-encoded RGB [0,1]
 *
 * Supports YUV420P10LE, YUV422P10LE, YUV444P10LE.
 * Assumes BT.2020 limited-range (standard for HDR10).
 *
 * Returns 0 on success, -1 if format unsupported.
 * ═══════════════════════════════════════════════════════════════════════════*/
static int frame_get_pq_rgb(const AVFrame *frm, int x, int y,
                             float *r, float *g, float *b)
{
    int fmt = frm->format;
    if (fmt != AV_PIX_FMT_YUV420P10LE &&
        fmt != AV_PIX_FMT_YUV422P10LE &&
        fmt != AV_PIX_FMT_YUV444P10LE)
        return -1;

    int Y_stride  = frm->linesize[0] / 2;
    int UV_stride = frm->linesize[1] / 2;

    /* Chroma pixel coordinates based on subsampling */
    int cx = x, cy = y;
    if (fmt == AV_PIX_FMT_YUV420P10LE) { cx = x >> 1; cy = y >> 1; }
    else if (fmt == AV_PIX_FMT_YUV422P10LE) { cx = x >> 1; }

    const uint16_t *Yp  = (const uint16_t *)frm->data[0];
    const uint16_t *Cbp = (const uint16_t *)frm->data[1];
    const uint16_t *Crp = (const uint16_t *)frm->data[2];

    int Yv  = Yp [y  * Y_stride  + x];
    int Cbv = Cbp[cy * UV_stride + cx];
    int Crv = Crp[cy * UV_stride + cx];

    /* BT.2020 10-bit limited-range normalisation:
     *   Y  : [64, 940] → [0, 1]
     *   Cb,Cr: [64, 960] → [-0.5, +0.5]  (offset 512 = zero) */
    float Yn  = ((float)Yv  -  64.0f) / 876.0f;
    float Pbn = ((float)Cbv - 512.0f) / 896.0f;
    float Prn = ((float)Crv - 512.0f) / 896.0f;

    /* ITU-R BT.2020 YCbCr → PQ-encoded R'G'B'
     *   Kr=0.2627, Kg=0.6780, Kb=0.0593
     *   R = Y + 1.4746*Pr
     *   B = Y + 1.8814*Pb
     *   G = (Y - Kr*R - Kb*B) / Kg                */
    float R = Yn + 1.4746f * Prn;
    float B = Yn + 1.8814f * Pbn;
    float G = (Yn - 0.2627f * R - 0.0593f * B) / 0.6780f;

    *r = fmaxf(0.0f, fminf(1.0f, R));
    *g = fmaxf(0.0f, fminf(1.0f, G));
    *b = fmaxf(0.0f, fminf(1.0f, B));
    return 0;
}

/* qsort comparator for float ascending */
static int cmp_float_asc(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * libplacebo tone-mapping wrapper (CPU path)
 * ═══════════════════════════════════════════════════════════════════════════*/
static struct pl_tone_map_params build_tm_params(const HDRMeta *hdr)
{
    struct pl_tone_map_params p = {0};
    p.function   = &pl_tone_map_bt2390;
    p.input_max  = hdr->peak_nits;
    p.input_min  = hdr->min_nits;
    p.output_max = SDR_REF_WHITE;
    p.output_min = 0.0f;

    p.hdr.max_luma = hdr->peak_nits;
    p.hdr.min_luma = hdr->min_nits;
    p.hdr.max_cll  = hdr->max_cll;
    p.hdr.max_fall = hdr->max_fall;

    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Core per-sample colour pipeline
 *
 *  Input  – (r,g,b) as PQ-encoded signal in [0, 1]  (HDR10 / BT.2020)
 *  Output – (r,g,b) as sRGB gamma-encoded in [0, 1]  (SDR / BT.709)
 * ═══════════════════════════════════════════════════════════════════════════*/
static void convert_sample(float ri, float gi, float bi,
                           float *ro, float *go, float *bo,
                           const struct pl_tone_map_params *tm)
{
    /* 1. PQ EOTF → absolute scene-linear BT.2020 (nits) */
    float lr = pq_eotf(ri);
    float lg = pq_eotf(gi);
    float lb = pq_eotf(bi);

    /* 2. BT.2020 → BT.709 (still linear, nits) */
    float r709, g709, b709;
    mat3_mul(M_2020_709, lr, lg, lb, &r709, &g709, &b709);

    /* 3. Luminance of the BT.709 signal (nits) */
    float Y = 0.2126f*r709 + 0.7152f*g709 + 0.0722f*b709;

    /* 4. Tone-map luminance with libplacebo  (nits → nits, SDR range) */
    float Y_sdr = (Y > 0.001f) ? pl_tone_map_sample(Y, tm) : 0.0f;

    /* 5. Scale RGB by the luminance ratio (preserves hue/saturation) */
    float scale = (Y > 0.001f) ? (Y_sdr / Y) : 0.0f;

    /* 6. Apply luminance-preserving highlight desaturation */
    float sdr_luma_norm = Y_sdr / tm->output_max;
    float desat = fmaxf(0.0f, (sdr_luma_norm - 0.8f) / 0.2f);

    float rn = r709 * scale / tm->output_max;
    float gn = g709 * scale / tm->output_max;
    float bn = b709 * scale / tm->output_max;

    rn = rn + desat * (sdr_luma_norm - rn);
    gn = gn + desat * (sdr_luma_norm - gn);
    bn = bn + desat * (sdr_luma_norm - bn);

    /* 7. Clamp + sRGB gamma */
    *ro = srgb_oetf(fminf(fmaxf(rn, 0.0f), 1.0f));
    *go = srgb_oetf(fminf(fmaxf(gn, 0.0f), 1.0f));
    *bo = srgb_oetf(fminf(fmaxf(bn, 0.0f), 1.0f));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HDR metadata extraction via libavcodec side-data
 * ═══════════════════════════════════════════════════════════════════════════*/
static void extract_from_stream(AVStream *st, HDRMeta *meta)
{
    const AVPacketSideData *sd = av_packet_side_data_get(
                                    st->codecpar->coded_side_data,
                                    st->codecpar->nb_coded_side_data,
                                    AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
    if (sd) {
        const AVMasteringDisplayMetadata *mdm =
            (const AVMasteringDisplayMetadata *)sd->data;
        if (mdm->has_luminance) {
            meta->peak_nits  = (float)(av_q2d(mdm->max_luminance));
            meta->min_nits   = (float)(av_q2d(mdm->min_luminance));
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
        meta->have_cll = 1;
    }
}

static void extract_from_frame(AVFrame *frame, HDRMeta *meta)
{
    AVFrameSideData *sd = av_frame_get_side_data(frame,
                              AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd && !meta->have_mastering) {
        AVMasteringDisplayMetadata *mdm = (AVMasteringDisplayMetadata *)sd->data;
        if (mdm->has_luminance) {
            meta->peak_nits  = (float)av_q2d(mdm->max_luminance);
            meta->min_nits   = (float)av_q2d(mdm->min_luminance);
            meta->have_mastering = 1;
        }
    }

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (sd && !meta->have_cll) {
        AVContentLightMetadata *clm = (AVContentLightMetadata *)sd->data;
        meta->max_cll  = (float)clm->MaxCLL;
        meta->max_fall = (float)clm->MaxFALL;
        meta->have_cll = 1;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Scan video for HDR metadata
 * ═══════════════════════════════════════════════════════════════════════════*/
static int scan_video(const char *path, HDRMeta *meta)
{
    AVFormatContext *fctx = NULL;

    if (avformat_open_input(&fctx, path, NULL, NULL) < 0) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        return -1;
    }
    if (avformat_find_stream_info(fctx, NULL) < 0) {
        fprintf(stderr, "Cannot find stream info\n");
        avformat_close_input(&fctx);
        return -1;
    }

    int vidx = av_find_best_stream(fctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vidx < 0) {
        fprintf(stderr, "No video stream found\n");
        avformat_close_input(&fctx);
        return -1;
    }
    AVStream *vst = fctx->streams[vidx];

    printf("[info] Video: %s, %s, %dx%d\n",
           avcodec_get_name(vst->codecpar->codec_id),
           av_get_pix_fmt_name(vst->codecpar->format),
           vst->codecpar->width, vst->codecpar->height);

    extract_from_stream(vst, meta);

    if (!meta->have_mastering || !meta->have_cll) {
        const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
        if (!codec) {
            fprintf(stderr, "No decoder for codec\n");
            goto done;
        }
        AVCodecContext *cctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(cctx, vst->codecpar);
        cctx->thread_count = 2;
        if (avcodec_open2(cctx, codec, NULL) < 0) {
            fprintf(stderr, "Cannot open codec\n");
            avcodec_free_context(&cctx);
            goto done;
        }

        AVPacket *pkt  = av_packet_alloc();
        AVFrame  *frm  = av_frame_alloc();
        int frames = 0;

        while (av_read_frame(fctx, pkt) >= 0) {
            if (pkt->stream_index != vidx) { av_packet_unref(pkt); continue; }
            if (avcodec_send_packet(cctx, pkt) == 0) {
                while (avcodec_receive_frame(cctx, frm) == 0) {
                    extract_from_frame(frm, meta);
                    frames++;
                    av_frame_unref(frm);
                    if ((meta->have_mastering && meta->have_cll)
                        || frames >= MAX_SCAN_FRAMES)
                        goto decode_done;
                }
            }
            av_packet_unref(pkt);
        }
decode_done:
        av_frame_free(&frm);
        av_packet_free(&pkt);
        avcodec_free_context(&cctx);
        printf("[info] Scanned %d frame(s) for HDR metadata\n", frames);
    }

done:
    avformat_close_input(&fctx);

    if (!meta->have_mastering) {
        fprintf(stderr,
                "[warn] No mastering display metadata found — "
                "assuming peak=%.0f nits, min=%.4f nits\n",
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

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * LUT Validation Pass
 *
 * Decodes VAL_FRAMES frames spread across the video, samples every
 * VAL_PIXEL_STEP-th pixel, and:
 *   1. Measures real scene luminance (nits) percentiles.
 *   2. Applies the full LUT pipeline to sampled pixels.
 *   3. Reports output histogram, black/white clipping percentages, mean luma.
 *   4. Auto-adjusts meta->peak_nits if the measured p99.9 is significantly
 *      below the declared metadata value (inaccurate HDR10 metadata).
 * ═══════════════════════════════════════════════════════════════════════════*/
static void validate_and_adjust(const char *path, HDRMeta *meta)
{
    printf("\n[validate] Opening video for LUT accuracy validation...\n");

    AVFormatContext *fctx = NULL;
    if (avformat_open_input(&fctx, path, NULL, NULL) < 0) {
        fprintf(stderr, "[validate] Cannot open video — skipping validation\n");
        return;
    }
    avformat_find_stream_info(fctx, NULL);
    int vidx = av_find_best_stream(fctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vidx < 0) { avformat_close_input(&fctx); return; }
    AVStream *vst = fctx->streams[vidx];

    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) { avformat_close_input(&fctx); return; }

    AVCodecContext *cctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(cctx, vst->codecpar);
    cctx->thread_count  = 4;
    cctx->skip_frame    = AVDISCARD_NONKEY;   /* keyframes only → fast  */
    if (avcodec_open2(cctx, codec, NULL) < 0) {
        avcodec_free_context(&cctx);
        avformat_close_input(&fctx);
        return;
    }

    /* Build tone-map params from the metadata as-is (pre-adjustment) */
    struct pl_tone_map_params tm = build_tm_params(meta);

    /* Storage for luminance samples (for percentile computation) */
    float *luma_buf = malloc(VAL_MAX_SAMPLES * sizeof(float));
    if (!luma_buf) {
        avcodec_free_context(&cctx);
        avformat_close_input(&fctx);
        return;
    }

    long   total_px   = 0;
    long   black_clip = 0;
    long   white_clip = 0;
    long   luma_n     = 0;
    double sum_out    = 0.0;
    long   hist[VAL_HIST_BINS] = {0};
    int    warned_fmt = 0;
    int    frames_done = 0;

    int64_t dur = fctx->duration;   /* in AV_TIME_BASE units */

    AVPacket *pkt = av_packet_alloc();
    AVFrame  *frm = av_frame_alloc();

    for (int fi = 0; fi < VAL_FRAMES; fi++) {
        /* Seek to evenly-spaced positions across the video              */
        if (dur > 0) {
            double frac = (VAL_FRAMES > 1)
                          ? (double)fi / (VAL_FRAMES - 1) : 0.0;
            int64_t target = av_rescale_q((int64_t)(frac * dur),
                                          AV_TIME_BASE_Q, vst->time_base);
            av_seek_frame(fctx, vidx, target, AVSEEK_FLAG_BACKWARD);
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
                                "[validate] Pixel format '%s' not supported "
                                "(need 10-bit YUV) — skipping validation\n",
                                av_get_pix_fmt_name(fmt));
                            warned_fmt = 1;
                        }
                        av_frame_unref(frm);
                        continue;
                    }

                    int w = frm->width, h = frm->height;
                    for (int py = 0; py < h; py += VAL_PIXEL_STEP) {
                        for (int px = 0; px < w; px += VAL_PIXEL_STEP) {
                            float ri, gi, bi;
                            if (frame_get_pq_rgb(frm, px, py,
                                                 &ri, &gi, &bi) < 0)
                                continue;

                            /* ── Scene luminance in nits (BT.709 gamut) ── */
                            float lr = pq_eotf(ri);
                            float lg = pq_eotf(gi);
                            float lb = pq_eotf(bi);
                            float r9, g9, b9;
                            mat3_mul(M_2020_709, lr, lg, lb, &r9, &g9, &b9);
                            float Y_in = 0.2126f*r9 + 0.7152f*g9 + 0.0722f*b9;

                            if (luma_n < VAL_MAX_SAMPLES)
                                luma_buf[luma_n++] = Y_in;

                            /* ── Apply full LUT pipeline ──────────────── */
                            float ro, go, bo;
                            convert_sample(ri, gi, bi, &ro, &go, &bo, &tm);
                            float Y_out = 0.2126f*ro + 0.7152f*go + 0.0722f*bo;

                            sum_out += Y_out;
                            total_px++;

                            if (ro < VAL_BLACK_CLIP_THRESH &&
                                go < VAL_BLACK_CLIP_THRESH &&
                                bo < VAL_BLACK_CLIP_THRESH)
                                black_clip++;

                            if (ro > VAL_WHITE_CLIP_THRESH ||
                                go > VAL_WHITE_CLIP_THRESH ||
                                bo > VAL_WHITE_CLIP_THRESH)
                                white_clip++;

                            int bin = (int)(Y_out * VAL_HIST_BINS);
                            if (bin < 0) bin = 0;
                            if (bin >= VAL_HIST_BINS) bin = VAL_HIST_BINS - 1;
                            hist[bin]++;
                        }
                    }
                    got = 1;
                    frames_done++;
                    av_frame_unref(frm);
                }
            }
            av_packet_unref(pkt);
        }

        printf("\r[validate] Sampled frame %2d / %d ...",
               fi + 1, VAL_FRAMES);
        fflush(stdout);
    }
    printf("\n");

    av_frame_free(&frm);
    av_packet_free(&pkt);
    avcodec_free_context(&cctx);
    avformat_close_input(&fctx);

    if (total_px == 0 || luma_n == 0) {
        fprintf(stderr, "[validate] No pixels sampled — skipping report\n");
        free(luma_buf);
        return;
    }

    /* ── Percentiles of scene luminance ──────────────────────────────── */
    qsort(luma_buf, luma_n, sizeof(float), cmp_float_asc);
    float p50  = luma_buf[(long)(luma_n * 0.500)];
    float p90  = luma_buf[(long)(luma_n * 0.900)];
    float p99  = luma_buf[(long)(luma_n * 0.990)];
    float p999 = luma_buf[(long)(luma_n * 0.999)];
    float pmax = luma_buf[luma_n - 1];
    free(luma_buf);

    double mean_out    = sum_out / total_px;
    double pct_black   = 100.0 * black_clip / total_px;
    double pct_white   = 100.0 * white_clip / total_px;

    /* ── Print report ─────────────────────────────────────────────────── */
    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│                    LUT Validation Report                     │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│  Frames sampled : %-4d    Pixels sampled : %-14ld  │\n",
           frames_done, total_px);
    printf("│                                                              │\n");
    printf("│  Scene luminance (measured from real pixels, BT.709 nits):  │\n");
    printf("│    Median (p50)    : %8.2f nits                         │\n", (double)p50);
    printf("│    p90             : %8.2f nits                         │\n", (double)p90);
    printf("│    p99             : %8.2f nits                         │\n", (double)p99);
    printf("│    p99.9           : %8.2f nits  ← effective peak       │\n", (double)p999);
    printf("│    Absolute max    : %8.2f nits                         │\n", (double)pmax);
    printf("│    Metadata peak   : %8.2f nits                         │\n", (double)meta->peak_nits);
    printf("│                                                              │\n");
    printf("│  LUT output statistics (sRGB gamma-encoded signal):         │\n");
    printf("│    Mean luma       :   %.4f  (%.1f%% of SDR range)       │\n",
           mean_out, mean_out * 100.0);
    printf("│    Black clipping  :   %5.2f%% of pixels clipped to 0      │\n", pct_black);
    printf("│    White clipping  :   %5.2f%% of pixels clipped to 1      │\n", pct_white);
    printf("│                                                              │\n");
    printf("│  Output luma histogram (sRGB gamma, 0%%..100%%):             │\n");

    /* Draw ASCII histogram bars */
    for (int b = 0; b < VAL_HIST_BINS; b++) {
        float pct = 100.0f * (float)hist[b] / (float)total_px;
        int bar = (int)(pct / 2.0f);    /* 2% per █, max 25 chars */
        if (bar > 25) bar = 25;
        printf("│   %3.0f%%–%3.0f%%  [",
               100.0f * b / VAL_HIST_BINS,
               100.0f * (b + 1) / VAL_HIST_BINS);
        for (int k = 0; k < bar; k++)       printf("█");
        for (int k = bar; k < 25; k++)      printf(" ");
        printf("] %5.1f%%  │\n", (double)pct);
    }
    printf("└──────────────────────────────────────────────────────────────┘\n\n");

    /* ── Diagnostics ──────────────────────────────────────────────────── */
    int issues = 0;

    if (pct_white > 2.0) {
        fprintf(stderr,
            "[validate] WARNING: %.1f%% of pixels clip to white. "
            "Tone mapping is overexposed (peak_nits too low?).\n", pct_white);
        issues++;
    }
    if (pct_black > 10.0) {
        fprintf(stderr,
            "[validate] WARNING: %.1f%% of pixels clip to black. "
            "Shadow detail may be crushed.\n", pct_black);
        issues++;
    }
    if (mean_out < 0.08) {
        fprintf(stderr,
            "[validate] WARNING: mean output luma %.3f is very low — "
            "image will appear too dark.\n", mean_out);
        issues++;
    }
    if (mean_out > 0.70) {
        fprintf(stderr,
            "[validate] WARNING: mean output luma %.3f is very high — "
            "image may appear washed out.\n", mean_out);
        issues++;
    }

    /* ── Auto-adjust peak_nits if metadata is significantly overstated ── */
    if (p999 > 1.0f && p999 < meta->peak_nits * 0.80f) {
        float adjusted = p999 * 1.05f;   /* 5% headroom above p99.9 */
        printf("[validate] Metadata peak = %.0f nits, but measured p99.9 = %.0f nits.\n",
               (double)meta->peak_nits, (double)p999);
        printf("[validate] Metadata overstates peak by %.0f%%. "
               "Auto-adjusting to %.0f nits.\n",
               100.0 * (meta->peak_nits - p999) / meta->peak_nits,
               (double)adjusted);
        meta->peak_nits = adjusted;
        /* Also update max_cll to match if it was just following peak */
        if (meta->max_cll > meta->peak_nits)
            meta->max_cll = meta->peak_nits;
        issues++;   /* counted so summary isn't "OK" */
        printf("[validate] LUT will be regenerated with adjusted peak.\n\n");
    }

    if (issues == 0)
        printf("[validate] ✓  LUT looks accurate — no significant issues found.\n\n");
    else
        printf("[validate] ⚠   %d issue(s) flagged above.\n\n", issues);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3D LUT generation
 *
 * .cube iteration order: R fastest (inner), B slowest (outer) — per spec.
 * Input domain : [0,1]³  PQ-encoded HDR10 (BT.2020)
 * Output domain: [0,1]³  sRGB-encoded SDR  (BT.709)
 * ═══════════════════════════════════════════════════════════════════════════*/
static int write_cube(const char *outpath, int N, const HDRMeta *meta)
{
    struct pl_tone_map_params tm = build_tm_params(meta);

    printf("[info] Peak luminance : %.1f nits\n", (double)meta->peak_nits);
    printf("[info] Min  luminance : %.4f nits\n", (double)meta->min_nits);
    printf("[info] MaxCLL / MaxFALL: %.0f / %.0f nits\n",
           (double)meta->max_cll, (double)meta->max_fall);
    printf("[info] LUT size       : %d³ (%d entries)\n", N, N*N*N);
    printf("[info] Tone mapping   : BT.2390 EETF (via libplacebo)\n");
    printf("[lut] Writing '%s' ...\n", outpath);

    FILE *f = fopen(outpath, "w");
    if (!f) { perror("fopen"); return -1; }

    fprintf(f, "# 3D LUT – HDR10 (BT.2020/PQ) → SDR (BT.709/sRGB)\n");
    fprintf(f, "# Generated by hdr10_to_sdr_lut\n");
    fprintf(f, "# Source peak : %.1f nits\n", (double)meta->peak_nits);
    fprintf(f, "# Tone mapping: BT.2390 EETF (libplacebo)\n");
    fprintf(f, "TITLE \"HDR10 to SDR\"\n");
    fprintf(f, "LUT_3D_SIZE %d\n", N);
    fprintf(f, "DOMAIN_MIN 0.0 0.0 0.0\n");
    fprintf(f, "DOMAIN_MAX 1.0 1.0 1.0\n\n");

    float step = 1.0f / (float)(N - 1);

    for (int bi = 0; bi < N; bi++) {
        for (int gi = 0; gi < N; gi++) {
            for (int ri = 0; ri < N; ri++) {
                float r_in = (float)ri * step;
                float g_in = (float)gi * step;
                float b_in = (float)bi * step;

                float r_out, g_out, b_out;
                convert_sample(r_in, g_in, b_in,
                               &r_out, &g_out, &b_out, &tm);

                fprintf(f, "%.6f %.6f %.6f\n",
                        (double)r_out, (double)g_out, (double)b_out);
            }
        }
        if ((bi & 7) == 0)
            printf("\r  progress: %3d / %d slices ...", bi + 1, N);
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
            "Apply with FFmpeg (CPU only, no GPU needed):\n"
            "  ffmpeg -i input.mkv -vf \"lut3d=output.cube\" \\\n"
            "         -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4\n",
            argv[0], DEFAULT_LUT_SIZE, DEFAULT_LUT_SIZE);
        return 1;
    }

    const char *infile  = argv[1];
    const char *outfile = (argc >= 3) ? argv[2] : "output.cube";
    int lut_size        = (argc >= 4) ? atoi(argv[3]) : DEFAULT_LUT_SIZE;

    if (lut_size < 2 || lut_size > 128) {
        fprintf(stderr, "lut_size must be between 2 and 128\n");
        return 1;
    }

    printf("═══════════════════════════════════════════\n");
    printf("  HDR10 → SDR 3D LUT generator\n");
    printf("  input  : %s\n", infile);
    printf("  output : %s\n", outfile);
    printf("═══════════════════════════════════════════\n");

    /* Step 1: extract static HDR10 metadata */
    HDRMeta meta = {0};
    if (scan_video(infile, &meta) < 0) return 1;

    /* Step 2: validate against real pixel data, auto-adjust if needed */
    validate_and_adjust(infile, &meta);

    /* Step 3: generate LUT with (possibly corrected) parameters */
    if (write_cube(outfile, lut_size, &meta) < 0) return 1;

    printf("\n✓ Done.  Apply with:\n");
    printf("  ffmpeg -i \"%s\" \\\n", infile);
    printf("    -vf \"lut3d=%s\" \\\n", outfile);
    printf("    -c:v libx264 -crf 18 -pix_fmt yuv420p out_sdr.mp4\n\n");
    return 0;
}
