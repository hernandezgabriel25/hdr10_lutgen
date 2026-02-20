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
 * libplacebo tone-mapping wrapper (CPU path)
 *
 * pl_tone_map_sample(x, params) maps a single luminance value.
 * x        – input luminance in nits  [0, input_max]
 * returns  – output luminance in nits [0, output_max]
 * ═══════════════════════════════════════════════════════════════════════════*/
static struct pl_tone_map_params build_tm_params(const HDRMeta *hdr)
{
    struct pl_tone_map_params p = {0};
    p.function   = &pl_tone_map_bt2390;   /* ITU-R BT.2390 EETF             */
    p.input_max  = hdr->peak_nits;
    p.input_min  = hdr->min_nits;
    p.output_max = SDR_REF_WHITE;          /* 203 nit reference white        */
    p.output_min = 0.0f;

    /* Pass static HDR10 metadata so the function can adapt */
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

    /* 6. Apply luminance-preserving highlight desaturation
     *    (blend towards white in the SDR range – matches BT.2390 intent) */
    float sdr_luma_norm = Y_sdr / tm->output_max;   /* normalised [0,1] */
    float desat = fmaxf(0.0f, (sdr_luma_norm - 0.8f) / 0.2f); /* 0..1 */

    float rn = r709 * scale / tm->output_max;
    float gn = g709 * scale / tm->output_max;
    float bn = b709 * scale / tm->output_max;

    /* Blend towards SDR luma white on overexposed highlights */
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
    /* Mastering display metadata from stream side-data */
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

    /* MaxCLL / MaxFALL */
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
    /* Mastering display */
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

    /* MaxCLL */
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

    /* Find first video stream */
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

    /* Try to grab metadata from stream side-data first */
    extract_from_stream(vst, meta);

    /* Open decoder and scan frames only if still missing metadata */
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

    /* Apply fallbacks */
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
    /* Use MaxCLL as effective peak if it's lower (more accurate) */
    if (meta->max_cll > 1.0f && meta->max_cll < meta->peak_nits)
        meta->peak_nits = meta->max_cll;

    return 0;
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

    /* Header */
    fprintf(f, "# 3D LUT – HDR10 (BT.2020/PQ) → SDR (BT.709/sRGB)\n");
    fprintf(f, "# Generated by hdr10_to_sdr_lut\n");
    fprintf(f, "# Source peak : %.1f nits\n", (double)meta->peak_nits);
    fprintf(f, "# Tone mapping: BT.2390 EETF (libplacebo)\n");
    fprintf(f, "TITLE \"HDR10 to SDR\"\n");
    fprintf(f, "LUT_3D_SIZE %d\n", N);
    fprintf(f, "DOMAIN_MIN 0.0 0.0 0.0\n");
    fprintf(f, "DOMAIN_MAX 1.0 1.0 1.0\n\n");

    float step = 1.0f / (float)(N - 1);

    /* Outer=B, middle=G, inner=R  (standard .cube order) */
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
        /* Progress every 8 B slices */
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

    HDRMeta meta = {0};
    if (scan_video(infile, &meta) < 0) return 1;
    if (write_cube(outfile, lut_size, &meta) < 0)  return 1;

    printf("\n✓ Done.  Apply with:\n");
    printf("  ffmpeg -i \"%s\" \\\n", infile);
    printf("    -vf \"lut3d=%s\" \\\n", outfile);
    printf("    -c:v libx264 -crf 18 -pix_fmt yuv420p out_sdr.mp4\n\n");
    return 0;
}
