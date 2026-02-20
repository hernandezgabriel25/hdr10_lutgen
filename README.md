# hdr10_to_sdr_lut

Analyses an HDR10 video and writes a 3D LUT (`.cube`) file for HDR10 → SDR
conversion. The LUT can then be applied on **any machine, GPU-free**, using
FFmpeg's built-in `lut3d` filter.

## How it works

```
HDR10 PQ signal [0,1]                 (BT.2020 primaries)
        │
        ▼ PQ EOTF (ST.2084)
Linear scene light [0, peak nits]     (BT.2020 linear)
        │
        ▼ 3×3 colour matrix
Linear scene light [0, peak nits]     (BT.709 primaries)
        │
        ▼ BT.2390 EETF tone map       (via libplacebo – CPU path)
          peak nits → 203 nit reference white
Linear SDR light  [0, 1]              (BT.709)
        │
        ▼ sRGB OETF (gamma 2.4)
sRGB signal       [0, 1]
```

Static HDR10 metadata (MaxCLL, MaxFALL, mastering display luminance) is
extracted directly from the video stream side-data, with a frame-scan fallback.

## Build

### Shared FFmpeg + libplacebo (typical distro or custom build with pkg-config)

```sh
# Make sure PKG_CONFIG_PATH points to your build
export PKG_CONFIG_PATH=/path/to/ffmpeg/lib/pkgconfig:/path/to/placebo/lib/pkgconfig
make
```

### Local FFmpeg static build (no pkg-config)

```sh
make FFMPEG_PREFIX=/path/to/ffmpeg-build PLACEBO_PREFIX=/path/to/placebo-build
```

The Makefile will pass the right `-I` / `-L` flags automatically.

If you built FFmpeg as fully static (`--enable-static --disable-shared`), you
may need to add extra `-l` flags (see the comment block in the Makefile).

## Usage

```sh
./hdr10_to_sdr_lut  <input_hdr10_video>  [output.cube]  [lut_size]
```

| Argument | Default | Notes |
|---|---|---|
| `input_hdr10_video` | — | Any format FFmpeg can demux/decode |
| `output.cube` | `output.cube` | Standard Adobe/DaVinci .cube format |
| `lut_size` | `65` | Grid points per axis (2–128). 33 = fast, 65 = quality |

### Apply the LUT with FFmpeg (CPU, no GPU)

```sh
ffmpeg -i input.mkv \
  -vf "lut3d=output.cube" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  out_sdr.mp4
```

Full recommended pipeline (also strips lingering HDR metadata):

```sh
ffmpeg -i input.mkv \
  -vf "lut3d=output.cube,
       setparams=colorspace=bt709:color_primaries=bt709:color_trc=iec61966-2-1,
       format=yuv420p" \
  -c:v libx264 -crf 18 -movflags +faststart \
  out_sdr.mp4
```

## Notes

- The program scans **up to 500 frames** searching for HDR10 side-data. Most
  standards-compliant files embed the metadata on the first frame or in the
  stream header, so scanning completes quickly.
- If no metadata is found, a safe fallback of **1000 nits peak / 0.005 nits
  black** is assumed and a warning is printed.
- LUT generation is purely CPU-bound and typically takes a few seconds for a
  65³ grid.
- The `.cube` format is compatible with: FFmpeg `lut3d`, DaVinci Resolve,
  Premiere Pro, Nuke, and any other tool that supports the Adobe .cube spec.

## Dependencies

| Library | Minimum version | Notes |
|---|---|---|
| libavformat | 60+ (FFmpeg 6.x) | Demuxing |
| libavcodec  | 60+              | Decoding (for frame-scan fallback) |
| libavutil   | 58+              | Side-data types |
| libplacebo  | 4.x+             | `pl_tone_map_sample`, `pl_tone_map_bt2390` |
