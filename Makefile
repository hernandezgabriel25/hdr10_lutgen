# ─────────────────────────────────────────────────────────────────────────────
# Makefile – hdr10_to_sdr_lut
#
# TYPICAL USAGE with a local static FFmpeg + libplacebo build:
#
#   PKG_CONFIG_PATH=../ffbuild/prefix/lib/pkgconfig:../libplacebo/build/src/meson-info \
#       make FFMPEG_PREFIX=../ffbuild/prefix PLACEBO_PREFIX=../libplacebo/build/src
#
# If pkg-config is unavailable, set MANUAL_LIBS (see bottom) and run:
#   make MANUAL=1
# ─────────────────────────────────────────────────────────────────────────────

TARGET  := hdr10_to_sdr_lut
SRCS    := hdr10_to_sdr_lut.c
CC      := gcc

# ── Optional prefix overrides ──────────────────────────────────────────────
ifdef FFMPEG_PREFIX
  EXTRA_CFLAGS  += -I$(FFMPEG_PREFIX)/include
  EXTRA_LDFLAGS += -L$(FFMPEG_PREFIX)/lib
endif

ifdef PLACEBO_PREFIX
  EXTRA_CFLAGS  += -I$(PLACEBO_PREFIX)/include
  EXTRA_LDFLAGS += -L$(PLACEBO_PREFIX)/lib
endif

# ── pkg-config (--static to get ALL transitive deps) ──────────────────────
# For a static FFmpeg build you MUST use --static; it adds Libs.private entries
# like -lva -ldrm -lX11 -lssl -lcrypto -lpthread etc.
PKGS := libavformat libavcodec libavutil libswresample libplacebo

PKG_CFLAGS  := $(shell pkg-config --cflags          $(PKGS) 2>/dev/null)
ifeq ($(MANUAL),)
  PKG_LIBS  := $(shell pkg-config --libs --static    $(PKGS) 2>/dev/null)
endif

# ── Manual fallback ────────────────────────────────────────────────────────
# Use `make MANUAL=1` if pkg-config is not set up.  Adjust the -l list to
# match whatever your FFmpeg was compiled with (check ffbuild/config.mak or
# run:  pkg-config --static --libs libavformat libavcodec libavutil)
ifeq ($(MANUAL),1)
  PKG_LIBS := \
    -lavformat -lavcodec -lavutil -lswresample \
    -lplacebo \
    -lva -lva-drm -lva-x11 -ldrm \
    -lX11 \
    -lssl -lcrypto \
    -lpthread -ldl -lz -lbz2 -llzma \
    -lm
endif

ifeq ($(strip $(PKG_LIBS)),)
  $(warning WARNING: pkg-config returned nothing. Run: make MANUAL=1)
endif

# ── Compiler / linker flags ────────────────────────────────────────────────
CFLAGS  := -O2 -std=c11 -Wall -Wextra \
           $(EXTRA_CFLAGS) $(PKG_CFLAGS)

# --start-group / --end-group: multiple passes to resolve circular archive refs.
# --allow-multiple-definition: suppresses duplicate-symbol errors that arise
#   when several bundled static libs (e.g. libaribcaption + libcrypto) each
#   ship their own copy of MD5 / other common routines.  The first definition
#   wins; both are identical standard implementations, so this is safe.
LDFLAGS := $(EXTRA_LDFLAGS) \
           -Wl,--allow-multiple-definition \
           -Wl,--start-group $(PKG_LIBS) -Wl,--end-group \
           -lm

# ── Rules ──────────────────────────────────────────────────────────────────
.PHONY: all clean deps-check

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

clean:
	$(RM) $(TARGET)

# Show what pkg-config resolves – useful for debugging link issues
deps-check:
	@echo "--- CFLAGS ---"
	@echo "$(PKG_CFLAGS)"
	@echo "--- LIBS ---"
	@echo "$(PKG_LIBS)"

# ── Quick-test (replace sample.mkv with your actual HDR10 file) ───────────
.PHONY: test
test: $(TARGET)
	./$(TARGET) sample.mkv output.cube 65
