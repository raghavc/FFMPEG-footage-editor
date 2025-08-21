# main.py
# TikTok vertical canvas with blurred background, centered 16:9 foreground (no crop),
# live word-by-word captions, bottom "Part N" pill, and a centered TITLE PILL
# placed in the top blur zone. Title text is derived from the input filename.
#
# Requirements (Python 3.12 recommended):
#   moviepy==1.0.3 pillow==9.5.0 numpy==1.26.4 imageio==2.34.1 imageio-ffmpeg==0.4.8
#   proglog==0.1.10 requests==2.31.0 tqdm==4.66.4 setuptools==80.9.0 wheel==0.44.0
#   faster-whisper==0.10.0 opencv-python-headless==4.10.0.84
#
# macOS: brew install ffmpeg
#
# Usage:
#   mkdir -p input_videos output_videos
#   put your videos in input_videos/
#   python main.py

import os
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from faster_whisper import WhisperModel
from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    ImageClip,
)

# ===================== CONFIG =====================
INPUT_FOLDER   = "input_videos"
OUTPUT_FOLDER  = "output_videos"

# Split each video into this many equal parts
NUM_PARTS      = 3

# Speech-to-text model (for word-timed captions)
MODEL_SIZE     = "base"      # "tiny" | "base" | "small" | "medium" (bigger = better, slower)
COMPUTE_TYPE   = "int8"      # good on CPU; try "int8_float16" on Apple Silicon if you want

# Final vertical canvas (TikTok)
FINAL_W, FINAL_H = 1080, 1920

# Foreground layout: keep full 16:9 (fit within vertical canvas), no crop
# Reserve some bottom space for captions + "Part" pill
BOTTOM_RESERVE   = 380

# Blur background styling
BLUR_SIGMA       = 22     # 18–30 typical
BG_DARKEN        = 0.82   # 0..1 multipler (lower = darker)

# Caption styling
CAPTION_FONTSIZE = 72
CAPTION_COLOR    = (255, 255, 255)
CAPTION_STROKE   = (0, 0, 0)
CAPTION_STROKE_W = 6

# Highlight (current word) styling
HILITE_COLOR     = (220, 20, 60)  # red
HILITE_STROKE    = (0, 0, 0)
HILITE_STROKE_W  = 6

# Where captions sit (relative to bottom of 1920 canvas)
CAPTION_Y        = FINAL_H - 260

# "Part" pill styling (bottom)
PILL_TEXT_COLOR  = (255, 255, 255)
PILL_BG_COLOR    = (26, 101, 194)
PILL_PADDING_X   = 36
PILL_PADDING_Y   = 14
PILL_RADIUS      = 24
PILL_FONTSIZE    = 68
PILL_BOTTOM_PAD  = 80  # distance from bottom of screen

# Title pill (top center in blur zone)
TITLE_PILL_BG_COLOR   = (26, 101, 194)   # blue
TITLE_PILL_TEXT_COLOR = (255, 255, 255)
TITLE_PILL_FONTSIZE   = 58
TITLE_PILL_PADDING_X  = 34
TITLE_PILL_PADDING_Y  = 14
TITLE_PILL_RADIUS     = 28
TITLE_MAX_WIDTH_FRAC  = 0.86   # wrap if longer than 86% of canvas width
TITLE_MIN_MARGIN_TOP  = 40     # don’t glue to screen edge

# Fonts (adjust to something on your system if these aren’t present)
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial.ttf",
]
# ==================================================


# ----------------- Font helpers -------------------
def ensure_font() -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=60)
    return ImageFont.load_default()

def font_of(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

_ = ensure_font()  # prime font cache


# ------------- Drawing / compositing --------------
def draw_text_image(
    text: str,
    fontsize: int,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int],
    stroke_width: int,
    max_width: int = None,
    align: str = "center",
    bg: Tuple[int, int, int, int] = None
) -> Image.Image:
    """Render multiline text (Pillow) with stroke; returns RGBA image sized to content."""
    font = font_of(fontsize)

    # Simple greedy wrap
    lines = [text]
    if max_width is not None:
        lines = []
        words = text.split()
        cur = ""
        tmp_img = Image.new("RGBA", (max_width, 10), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp_img)
        for w in words:
            test = (cur + " " + w).strip()
            bbox = d.textbbox((0, 0), test, font=font, stroke_width=stroke_width)
            width = bbox[2] - bbox[0]
            if width > max_width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)

    # Calculate total img size
    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    widths, heights = [], []
    for ln in lines:
        bbox = d.textbbox((0, 0), ln, font=font, stroke_width=stroke_width)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])

    W = max(widths) if widths else 2
    H = sum(heights) + (len(lines) - 1) * 10

    out = Image.new("RGBA", (W, H), bg if bg else (0, 0, 0, 0))
    d2 = ImageDraw.Draw(out)

    y = 0
    for ln in lines:
        bbox = d2.textbbox((0, 0), ln, font=font, stroke_width=stroke_width)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if align == "center":
            x = (W - w) // 2
        elif align == "left":
            x = 0
        else:
            x = W - w
        d2.text(
            (x, y),
            ln,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        y += h + 10
    return out

def rounded_rectangle(w: int, h: int, radius: int, color: Tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(0, 0), (w, h)], radius=radius, fill=color + (255,))
    return img

def make_part_pill_clip(text: str, duration: float, video_w: int, video_h: int) -> ImageClip:
    font = font_of(PILL_FONTSIZE)
    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pill_w = tw + 2 * PILL_PADDING_X
    pill_h = th + 2 * PILL_PADDING_Y
    pill = rounded_rectangle(pill_w, pill_h, PILL_RADIUS, PILL_BG_COLOR)
    d2 = ImageDraw.Draw(pill)
    d2.text(
        ((pill_w - tw) // 2, (pill_h - th) // 2),
        text,
        font=font,
        fill=PILL_TEXT_COLOR,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    return (
        ImageClip(np.array(pill))
        .set_duration(duration)
        .set_position(("center", video_h - pill_h - PILL_BOTTOM_PAD))
    )

def derive_title_from_filename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    junk_prefixes = ["ssvid.net--", "yt5s.io--", "y2mate--", "savefrom--"]
    bl = base.lower()
    for j in junk_prefixes:
        if bl.startswith(j):
            base = base[len(j):]
            break
    title = base.replace("_", " ").replace("-", " ").strip()
    title = " ".join(title.split())
    return title.title()

def make_title_pill_clip(text: str, duration: float, canvas_w: int) -> ImageClip:
    max_w = int(canvas_w * TITLE_MAX_WIDTH_FRAC)
    txt_img = draw_text_image(
        text=text,
        fontsize=TITLE_PILL_FONTSIZE,
        fill=TITLE_PILL_TEXT_COLOR,
        stroke_fill=(0, 0, 0),
        stroke_width=2,
        max_width=max_w,
        align="center",
    )
    tw, th = txt_img.size
    pill_w = tw + 2 * TITLE_PILL_PADDING_X
    pill_h = th + 2 * TITLE_PILL_PADDING_Y
    pill_bg = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(pill_bg)
    d.rounded_rectangle(
        [(0, 0), (pill_w, pill_h)],
        radius=TITLE_PILL_RADIUS,
        fill=TITLE_PILL_BG_COLOR + (255,),
    )
    pill_bg.alpha_composite(txt_img, ((pill_w - tw) // 2, (pill_h - th) // 2))
    return ImageClip(np.array(pill_bg)).set_duration(duration)

# --------------- Captions (karaoke-ish) ---------------
def transcribe_words(path: str) -> List[Tuple[float, float, str]]:
    """Return list of (start, end, word) using faster-whisper."""
    model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)
    segments, _ = model.transcribe(path, word_timestamps=True)
    words: List[Tuple[float, float, str]] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                ww = (w.word or "").strip()
                if ww:
                    # ensure non-zero duration
                    start = max(0.0, float(w.start))
                    end = max(start + 0.05, float(w.end))
                    words.append((start, end, ww))
        else:
            txt = (seg.text or "").strip()
            if txt:
                words.append((float(seg.start), float(seg.end), txt))
    return words

def make_word_clip(word: str, duration: float, max_width: int) -> ImageClip:
    img = draw_text_image(
        word,
        fontsize=CAPTION_FONTSIZE,
        fill=HILITE_COLOR,
        stroke_fill=HILITE_STROKE,
        stroke_width=HILITE_STROKE_W,
        max_width=max_width,
    )
    return ImageClip(np.array(img)).set_duration(duration)

def make_caption_base(text: str, duration: float, max_width: int) -> ImageClip:
    img = draw_text_image(
        text,
        fontsize=CAPTION_FONTSIZE,
        fill=CAPTION_COLOR,
        stroke_fill=CAPTION_STROKE,
        stroke_width=CAPTION_STROKE_W,
        max_width=max_width,
    )
    return ImageClip(np.array(img)).set_duration(duration)

def build_live_captions_clip(
    words: List[Tuple[float, float, str]],
    part_start: float,
    part_end: float,
    video_w: int,
    video_h: int,
) -> CompositeVideoClip:
    """
    Karaoke-like captions: show rolling base text in white and overlay
    the current word in red, timed to speech.
    """
    selected = [(s, e, w) for (s, e, w) in words if e > part_start and s < part_end]
    clips = []
    LOOKBACK = 2.2
    MAX_W = int(video_w * 0.86)

    for (s, e, w) in selected:
        ws = max(0.0, s - part_start)
        we = min(part_end - part_start, e - part_start)
        if we <= ws:
            continue

        rolling = [ww for (ss, ee, ww) in selected if ss <= s and (s - ss) <= LOOKBACK]
        base_text = " ".join(rolling).strip() or w

        base = make_caption_base(base_text, duration=we - ws, max_width=MAX_W).set_start(ws)
        red = make_word_clip(w, duration=we - ws, max_width=MAX_W).set_start(ws)

        base = base.set_position(("center", CAPTION_Y))
        red = red.set_position(("center", CAPTION_Y))

        clips += [base, red]

    return CompositeVideoClip(clips) if clips else CompositeVideoClip([])


# --------------- Blurred background -------------------
def build_blurred_bg(
    base_clip: VideoFileClip,
    out_w: int = FINAL_W,
    out_h: int = FINAL_H,
    sigma: int = BLUR_SIGMA,
    darken: float = BG_DARKEN,
) -> VideoFileClip:
    """
    Duplicate the original video, scale to fill 9:16 canvas, apply heavy Gaussian blur and darken.
    """
    resized = base_clip.resize(height=out_h)
    x1 = int(max(0, (resized.w - out_w) // 2))
    bg = resized.crop(x1=x1, y1=0, x2=x1 + out_w, y2=out_h)

    def _blur(frame):
        f = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return np.clip(f * darken, 0, 255).astype(np.uint8)

    bg = bg.fl_image(_blur)
    return bg


# ----------------- Processing pipeline -----------------
def process_video(path: str):
    print(f"Processing: {path}")
    base_name = os.path.splitext(os.path.basename(path))[0]
    video_title = derive_title_from_filename(path)

    # Open source
    src = VideoFileClip(path)
    src_w, src_h = src.w, src.h

    # Background: blurred + darkened full canvas
    bg = build_blurred_bg(src, out_w=FINAL_W, out_h=FINAL_H, sigma=BLUR_SIGMA, darken=BG_DARKEN)

    # Foreground: FIT (no crop), centered between top space and bottom reserve
    avail_h = FINAL_H - BOTTOM_RESERVE
    scale = min(FINAL_W / src_w, avail_h / src_h)
    fg = src.resize(scale)
    fg_w, fg_h = fg.w, fg.h
    fg_y = (avail_h - fg_h) // 2
    fg = fg.set_position(("center", fg_y))

    # Title pill Y position: center of top blur band (space above the foreground video)
    top_blur_height = fg_y  # pixels from top of screen down to top of the foreground
    title_y = max(TITLE_MIN_MARGIN_TOP, int(top_blur_height * 0.5))

    # Transcribe once for word-timed captions
    word_times = transcribe_words(path)

    total = src.duration
    part_len = total / NUM_PARTS
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for i in range(NUM_PARTS):
        p_start = i * part_len
        p_end = (i + 1) * part_len

        sub_bg = bg.subclip(p_start, p_end)
        sub_fg = fg.subclip(p_start, p_end)

        overlays = []
        # Title pill from filename, placed in the top blur zone
        title_pill = make_title_pill_clip(video_title, sub_fg.duration, FINAL_W).set_position(("center", title_y))
        overlays.append(title_pill)

        # Bottom "Part N" pill
        overlays.append(make_part_pill_clip(f"Part {i+1}", sub_fg.duration, FINAL_W, FINAL_H))

        # Live captions
        overlays.append(build_live_captions_clip(word_times, p_start, p_end, FINAL_W, FINAL_H))

        final = CompositeVideoClip([sub_bg, sub_fg] + overlays, size=(FINAL_W, FINAL_H))

        out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_part{i+1}.mp4")
        final.write_videofile(out_path, codec="libx264", fps=30, audio_codec="aac")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for f in os.listdir(INPUT_FOLDER):
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
            process_video(os.path.join(INPUT_FOLDER, f))


if __name__ == "__main__":
    main()
