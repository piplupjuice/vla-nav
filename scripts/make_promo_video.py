"""VLA-NAV cinematic overview video (v3).

Programmatically motion-designed paper-video style: every frame is rendered
in PIL (eased Ken Burns on real simulation frames, synchronized three-panel
lineups with live query counters, animated result chart, kinetic type),
then piped to ffmpeg. A quiet synthesized ambient bed is muxed underneath.

Content is strictly WHAT WAS DONE (no future work). Every number comes from
demo_meta.json / the verified logs.

Usage:  python scripts/make_promo_video.py
Output: results/videos/vlanav_overview.mp4   (1920x1080, 24 fps, ~50 s)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import wave

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(HERE, "kaggle_notebooks", "06_demo", "output", "demo")
OUT_DIR = os.path.join(HERE, "results", "videos")
OUT = os.path.join(OUT_DIR, "vlanav_overview.mp4")

W, H, FPS = 1920, 1080, 24
BG = (10, 14, 20)
PANEL_BG = (14, 20, 30)
ACCENT = (86, 214, 240)
GOOD = (94, 217, 133)
WHITE = (245, 246, 248)
GRAY = (150, 158, 168)
DIM = (96, 104, 114)

FF = imageio_ffmpeg.get_ffmpeg_exe()
META = json.load(open(os.path.join(DEMO, "demo_meta.json"), encoding="utf-8"))
_EPS = {(e["episode_index"], e["regime"]): e for e in META["episodes"]}


def font(size, bold=True):
    for name in (["arialbd.ttf"] if bold else ["arial.ttf"]) + ["arial.ttf"]:
        p = os.path.join("C:/Windows/Fonts", name)
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def ease_in_out(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


# ----------------------------------------------------------------------
# simulation footage
# ----------------------------------------------------------------------
def load_clip(ep_idx, regime):
    """[(PIL Image 400x400), ...] + meta for one episode/regime."""
    fn = [e["mp4"] for e in META["episodes"]
          if e["episode_index"] == ep_idx and e["regime"] == regime][0]
    frames = [Image.fromarray(a) for a in imageio.get_reader(os.path.join(DEMO, fn))]
    return frames, _EPS[(ep_idx, regime)]


CLIPS = {
    (0, "chunked_frozen"): load_clip(0, "chunked_frozen"),
    (0, "fixed_frequency"): load_clip(0, "fixed_frequency"),
    (0, "reliability_triggered"): load_clip(0, "reliability_triggered"),
    (1, "chunked_frozen"): load_clip(1, "chunked_frozen"),
    (1, "fixed_frequency"): load_clip(1, "fixed_frequency"),
    (1, "reliability_triggered"): load_clip(1, "reliability_triggered"),
}
# pre-scale panels once (1.6x) for crisp Ken Burns cropping
PRESCALE = {}
for k, (frames, meta) in CLIPS.items():
    PRESCALE[k] = [f.resize((640, 640), Image.LANCZOS) for f in frames]


def panel_img(key, t_play, play_dur, zoom=0.0):
    """Ken-Burns panel: eased crop of the pre-scaled frame sequence."""
    frames = PRESCALE[key]
    idx = min(int(t_play * 8), len(frames) - 1)          # native 8 fps
    base = frames[idx]
    z = 1.0 + zoom * ease_in_out(t_play / max(play_dur, 1e-6))
    win = int(640 / z)
    off = (640 - win) // 2
    return base.crop((off, off, off + win, off + win)).resize((560, 560), Image.LANCZOS)


# ----------------------------------------------------------------------
# text helpers (draw once per label on RGBA tiles, paste with live alpha)
# ----------------------------------------------------------------------
def text_tile(text, size, color, bold=True, tracking=0):
    f = font(size, bold)
    if tracking:
        text = (" " * tracking).join(list(text))
    tmp = Image.new("RGBA", (10, 10))
    w = int(ImageDraw.Draw(tmp).textlength(text, font=f)) + 8
    tile = Image.new("RGBA", (w, int(size * 1.35)), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((4, 0), text, font=f, fill=color + (255,))
    return tile


def paste_center(canvas, tile, y, alpha=1.0, x=None, dy=0):
    if alpha <= 0:
        return
    a = tile.getchannel("A").point(lambda v: int(v * alpha))
    if x is None:
        x = (W - tile.width) // 2
    canvas.paste(tile, (x, int(y + dy)), a)


def paste_left(canvas, tile, x, y, alpha=1.0):
    if alpha <= 0:
        return
    a = tile.getchannel("A").point(lambda v: int(v * alpha))
    canvas.paste(tile, (int(x), int(y)), a)


# ----------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------
def sec_cold_open(t, dur=5.4):
    """Hero shot: B2's successful final frame of episode 1, slow zoom."""
    frame = Image.new("RGB", (W, H), BG)
    # hero: fixed_frequency ep0 final frame, full-bleed cover with zoom
    frames = PRESCALE[(1, "fixed_frequency")]
    hero = frames[len(frames) - 1]
    z = 1.00 + 0.06 * ease_in_out(t / dur)
    win = int(640 / z)
    off = (640 - win) // 2
    crop = hero.crop((off, off, off + win, off + win)).resize((H, H), Image.LANCZOS)
    x0 = (W - H) // 2
    dark = Image.new("L", (H, H), 170)
    frame.paste(Image.composite(crop, Image.new("RGB", (H, H), BG), dark), (x0, 0))
    d = ImageDraw.Draw(frame)
    for gx in (x0 - 2, x0 + H):
        d.rectangle([gx, 0, gx + 2, H], fill=BG)
    a1 = ease_in_out((t - 0.7) / 0.9)
    a2 = ease_in_out((t - 2.6) / 0.9)
    t1 = text_tile("A simulation study: an 8-billion-parameter model",
                   38, WHITE)
    t2 = text_tile("navigates a photorealistic indoor world - from a camera view.",
                   38, WHITE)
    t3 = text_tile("THE ONLY QUESTION: HOW OFTEN DOES THE MODEL NEED TO BE ASKED?",
                   22, ACCENT)
    t4 = text_tile("OmniVLA  ·  AI2-THOR  ·  2×T4  ·  every number from logged runs",
                   20, DIM, bold=False)
    paste_center(frame, t1, H - 320, a1, dy=18 * (1 - a1))
    paste_center(frame, t2, H - 260, a1, dy=18 * (1 - a1))
    paste_center(frame, t3, H - 180, a2, dy=18 * (1 - a2))
    paste_center(frame, t4, H - 110, ease_in_out((t - 3.6) / 0.9))
    return frame


LINEUP_EP = {
    0: dict(head="One task: reach the light switch. Three querying policies.",
            sub="Episode 1 — 'Navigate to the lightswitch and stop there.'   "
                "(text on each view = live robot state: policy, step, speed, distance to goal)",
            panels=[("chunked_frozen", "B1 · ASKS ONCE EVERY 8 STEPS",
                     "then follows the plan blindly - and keeps colliding"),
                    ("fixed_frequency", "B2 · ASKS AT EVERY STEP (official)",
                     "keeps correcting - reaches the switch in 6 steps"),
                    ("reliability_triggered", "B3 · ASKS AGAIN ONLY WHEN STUCK",
                     "re-queries when the plan stops working")]),
    1: dict(head="Same comparison. A harder episode.",
            sub="Episode 2 — 'Go to the toaster.'   "
                "(text on each view = live robot state: policy, step, speed, distance to goal)",
            panels=[("chunked_frozen", "B1 · ASKS ONCE EVERY 8 STEPS",
                     "pushes into the counter between its rare queries"),
                    ("fixed_frequency", "B2 · ASKS AT EVERY STEP (official)",
                     "keeps correcting - still ends 0.7 m short"),
                    ("reliability_triggered", "B3 · ASKS AGAIN ONLY WHEN STUCK",
                     "re-queries 14 times - the plan keeps failing")]),
}


def sec_lineup(ep, t, dur):
    frame = Image.new("RGB", (W, H), BG)
    cfg = LINEUP_EP[ep]
    head = text_tile(cfg["head"], 44, WHITE)
    sub = text_tile(cfg["sub"], 26, GRAY, bold=False)
    ah = ease_out(t / 0.5)
    paste_center(frame, head, 52, ah, dy=24 * (1 - ah))
    paste_center(frame, sub, 122, ease_out((t - 0.25) / 0.5))

    px, py, gap = 60, 190, 24
    play_dur = dur - 1.6          # footage plays, then finals hold
    for i, (regime, label, desc) in enumerate(cfg["panels"]):
        key = (ep, regime)
        frames, meta = CLIPS[key]
        n = len(frames)
        t_play = min(max(t - 0.7 - i * 0.0, 0.0), play_dur)
        img = panel_img(key, t_play, play_dur)
        a = ease_out((t - 0.5 - i * 0.28) / 0.6)
        dy = 26 * (1 - a)
        x = px + i * (560 + gap)
        if a <= 0:
            continue
        # panel chrome
        d = ImageDraw.Draw(frame, "RGBA")
        d.rounded_rectangle([x - 6, py - 6 + dy, x + 566, py + 642 + dy],
                            10, fill=(20, 28, 40, int(220 * a)))
        frame.paste(img, (x, int(py + dy)))
        paste_left(frame, text_tile(label, 24, ACCENT), x + 2, py + 656 + dy, a)
        paste_left(frame, text_tile(desc, 20, GRAY, bold=False),
                   x + 2, py + 692 + dy, a)
        # live counters
        done = t_play >= play_dur - 1e-6
        steps = meta["steps"] if done else min(int(t_play * 8), meta["steps"])
        queries = meta["inferences"] if done else \
            int(round(meta["inferences"] * min(t_play / max(play_dur, 1e-6), 1.0)))
        if done and meta["success"]:
            tag = text_tile("SUCCESS", 22, (10, 20, 14), bold=True)
            chip = Image.new("RGBA", (tag.width + 28, tag.height + 12), GOOD + (255,))
            chip.paste(tag, (14, 6), tag)
            paste_left(frame, chip, x + 300, py + 652 + dy, a)
            ctext = f"{meta['inferences']} queries  ·  {meta['steps']} steps"
        else:
            ctext = f"queries {queries}   ·   steps {steps}"
        paste_left(frame, text_tile(ctext, 24, WHITE if not done else GRAY),
                   x + 2, py + 726 + dy, a)
    return frame


def sec_result(t, dur=11.0):
    frame = Image.new("RGB", (W, H), BG)
    a_head = ease_out(t / 0.5)
    paste_center(frame, text_tile("Same success. 60% fewer queries.", 56, WHITE),
                 90, a_head, dy=26 * (1 - a_head))
    paste_center(frame, text_tile("model queries per episode — 192 paired "
                                  "test runs", 26, GRAY, bold=False),
                 172, ease_out((t - 0.3) / 0.5))
    # animated bars
    data = [("B1 · chunked", 6.7, DIM, False),
            ("B2 · official every-step", 41.4, (72, 88, 110), False),
            ("B3 · reliability-triggered", 16.4, ACCENT, True)]
    bx, by, bw, bh, gap = 330, 300, 360, 470, 120
    for i, (label, val, color, hero) in enumerate(data):
        x = bx + i * (bw + gap)
        gr = ease_out((t - 0.7 - i * 0.35) / 1.1)
        h = bh * (val / 50.0) * gr
        d = ImageDraw.Draw(frame, "RGBA")
        d.rounded_rectangle([x, by + bh - h, x + bw, by + bh], 8,
                            fill=color + (255,))
        if gr >= 1.0:
            paste_center(frame, text_tile(f"{val:.1f}", 34, WHITE),
                         0, 1.0, x=x + bw // 2 - 30, dy=by + bh - h - 58)
        paste_center(frame, text_tile(label, 22, GRAY, bold=False),
                     0, ease_out((t - 1.2 - i * 0.35) / 0.5),
                     x=x + bw // 2 - 150, dy=by + bh + 18)
        if hero and t > 2.6:
            ap = ease_out((t - 2.6) / 0.5)
            chip = text_tile("−60% vs B2", 26, (8, 24, 30))
            bgw = Image.new("RGBA", (chip.width + 30, chip.height + 14),
                            ACCENT + (255,))
            bgw.paste(chip, (15, 7), chip)
            paste_center(frame, bgw, 0, ap, x=x + bw // 2 - bgw.width // 2,
                         dy=by + bh - h - 130)
        if hero and t > 3.4:
            ap = ease_out((t - 3.4) / 0.5)
            paste_center(frame, text_tile("same success rate", 22, GOOD,
                                          bold=False),
                         0, ap, x=x + bw // 2 - 110, dy=by + bh + 52)
    # footer line
    a = ease_out((t - 4.2) / 0.8)
    paste_center(frame, text_tile("70 s vs 175 s of pure inference per "
                                  "episode  ·  exact McNemar on paired "
                                  "episodes  ·  recomputes from raw logs",
                                  24, DIM, bold=False),
                 H - 96, a, dy=16 * (1 - a))
    return frame


def sec_summary(t, dur=8.6):
    frame = Image.new("RGB", (W, H), BG)
    paste_center(frame, text_tile("What this work did", 52, WHITE),
                 96, ease_out(t / 0.5))
    rows = [
        ("Ran a frozen OmniVLA-8B on consumer GPUs (2×T4)", "inference 3.6–4.5 s / query"),
        ("Built the benchmark: photorealistic AI2-THOR, language goals,", "4 shift conditions"),
        ("192 paired episodes — identical seeds across all regimes", "exact McNemar + bootstrap"),
        ("Reliability-triggered replanning", "−60% inference at unchanged success"),
    ]
    y = 220
    for i, (a_txt, b_txt) in enumerate(rows):
        a = ease_out((t - 0.5 - i * 0.42) / 0.6)
        dx = 40 * (1 - a)
        paste_left(frame, text_tile(a_txt, 34, WHITE), 240, y, a and dx and 1.0 or a)
        paste_left(frame, text_tile(b_txt, 24, ACCENT, bold=False), 250, y + 52, a)
        d = ImageDraw.Draw(frame, "RGBA")
        d.line([(240, y + 108), (W - 240, y + 108)],
               fill=(40, 52, 66, int(200 * a)), width=1)
        y += 150
    fn = text_tile("outdoor-trained models still struggle indoors — measured "
                   "and reported, not hidden", 22, GRAY, bold=False)
    paste_center(frame, fn, H - 92, ease_out((t - 3.6) / 0.7))
    return frame


def sec_end(t, dur=5.0):
    frame = Image.new("RGB", (W, H), BG)
    frames = PRESCALE[(1, "fixed_frequency")]
    hero = frames[len(frames) - 1].resize((H, H), Image.LANCZOS)
    x0 = (W - H) // 2
    dark = Image.new("L", (H, H), 60)
    frame.paste(Image.composite(hero, Image.new("RGB", (H, H), BG), dark), (x0, 0))
    a1 = ease_out(t / 0.7)
    paste_center(frame, text_tile("VLA-NAV", 96, WHITE), 330, a1, dy=20 * (1 - a1))
    a2 = ease_out((t - 0.5) / 0.7)
    paste_center(frame, text_tile("When Should a Robot Think Again?", 40, GRAY),
                 470, a2, dy=16 * (1 - a2))
    a3 = ease_out((t - 1.1) / 0.7)
    paste_center(frame, text_tile("REAL MODEL  ·  REAL SIMULATOR  ·  "
                                  "FULLY REPRODUCIBLE", 22, ACCENT),
                 620, a3, dy=14 * (1 - a3))
    return frame


# ----------------------------------------------------------------------
# timeline
# ----------------------------------------------------------------------
# (section_fn, args, start, end) — 0.4 s dip-to-black between sections
TIMELINE = [
    (sec_cold_open, (), 0.0, 5.4),
    (lambda t, dur: sec_lineup(0, t, dur), (9.5,), 5.8, 15.3),
    (lambda t, dur: sec_lineup(1, t, dur), (7.5,), 15.7, 23.2),
    (sec_result, (), 23.6, 34.6),
    (sec_summary, (), 35.0, 43.6),
    (sec_end, (), 44.0, 49.0),
]
DIP = 0.4


def render_frame(t_global):
    for fn, args, start, end in TIMELINE:
        if start <= t_global < end:
            return fn(t_global - start, *args)
    # dip-to-black: blend the two neighbours through darkness
    for (fn0, a0, s0, e0), (fn1, a1, s1, e1) in zip(TIMELINE, TIMELINE[1:]):
        if e0 <= t_global < s1:
            k = (t_global - e0) / max(s1 - e0, 1e-6)
            if k < 0.5:
                img = fn0(e0 - s0 - 1e-3, *a0)
            else:
                img = fn1(0.0, *a1)
            return Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)),
                               abs(1 - 2 * k))
    return Image.new("RGB", (W, H), BG)


# ----------------------------------------------------------------------
# ambient audio bed (synthesized; quiet, warm, slow chord drift + swells)
# ----------------------------------------------------------------------
def make_audio(path, total):
    sr = 44100
    n = int(total * sr)
    t = np.arange(n) / sr
    chords = [  # (freqs, start, end)
        ([110.0, 164.8, 261.6], 0.0, total * 0.25),      # Am
        ([87.3, 130.8, 220.0], total * 0.22, total * 0.47),  # F
        ([130.8, 196.0, 261.6], total * 0.44, total * 0.72),  # C
        ([98.0, 146.8, 246.9], total * 0.69, total),     # G
    ]
    mix = np.zeros(n)
    for freqs, s0, s1 in chords:
        i0, i1 = int(s0 * sr), min(int(s1 * sr), n)
        seg = np.zeros(i1 - i0)
        tt = np.arange(i1 - i0) / sr
        for f in freqs:
            for det in (-0.4, 0.0, 0.4):
                ph = 2 * np.pi * (f + det) * tt
                seg += 0.33 * np.sin(ph) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * tt))
        env = np.minimum(1, tt / 2.5) * np.minimum(1, (s1 - s0 - (tt)) / 2.5 + 1e-9)
        env = np.clip(env, 0, 1)
        mix[i0:i1] += seg * env / (len(freqs) * 3)
    # soft swells at section starts
    for _, _, s, _e in TIMELINE[1:]:
        i0 = int(max(s - 0.9, 0) * sr)
        i1 = int(min((s + 1.2), total) * sr)
        seg_t = np.arange(i1 - i0) / sr
        swell = np.sin(2 * np.pi * 440 * seg_t) * np.exp(-4 * seg_t) * 0.05
        mix[i0:i1] += swell
    mix = mix / max(np.max(np.abs(mix)), 1e-9) * 0.16
    st = (np.stack([mix, mix], 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(st.tobytes())


def main():
    total = TIMELINE[-1][3]
    tmp = tempfile.mkdtemp(prefix="vlanav_promo3_")
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        silent = os.path.join(tmp, "video.mp4")
        frames = imageio_ffmpeg.write_frames(
            silent, (W, H), fps=FPS, codec="libx264", quality=None,
            macro_block_size=1,
            output_params=["-crf", "19", "-preset", "medium",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        frames.send(None)
        n_frames = int(total * FPS)
        for i in range(n_frames):
            frames.send(np.asarray(render_frame(i / FPS)))
        frames.close()

        wav = os.path.join(tmp, "bed.wav")
        make_audio(wav, total)
        run = [FF, "-y", "-i", silent, "-i", wav,
               "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
               "-shortest", "-movflags", "+faststart", OUT]
        r = subprocess.run(run, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1200:])
        print("wrote", os.path.relpath(OUT, HERE),
              f"({os.path.getsize(OUT) / 1e6:.1f} MB, {total:.0f} s, "
              f"{n_frames} rendered frames)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
