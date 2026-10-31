"""Generate report figures from the verified analysis artifacts.

Every number plotted here comes from results/analysis/full_analysis.json
(written by scripts/analyze_full.py from results/raw/**/episodes.jsonl) —
nothing is typed by hand. Demo frames come from the fetched 06_demo output.

Outputs (results/figures/):
  fig_success_rate.png   SR per condition x regime + pooled, Wilson 95% CI bars
  fig_inference_cost.png mean model queries per episode per regime (pooled)
  fig_demo_frames.png    final frames of demo episode 0, B1/B2/B3 side by side
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(HERE, "results", "analysis", "full_analysis.json")
DEMO_DIR = os.path.join(HERE, "kaggle_notebooks", "06_demo", "output", "demo")
OUT = os.path.join(HERE, "results", "figures")

# Morandi-style low-saturation colors, one per regime
COLORS = {"chunked_frozen": "#8A94A6", "fixed_frequency": "#3D4A5C",
          "reliability_triggered": "#1B6B7A"}
LABELS = {"chunked_frozen": "B1 chunked-frozen",
          "fixed_frequency": "B2 fixed-frequency (official)",
          "reliability_triggered": "B3 reliability-triggered (ours)"}
ORDER = ["chunked_frozen", "fixed_frequency", "reliability_triggered"]
CONDS = ["A_in_dist", "B_env_shift", "C_instr_shift", "D_recovery"]
COND_LABELS = {"A_in_dist": "A\nin-dist", "B_env_shift": "B\nenv shift",
               "C_instr_shift": "C\ninstr shift", "D_recovery": "D\nrecovery"}

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(p, HERE))


def fig_success_rate(a):
    conds = CONDS + ["POOLED"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    width = 0.26
    xs = np.arange(len(conds))
    for i, m in enumerate(ORDER):
        sr, lo, hi = [], [], []
        for c in CONDS:
            cell = a["per_cell"].get(f"{c}|{m}")
            sr.append(cell["SR"] if cell else np.nan)
            lo.append(cell["wilson95"][0] if cell else np.nan)
            hi.append(cell["wilson95"][1] if cell else np.nan)
        pooled = a["pooled"][m]
        sr.append(pooled["SR"])
        lo.append(pooled["wilson95"][0])
        hi.append(pooled["wilson95"][1])
        yerr = [np.array(sr) - np.array(lo), np.array(hi) - np.array(sr)]
        ax.bar(xs + (i - 1) * width, sr, width * 0.92, yerr=yerr,
               capsize=3, color=COLORS[m], label=LABELS[m],
               error_kw={"elinewidth": 1.0})
    ax.set_xticks(xs)
    ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 0.34)
    ax.set_title("Success rate with Wilson 95% confidence intervals (test split)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "fig_success_rate.png")


def fig_inference_cost(a):
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ms = ORDER
    vals = [a["pooled"][m]["inferences_per_ep"] for m in ms]
    bars = ax.bar([LABELS[m] for m in ms], vals,
                  color=[COLORS[m] for m in ms], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}",
                ha="center", fontsize=11)
    b2 = vals[1]
    for idx, m in enumerate(ms):
        pct = 100 * (1 - vals[idx] / b2)
        if m != "fixed_frequency":
            ax.text(idx, vals[idx] / 2, f"\u2212{pct:.0f}% vs B2",
                    ha="center", color="white", fontsize=10)
    ax.set_ylabel("model queries per episode (mean)")
    ax.set_ylim(0, 50)
    ax.set_title("Inference cost of each replanning regime (192 test episodes)")
    ax.tick_params(axis="x", labelsize=9)
    save(fig, "fig_inference_cost.png")


def fig_demo_frames():
    frames = [
        ("B1 chunked-frozen", "ep0_30001_chunked_frozen_last.jpg"),
        ("B2 fixed-frequency", "ep0_30001_fixed_frequency_last.jpg"),
        ("B3 reliability-triggered", "ep0_30001_reliability_triggered_last.jpg"),
    ]
    imgs = []
    for _, fn in frames:
        p = os.path.join(DEMO_DIR, fn)
        if not os.path.isfile(p):
            print("missing demo frame:", fn)
            return
        imgs.append(Image.open(p))
    w, h = imgs[0].size
    scale = 340 / h
    imgs = [im.resize((int(im.width * scale), 340)) for im in imgs]
    total_w = sum(im.width for im in imgs) + 20 * (len(imgs) + 1)
    canvas = Image.new("RGB", (total_w, 340 + 58), "white")
    x = 20
    for im in imgs:
        canvas.paste(im, (x, 44))
        x += im.width + 20
    fig, ax = plt.subplots(figsize=(total_w / 100, (340 + 58) / 100))
    ax.imshow(canvas)
    ax.axis("off")
    xx = 20
    for (label, _), im in zip(frames, imgs):
        ax.text(xx + im.width / 2, 22, label, ha="center", fontsize=10)
        xx += im.width + 20
    ax.set_title("Demo episode 0 (seed 30001, FloorPlan26) - final recorded frame of each regime",
                 fontsize=10)
    save(fig, "fig_demo_frames.png")


def fig_modality():
    """Bottleneck sweep result (08b): language-only vs language+goal-pose."""
    src = os.path.join(HERE, "kaggle_notebooks", "08b_bottleneck", "output",
                       "results", "bottleneck_summary.json")
    a = json.load(open(src, encoding="utf-8"))
    bars = [("language only", a["('language', 'fixed_frequency')"]),
            ("language + goal pose", a["('language_pose', 'fixed_frequency')"]),
            ("language + trigger", a["('language', 'reliability_triggered')"])]
    labels = [b[0] for b in bars]
    sr = [b[1]["SR"] for b in bars]
    n = [b[1]["n"] for b in bars]
    queries = [b[1]["queries"] for b in bars]
    colors = ["#8A94A6", "#3D4A5C", "#1B6B7A"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars_ = ax.bar(labels, sr, color=colors, width=0.55)
    for b, r, nn, q in zip(bars_, sr, n, queries):
        k = round(r * nn)
        ax.text(b.get_x() + b.get_width() / 2, r + 0.008, f"{k}/{nn}",
                ha="center", fontsize=12)
        ax.text(b.get_x() + b.get_width() / 2, r + 0.028, f"{q:.1f} queries/ep",
                ha="center", fontsize=9, color="#506070")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 0.25)
    ax.set_title("Giving the model the exact goal position did not help "
                 "(24 easy indoor episodes, paired seeds)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    save(fig, "fig_modality.png")


def main():
    a = json.load(open(ANALYSIS, encoding="utf-8"))
    if not a.get("coverage", {}).get("complete"):
        raise SystemExit("analysis is incomplete - refusing to draw final figures")
    fig_success_rate(a)
    fig_inference_cost(a)
    fig_demo_frames()
    fig_modality()


if __name__ == "__main__":
    main()
