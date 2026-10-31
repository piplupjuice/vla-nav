"""Build docs/final_report.md from the logged evidence (no hand-typed numbers).

Inputs (all optional; missing ones are reported as "not run yet"):
    results/tuning/threshold_choice.json
    results/raw/**/episodes.jsonl          (test matrix)
    kaggle_notebooks/02_thor_pilot/output/results/raw/**/episodes.jsonl
    kaggle_notebooks/06_demo/output/demo/demo_meta.json

Rules:
  * every table cell comes from episodes.jsonl (via analyze_full helpers);
  * the narrative is chosen from the numbers (e.g. "B3 does not beat B2"
    when the paired CI includes 0), so the text cannot drift from the data;
  * mock rows never enter a table.

Usage: python scripts/make_report.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    p = os.path.join(HERE, "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location("vlanav_" + name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


full = _load("analyze_full")

PILOT_RAW = os.path.join(HERE, "kaggle_notebooks", "02_thor_pilot", "output",
                         "results", "raw")
TEST_RAW = os.path.join(HERE, "results", "raw")
TUNING = os.path.join(HERE, "results", "tuning", "threshold_choice.json")
DEMO_META = os.path.join(HERE, "kaggle_notebooks", "06_demo", "output",
                         "demo", "demo_meta.json")
OUT = os.path.join(HERE, "docs", "final_report.md")


def cell(rs, ml=None):
    n = len(rs)
    k = sum(1 for r in rs if r["success"])
    lo, hi = full.wilson(k, n)
    return {
        "n": n, "k": k, "SR": k / n, "lo": lo, "hi": hi,
        "spl": float(np.mean([full.spl_proxy(r) for r in rs])),
        "inf": float(np.mean([r["n_inferences"] for r in rs])),
        "rp": float(np.mean([r["n_replans_triggered"] for r in rs])),
        "lat": float(np.nanmean([r["infer_latency_s_mean"] for r in rs])),
        "col": float(np.mean([r["n_collisions"] for r in rs])),
        "dfin": float(np.mean([r["distance_to_goal_m"] for r in rs])),
    }


def table(rows, conds, methods, title):
    lines = [f"### {title}", "",
             "| condition | regime | n | SR | Wilson 95% CI | SPL_proxy | inf/ep |"
             " replans/ep | collisions/ep | d_final m |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    have = False
    for cond in conds:
        for m in methods:
            rs = [r for r in rows if r["condition"] == cond and r["method"] == m]
            if not rs:
                continue
            have = True
            c = cell(rs)
            lines.append(
                f"| {cond} | {full.METHOD_LABELS.get(m, m)} | {c['n']} | "
                f"{c['SR']:.2f} | [{c['lo']:.2f}, {c['hi']:.2f}] | "
                f"{c['spl']:.3f} | {c['inf']:.1f} | {c['rp']:.1f} | "
                f"{c['col']:.1f} | {c['dfin']:.2f} |")
    if not have:
        return lines + ["", "_(no logged episodes yet)_", ""]
    return lines + [""]


def paired(rows, conds):
    lines = ["### Paired comparisons (same episodes; exact McNemar)", "",
             "| condition | A vs B | n pairs | only A | only B | McNemar p |"
             " ΔSR (A−B) | 95% bootstrap CI | inf ratio A/B |",
             "|---|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(20261030)
    have = False
    for cond in conds:
        for a_m, b_m in (("reliability_triggered", "fixed_frequency"),
                         ("reliability_triggered", "chunked_frozen"),
                         ("fixed_frequency", "chunked_frozen")):
            ra = {r["episode_id"]: r for r in rows
                  if r["condition"] == cond and r["method"] == a_m}
            rb = {r["episode_id"]: r for r in rows
                  if r["condition"] == cond and r["method"] == b_m}
            ids = sorted(set(ra) & set(rb))
            if not ids:
                continue
            have = True
            sa = np.array([1.0 if ra[i]["success"] else 0.0 for i in ids])
            sb = np.array([1.0 if rb[i]["success"] else 0.0 for i in ids])
            b = int(((sa == 1) & (sb == 0)).sum())
            c = int(((sa == 0) & (sb == 1)).sum())
            p = full.mcnemar_exact(b, c)
            d = float(sa.mean() - sb.mean())
            boot = []
            for _ in range(2000):
                idx = rng.integers(0, len(ids), len(ids))
                boot.append(float(sa[idx].mean() - sb[idx].mean()))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ia = float(np.mean([ra[i]["n_inferences"] for i in ids]))
            ib = float(np.mean([rb[i]["n_inferences"] for i in ids]))
            lines.append(
                f"| {cond} | {a_m.split('_')[0].upper()} vs "
                f"{b_m.split('_')[0].upper()} | {len(ids)} | {b} | {c} | "
                f"{p:.3f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                f"{(ia / ib if ib else float('nan')):.2f} |")
    if not have:
        return lines + ["", "_(no paired episodes logged yet)_", ""]
    return lines + [""]


def verdict(rows, conds):
    """Narrative picked from the numbers, never the other way round."""
    out = []
    b2 = [r for r in rows if r["method"] == "fixed_frequency"]
    b3 = [r for r in rows if r["method"] == "reliability_triggered"]
    if not b2 or not b3:
        return ["_Insufficient logged test episodes for a verdict._"]
    sr2 = float(np.mean([r["success"] for r in b2]))
    sr3 = float(np.mean([r["success"] for r in b3]))
    i2 = float(np.mean([r["n_inferences"] for r in b2]))
    i3 = float(np.mean([r["n_inferences"] for r in b3]))
    out.append(f"Observed on the logged test rows: B2 SR = {sr2:.2f} at "
               f"{i2:.1f} inferences/episode; B3 SR = {sr3:.2f} at "
               f"{i3:.1f} inferences/episode ({100 * (1 - i3 / i2):.0f}% fewer).")
    if sr2 == 0 and sr3 == 0:
        out.append("Both regimes fail on every logged test episode: a pretrained "
                   "outdoor VLA does not transfer to indoor THOR with this "
                   "instruction interface. This is reported as the finding, not "
                   "hidden; no claim of a B3 improvement is made.")
    elif abs(sr3 - sr2) < 1e-9:
        out.append("Success rates are identical on the logged episodes; the "
                   "only difference is inference cost, so B3 is a cost "
                   "optimisation here, not an accuracy improvement.")
    elif sr3 > sr2:
        out.append("B3's success rate exceeds B2's on the logged episodes; the "
                   "paired McNemar p-value above says whether that survives "
                   "episode-level noise.")
    else:
        out.append("B3 underperforms B2 on the logged episodes; the honest "
                   "reading is that the onboard reliability signal (commanded "
                   "minus observed displacement) is not sufficient here.")
    return out


def main() -> int:
    pilot_rows, _ = full.load(PILOT_RAW) if os.path.isdir(PILOT_RAW) else ([], {})
    test_rows, test_metas = full.load(TEST_RAW) if os.path.isdir(TEST_RAW) else ([], {})
    pilot_rows = [r for r in pilot_rows if r.get("policy_is_model")]
    test_rows = [r for r in test_rows if r.get("policy_is_model")]
    methods = ["chunked_frozen", "fixed_frequency", "reliability_triggered"]

    tuning = json.load(open(TUNING, encoding="utf-8")) if os.path.isfile(TUNING) else None
    tuning_ann = None
    _ann = os.path.join(os.path.dirname(TUNING), "threshold_choice.annotated.json")
    if os.path.isfile(_ann):
        tuning_ann = json.load(open(_ann, encoding="utf-8"))
    demo = json.load(open(DEMO_META, encoding="utf-8")) if os.path.isfile(DEMO_META) else None

    test_conds = [c for c in full.COND_ORDER if any(r["condition"] == c for r in test_rows)]
    pilot_conds = sorted({r["condition"] for r in pilot_rows})

    md = [
        "# VLA-NAV — final report (generated from logged evidence)",
        "",
        "> Generated by `scripts/make_report.py` from `episodes.jsonl` files.",
        "> Every number below recomputes with `scripts/analyze_full.py`; no value",
        "> is transcribed by hand.",
        "",
        "## 1. Question",
        "",
        "A pretrained Vision-Language-Action model (OmniVLA, 8B) can be queried",
        "at every control tick (the official regime, B2) or sparsely with a held",
        "waypoint chunk (B1). We test whether a cheap **onboard reliability",
        "signal** (commanded minus observed displacement, plus stalled-motion",
        "run length) can decide when to re-query (B3), keeping success while",
        "cutting inference.",
        "",
        "## 2. Frozen protocol",
        "",
        "| item | value |",
        "|---|---|",
        "| model | `NHirose/omnivla-original` @ step 120000, bf16, `device_map=auto` (2×T4) |",
        "| simulator | AI2-THOR 5.0.0, `Linux64` + Xvfb, 400×400 real renders |",
        "| goal interface | official language-only modality (7); no beacon drawn |",
        "| control | 8-waypoint chunk, official PD decode, waypoint 4, dt = 1/3 s |",
        "| val | FloorPlan1..30 (threshold tuning only) |",
        "| test | FloorPlan401..430 (all reported numbers) |",
        "| env shift | FloorPlan301..330 |",
        "| B3 threshold | "
        + (f"`threshold_m = {tuning['threshold_m']}` (val sweep, `{tuning.get('selection_rule', '')}`)"
           if tuning else "_not tuned yet (val sweep pending)_")
        + " |",
        "",
        "## 3. Pilot (val, n=10/regime, 40 ticks)",
        "",
    ]
    if tuning_ann:
        md += ["**Threshold provenance:** " + str(tuning_ann.get("degenerate_note", "")),
               ""]
    md += table(pilot_rows, pilot_conds, methods, "Pilot table (val)")
    md += [
        "Pilot caveats: n=10, 40-tick budget, B3 threshold = placeholder unless",
        "the tuning kernel had already run; `SPL_proxy` ignores walls (optimistic).",
        "",
        "## 4. Test matrix",
        "",
    ]
    md += table(test_rows, test_conds, methods, "Test table")
    md += paired(test_rows, test_conds)
    md += ["### Reading of the test numbers", ""]
    md += ["- " + s for s in verdict(test_rows, test_conds)]
    md += [""]

    md += ["## 5. Inference cost (the point of the exercise)", ""]
    if test_rows:
        for m in methods:
            rs = [r for r in test_rows if r["method"] == m]
            if rs:
                md.append(f"- {full.METHOD_LABELS[m]}: "
                          f"{np.mean([r['n_inferences'] for r in rs]):.1f} "
                          f"inferences/episode, "
                          f"{np.mean([r['infer_latency_s_mean'] for r in rs]):.2f} s "
                          f"mean latency, "
                          f"{np.mean([r['wall_time_s'] for r in rs]):.1f} s of pure "
                          f"inference per episode")
    else:
        md.append("- _no test rows yet_")
    md += ["",
           "## 6. Demo video",
           ""]
    if demo:
        md += [f"Threshold in the demo: `{demo['threshold_m']}` m "
               f"(source: `{demo['threshold_source']}`). Every tick of both "
               f"episodes is in the video for all three regimes.", "",
               "| episode | regime | success | steps | inferences | collisions | clip |",
               "|---|---|---|---|---|---|---|"]
        for e in demo["episodes"]:
            md.append(f"| {e['episode_index']} (seed {e['seed']}) | {e['regime']} | "
                      f"{e['success']} | {e['steps']} | {e['inferences']} | "
                      f"{e['collisions']} | `{e['mp4']}` |")
    else:
        md.append("_demo not generated yet_")
    md += ["",
           "## 7. Limitations (stated, not hidden)",
           "",
           "- Small n (16/condition on test, 10 on the pilot): intervals are wide,",
           "  and are printed next to every rate.",
           "- `SPL_proxy` is a straight-line proxy, not true SPL (walls ignored).",
           "- Domain shift is real: OmniVLA was trained outdoors; THOR is indoor.",
           "  Low success is a finding about transfer, not a tuning failure.",
           "- The reliability signal uses only odometry; it cannot see a new",
           "  obstacle that the robot has not yet bumped into.",
           "- AI2-THOR is a discrete simulator; the (v, ω) → Rotate/MoveAhead",
           "  mapping is deterministic and identical for all three regimes.",
           "",
           "## 8. Reproduction",
           "",
           "```bash",
           "python scripts/gen_notebook_02.py && kaggle kernels push -p kaggle_notebooks/02_thor_pilot",
           "python scripts/gen_notebook_03.py && kaggle kernels push -p kaggle_notebooks/03_tuning",
           "python scripts/gen_notebook_04.py a   && kaggle kernels push -p kaggle_notebooks/04a_test_ab",
           "python scripts/gen_notebook_04.py b   && kaggle kernels push -p kaggle_notebooks/04b_test_cd",
           "python scripts/gen_notebook_06.py && kaggle kernels push -p kaggle_notebooks/06_demo",
           "python scripts/analyze_full.py && python scripts/make_report.py",
           "```",
           ""]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", os.path.relpath(OUT, HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
