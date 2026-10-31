"""Full-matrix analysis: SR + Wilson CI, SPL proxy, paired McNemar, bootstrap.

Reads ONLY `results/raw/<exp_id>/{episodes.jsonl,meta.json}` (the evidence),
recomputes every reported number, and writes:
    results/analysis/full_analysis.md
    results/analysis/full_analysis.json

Rules encoded here (honesty):
  * SPL is the straight-line shortest-path *proxy* (walls ignored) -> an
    optimistic bound; stated in every output.
  * Paired tests resample EPISODES, not rows, so B1/B2/B3 share the same
    (condition, episode) pairing. Discordant pairs -> exact McNemar
    (two-sided binomial).
  * Any row with `policy_is_model` false is reported and excluded from
    headline numbers (a mock run must never look like a real one).
  * Threshold provably came from val: meta.extra.threshold_source must NOT
    mention the test split. If it does, the script refuses.

Usage:
  python scripts/analyze_full.py [--raw results/raw] [--out results/analysis]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD_LABELS = {
    "chunked_frozen": "B1 chunked-frozen",
    "fixed_frequency": "B2 fixed-frequency (official)",
    "reliability_triggered": "B3 reliability-triggered",
}
COND_ORDER = ["A_in_dist", "B_env_shift", "C_instr_shift", "D_recovery"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def spl_proxy(r) -> float:
    d = float(r["distance_start_to_goal_m"])
    if r["success"] and d > 1e-6:
        return d / max(float(r["path_length_m"]), d)
    return 0.0


def canon_method(name: str) -> str:
    """Normalize regime names to the three canonical ids.

    Regression found by the local dry-run (2026-10-30): `FixedFrequencyReplan`
    auto-derives the name `fixed_frequency_replan` from its class name, while
    the analyzers/report key on `fixed_frequency` -> B2 rows would have been
    silently dropped from every summary.
    """
    aliases = {
        "fixed_frequency_replan": "fixed_frequency",
        "chunkedfrozen": "chunked_frozen",
        "reliabilitytriggered": "reliability_triggered",
    }
    key = str(name)
    return aliases.get(key, key)


def load(raw_root):
    rows = []
    metas = {}
    for fp in sorted(glob.glob(os.path.join(raw_root, "*", "episodes.jsonl"))):
        exp_id = os.path.basename(os.path.dirname(fp))
        mp = os.path.join(os.path.dirname(fp), "meta.json")
        meta = json.load(open(mp, encoding="utf-8")) if os.path.isfile(mp) else {}
        metas[exp_id] = meta
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "steps":
                continue
            r["method"] = canon_method(r.get("method", ""))
            r["_exp_id"] = exp_id
            rows.append(r)
    return rows, metas


def check_provenance(metas) -> list:
    problems = []
    for exp_id, m in metas.items():
        src = str((m.get("extra") or {}).get("threshold_source", ""))
        if "reliability_triggered" in str(m.get("method", "")) and "test" in src.lower():
            problems.append(f"{exp_id}: threshold source mentions test split: {src}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "results", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "analysis"))
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()

    rows, metas = load(a.raw)
    if not rows:
        print("no rows under", a.raw)
        return 1
    problems = check_provenance(metas)
    if problems:
        print("REFUSING: protocol violation(s):")
        for p in problems:
            print(" -", p)
        return 2

    mock_rows = [r for r in rows if not r.get("policy_is_model", False)]
    real = [r for r in rows if r.get("policy_is_model", False)]
    if mock_rows:
        print(f"WARNING: {len(mock_rows)} rows are NOT a real model "
              f"({sorted({r['_exp_id'] for r in mock_rows})}); excluded.")

    groups = defaultdict(list)
    for r in real:
        groups[(r["condition"], r["method"])].append(r)

    conds = [c for c in COND_ORDER if any(k[0] == c for k in groups)]
    conds += sorted({k[0] for k in groups} - set(conds))
    methods = ["chunked_frozen", "fixed_frequency", "reliability_triggered"]

    md = ["# Full test-matrix analysis (recomputed from results/raw/*/episodes.jsonl)",
          "",
          "`SPL_proxy` = mean of `success * d_straight / max(path, d_straight)`; the "
          "denominator ignores walls, so it is an OPTIMISTIC bound, not true SPL.",
          ""]
    out = {"per_cell": {}, "paired": {}, "n_rows": len(real),
           "excluded_mock_rows": len(mock_rows)}

    # ---- coverage gate ---------------------------------------------------
    # A table that silently omits a condition reads like "we ran everything".
    # Name the gap instead.
    expected = ["A_in_dist", "B_env_shift", "C_instr_shift", "D_recovery"]
    present = [c for c in expected if any(k[0] == c for k in groups)]
    missing = [c for c in expected if c not in present]
    out["coverage"] = {"expected": expected, "present": present,
                       "missing": missing, "complete": not missing}
    md += ["## Coverage", "",
           f"- conditions present: {present if present else '(none)'}"]
    if missing:
        md += ["- **INCOMPLETE — missing: " + ", ".join(missing) + "**",
               "- every table below covers only the conditions listed above; do "
               "not read it as the full matrix"]
        print("WARNING: incomplete matrix, missing conditions:", missing)
    else:
        md += ["- all four protocol conditions present"]
    md += [""]

    md += ["| condition | regime | n | SR | Wilson 95% CI | SPL_proxy | "
           "inf/ep | replans/ep | lat mean s | collisions/ep | d_final m |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond in conds:
        for mname in methods:
            rs = groups.get((cond, mname), [])
            if not rs:
                continue
            n = len(rs)
            k = sum(1 for r in rs if r["success"])
            lo, hi = wilson(k, n)
            cell = {
                "n": n, "successes": k, "SR": k / n,
                "wilson95": [round(lo, 4), round(hi, 4)],
                "SPL_proxy": round(float(np.mean([spl_proxy(r) for r in rs])), 4),
                "inferences_per_ep": round(float(np.mean([r["n_inferences"] for r in rs])), 2),
                "replans_per_ep": round(float(np.mean([r["n_replans_triggered"] for r in rs])), 2),
                "latency_mean_s": round(float(np.nanmean([r["infer_latency_s_mean"] for r in rs])), 3),
                "collisions_per_ep": round(float(np.mean([r["n_collisions"] for r in rs])), 2),
                "dist_final_mean_m": round(float(np.mean([r["distance_to_goal_m"] for r in rs])), 3),
                "path_length_mean_m": round(float(np.mean([r["path_length_m"] for r in rs])), 3),
            }
            out["per_cell"][f"{cond}|{mname}"] = cell
            md.append(f"| {cond} | {METHOD_LABELS.get(mname, mname)} | {n} | "
                      f"{cell['SR']:.2f} | [{lo:.2f},{hi:.2f}] | {cell['SPL_proxy']:.3f} | "
                      f"{cell['inferences_per_ep']:.1f} | {cell['replans_per_ep']:.1f} | "
                      f"{cell['latency_mean_s']:.2f} | {cell['collisions_per_ep']:.1f} | "
                      f"{cell['dist_final_mean_m']:.2f} |")

    # ---- pooled across conditions (per regime) --------------------------
    md += ["", "## Pooled across conditions", "",
           "| regime | n | SR | Wilson 95% CI | SPL_proxy | inf/ep |", "|---|---|---|---|---|---|"]
    for mname in methods:
        rs = [r for r in real if r["method"] == mname]
        if not rs:
            continue
        n, k = len(rs), sum(1 for r in rs if r["success"])
        lo, hi = wilson(k, n)
        sp = float(np.mean([spl_proxy(r) for r in rs]))
        ie = float(np.mean([r["n_inferences"] for r in rs]))
        out.setdefault("pooled", {})[mname] = {
            "n": n, "successes": k, "SR": k / n,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "SPL_proxy": round(sp, 4), "inferences_per_ep": round(ie, 2)}
        md.append(f"| {METHOD_LABELS.get(mname, mname)} | {n} | {k / n:.2f} | "
                  f"[{lo:.2f},{hi:.2f}] | {sp:.3f} | {ie:.1f} |")

    # ---- paired comparisons over identical (condition, episode) ---------
    rng = np.random.default_rng(20261030)
    pairs = [("reliability_triggered", "fixed_frequency"),
             ("reliability_triggered", "chunked_frozen"),
             ("fixed_frequency", "chunked_frozen")]
    md += ["", "## Paired comparisons (same episodes; exact McNemar)", "",
           "| condition | A vs B | n pairs | only A | only B | McNemar p | "
           "delta SR (A-B) | 95% bootstrap CI | inf ratio A/B |",
           "|---|---|---|---|---|---|---|---|---|"]
    for cond in conds:
        for a_m, b_m in pairs:
            ra = {(r["episode_id"]): r for r in groups.get((cond, a_m), [])}
            rb = {(r["episode_id"]): r for r in groups.get((cond, b_m), [])}
            ids = sorted(set(ra) & set(rb))
            if not ids:
                continue
            sa = np.array([1.0 if ra[i]["success"] else 0.0 for i in ids])
            sb = np.array([1.0 if rb[i]["success"] else 0.0 for i in ids])
            b = int(np.sum((sa == 1) & (sb == 0)))
            c = int(np.sum((sa == 0) & (sb == 1)))
            p = mcnemar_exact(b, c)
            d = float(sa.mean() - sb.mean())
            deltas = []
            for _ in range(a.boot):
                idx = rng.integers(0, len(ids), len(ids))
                deltas.append(float(sa[idx].mean() - sb[idx].mean()))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            inf_a = float(np.mean([ra[i]["n_inferences"] for i in ids]))
            inf_b = float(np.mean([rb[i]["n_inferences"] for i in ids]))
            ratio = inf_a / inf_b if inf_b else float("nan")
            key = f"{cond}|{a_m}_vs_{b_m}"
            out["paired"][key] = {
                "n_pairs": len(ids), "only_A_success": b, "only_B_success": c,
                "mcnemar_exact_p": round(p, 5),
                "delta_SR": round(d, 4),
                "delta_SR_bootstrap95": [round(float(lo), 4), round(float(hi), 4)],
                "inferences_A": round(inf_a, 2), "inferences_B": round(inf_b, 2),
                "inference_ratio_A_over_B": round(ratio, 3)}
            md.append(f"| {cond} | {a_m.split('_')[0].upper()} vs "
                      f"{b_m.split('_')[0].upper()} | {len(ids)} | {b} | {c} | "
                      f"{p:.3f} | {d:+.2f} | [{lo:+.2f},{hi:+.2f}] | {ratio:.2f} |")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "full_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.out, "full_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n".join(md))
    print("\nwrote", os.path.join(a.out, "full_analysis.md"))
    print("wrote", os.path.join(a.out, "full_analysis.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
