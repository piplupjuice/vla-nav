"""Analyze pilot episode logs: SR with Wilson 95% CI, SPL (straight-line
lower-bound shortest-path proxy), inference counts, latency, collisions.

Reads:  kaggle_notebooks/02_thor_pilot/output/results/raw/<exp_id>/episodes.jsonl
Writes: printed table + kaggle_notebooks/02_thor_pilot/output/pilot_analysis.md

Honesty notes encoded here:
- SPL uses STRAIGHT-LINE start->goal distance as the "shortest path"
  denominator proxy (walls ignored) => it is an OPTIMISTIC lower-bound
  proxy, stated in every output.
- Wilson CI with z=1.96, n small; failures keep the interval honest.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "kaggle_notebooks", "02_thor_pilot", "output",
                   "results", "raw")
OUT_MD = os.path.join(HERE, "kaggle_notebooks", "02_thor_pilot", "output",
                      "pilot_analysis.md")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    files = sorted(glob.glob(os.path.join(RAW, "*", "episodes.jsonl")))
    if not files:
        print("no episode logs found under", RAW)
        return 1
    rows = []
    for fp in files:
        exp_id = os.path.basename(os.path.dirname(fp))
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_exp_id"] = exp_id
            rows.append(r)
    assert rows, "episodes.jsonl existed but had no rows"

    regimes = sorted({r["_exp_id"] for r in rows})
    md = ["# Pilot analysis (recomputed from results/raw/episodes.jsonl)", "",
          "SPL shown is the *optimistic straight-line shortest-path proxy* "
          "(walls ignored): SPL_proxy = mean over episodes of "
          "[ success * d_straight / max(path_taken, d_straight) ].", ""]
    md += ["| regime | n | SR | Wilson 95% CI | SPL_proxy | inference/ep | replans/ep |"
           " lat mean s | collisions/ep | dist_final mean m |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    print(f"{'regime':38s} {'n':>2s} {'SR':>5s} {'CI95':>17s} {'SPLp':>5s}"
          f" {'inf/ep':>6s} {'rp/ep':>5s} {'lat_s':>6s} {'col':>4s} {'dfinal':>6s}")
    for eid in regimes:
        rs = [r for r in rows if r["_exp_id"] == eid]
        n = len(rs)
        k = sum(1 for r in rs if r["success"])
        lo, hi = wilson(k, n)
        spl_vals = []
        for r in rs:
            d = float(r["distance_start_to_goal_m"])
            if r["success"] and d > 1e-6:
                spl_vals.append(d / max(float(r["path_length_m"]), d))
            else:
                spl_vals.append(0.0)
        spl = sum(spl_vals) / n
        inf = sum(r["n_inferences"] for r in rs) / n
        rp = sum(r["n_replans_triggered"] for r in rs) / n
        lat = sum(r["infer_latency_s_mean"] for r in rs) / n
        col = sum(r["n_collisions"] for r in rs) / n
        dfin = sum(float(r["distance_to_goal_m"]) for r in rs) / n
        label = eid.replace("pilot-thor-20261029-", "")
        md.append(f"| {label} | {n} | {k / n:.2f} | [{lo:.2f},{hi:.2f}] |"
                  f" {spl:.3f} | {inf:.1f} | {rp:.1f} | {lat:.2f} |"
                  f" {col:.1f} | {dfin:.2f} |")
        print(f"{label:38s} {n:2d} {k / n:5.2f} [{lo:.2f},{hi:.2f}]{'':>5s}"
              f" {spl:5.3f} {inf:6.1f} {rp:5.1f} {lat:6.2f} {col:4.1f}"
              f" {dfin:6.2f}")

    # environment/model provenance sanity from the rows themselves
    fams = {f"{r.get('env_family')}/model={r.get('policy_is_model')}"
            for r in rows}
    md += ["", f"provenance (env_family/policy_is_model in rows): {sorted(fams)}"]
    print("provenance:", sorted(fams))

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", os.path.relpath(OUT_MD, HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
