"""Evidence gate: verify a fetched results tree before quoting any number.

Checks (each is a hard failure, not a warning):
  1. every row has the required fields and `policy_is_model` is true;
  2. `env_family` is `ai2thor` (not the minimock fallback);
  3. `split` is the expected one (test for the full matrix, val for tuning);
  4. episode ids are unique inside an experiment;
  5. meta.json exists, records a model name and a threshold source for B3;
  6. inference counts and latencies are physically plausible (> 0, and
     deliberately slow enough to be the real 8B model: mean latency >= 0.5 s).

Usage: python scripts/verify_raw.py results/raw --split test
Exit code 0 = all experiments pass; 1 = at least one violated.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

METHODS = {"chunked_frozen", "fixed_frequency_replan", "fixed_frequency",
           "reliability_triggered"}
REQUIRED = ["episode_id", "method", "condition", "split", "seed", "step",
            "success", "path_length_m", "distance_to_goal_m",
            "distance_start_to_goal_m", "n_inferences", "n_replans_triggered",
            "infer_latency_s_mean", "n_collisions", "policy_is_model"]


def check_exp(exp_dir: str, split: str) -> list:
    errs = []
    name = os.path.basename(exp_dir)
    ep = os.path.join(exp_dir, "episodes.jsonl")
    mp = os.path.join(exp_dir, "meta.json")
    if not os.path.isfile(ep):
        return [f"{name}: episodes.jsonl missing"]
    if not os.path.isfile(mp):
        errs.append(f"{name}: meta.json missing")
        meta = {}
    else:
        meta = json.load(open(mp, encoding="utf-8"))
        if not meta.get("model_name"):
            errs.append(f"{name}: meta.model_name empty")
        method = meta.get("method", "")
        if "reliability" in method and not (
                meta.get("extra") or {}).get("threshold_source"):
            errs.append(f"{name}: B3 meta lacks extra.threshold_source")

    seen = set()
    n = 0
    for line in open(ep, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") == "steps":
            continue
        n += 1
        for k in REQUIRED:
            if k not in r:
                errs.append(f"{name}: row missing {k}")
                break
        if r.get("policy_is_model") is not True:
            errs.append(f"{name}: row {r.get('episode_id')} is NOT a real model")
        if r.get("env_family") != "ai2thor":
            errs.append(f"{name}: env_family={r.get('env_family')} (expected ai2thor)")
        if r.get("split") != split:
            errs.append(f"{name}: split={r.get('split')} (expected {split})")
        if r.get("method") not in METHODS:
            errs.append(f"{name}: unexpected method {r.get('method')}")
        if r["episode_id"] in seen:
            errs.append(f"{name}: duplicate episode_id {r['episode_id']}")
        seen.add(r["episode_id"])
        if int(r.get("n_inferences", 0)) <= 0:
            errs.append(f"{name}: episode {r['episode_id']} has no inference")
        lat = float(r.get("infer_latency_s_mean", 0.0))
        if not (0.5 <= lat <= 30.0):
            errs.append(f"{name}: mean latency {lat:.2f}s is implausible for the "
                        f"real 8B model")
    if n == 0:
        errs.append(f"{name}: no episode rows")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    a = ap.parse_args()
    exps = sorted({os.path.dirname(p)
                   for p in glob.glob(os.path.join(a.root, "**", "episodes.jsonl"),
                                      recursive=True)})
    if not exps:
        print("no experiments under", a.root)
        return 1
    bad = []
    passed = 0
    for d in exps:
        errs = check_exp(d, a.split)
        print(("PASS " if not errs else "FAIL ") + os.path.basename(d))
        for e in errs:
            print("   -", e)
        bad += errs
        passed += (not errs)
    print(f"\n{passed}/{len(exps)} experiments pass")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
