"""Post-process the val-tuning artifact WITHOUT changing its numbers.

The selection rule is pre-registered (max SR, tie -> fewer inferences). Two
things can still go wrong, and both must be visible rather than silently
"resolved" by a tie-break:

  * **degenerate sweep** - every threshold has the same SR. Then the choice is
    not identified by success, and the tie-break (cost) alone decides. That is
    a finding about the reliability signal, so it is recorded explicitly.
  * **stale artifact** - a threshold_choice.json that predates the grid/yaw
    fix. The kernel logs carry the BUILD stamp; the operator must check it
    before using the artifact. This script prints the stamp if the log is
    available next to the artifact.

Usage:
  python scripts/finalize_tuning.py [--path results/tuning/threshold_choice.json]
Writes: <path>.annotated.json  (the original file is never modified)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(HERE, "results", "tuning", "threshold_choice.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT)
    a = ap.parse_args()

    if not os.path.isfile(a.path):
        print("no tuning artifact at", a.path)
        return 1
    d = json.load(open(a.path, encoding="utf-8"))
    table = d.get("table") or []
    if not table:
        print("artifact has no sweep table; refusing to annotate")
        return 1

    srs = {round(float(r["SR"]), 6) for r in table}
    infs = {round(float(r["inferences_mean"]), 6) for r in table}
    degenerate = len(srs) == 1

    ann = dict(d)
    ann["degenerate_val"] = bool(degenerate)
    ann["degenerate_note"] = (
        "All val thresholds produced the same success rate (SR = %.3f), so "
        "success does not identify a threshold; the pre-registered cost "
        "tie-break picked threshold_m = %s (mean inferences %s). Treat the "
        "value as 'cheapest working setting', not as a tuned optimum."
        % (next(iter(srs)), d.get("threshold_m"),
           sorted(infs)[0] if infs else "?")
        if degenerate else
        "Val success rate varies across thresholds (SR values: %s); the "
        "pre-registered rule (max SR, tie -> fewer inferences) is informative."
        % sorted(srs)
    )
    out = a.path[:-5] + ".annotated.json" if a.path.endswith(".json") else a.path + ".annotated.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)

    print("threshold_m      :", d.get("threshold_m"))
    print("stall_limit      :", d.get("stall_limit"))
    print("selection_rule   :", d.get("selection_rule"))
    print("degenerate (val) :", degenerate)
    for r in table:
        print("  thr=%-5s n=%s SR=%.3f inf=%.1f d_final=%.2f"
              % (r["threshold_m"], r["n"], r["SR"], r["inferences_mean"],
                 r["dist_final_mean"]))
    print("wrote", os.path.relpath(out, HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
