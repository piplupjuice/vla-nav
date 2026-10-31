"""Evaluation metrics, statistics, and logging utilities.

All metric definitions used by the project are implemented exactly once,
here, and referenced by name from configs and the research report.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------


def _get(r, name, default=0.0):
    """Access a field whether the record is a dict or a dataclass."""
    if isinstance(r, dict):
        return r.get(name, default)
    return getattr(r, name, default)


def success_rate(results: Sequence[dict]) -> float:
    """Fraction of successful episodes. Higher better. Failed/dead count as 0."""
    return float(np.mean([bool(_get(r, "success", False)) for r in results]))


def spl(results: Sequence[dict]) -> float:
    """Success weighted by (normalized inverse) Path Length (per Anderson et al. 2018).

    SPL = (1/N) * sum_i S_i * (s_i / max(p_i, s_i))
    where S_i is success indicator, s_i is the shortest-path success
    episode length (here: straight-line distance / step_distance), p_i is
    actual path length. Episodes with success=False contribute 0.
    """
    vals = []
    for r in results:
        if not bool(_get(r, "success", False)):
            vals.append(0.0)  # failed episodes contribute 0 to the mean over N
            continue
        p = max(float(_get(r, "path_length_m")), 1e-9)
        s = max(float(_get(r, "distance_start_to_goal_m")), 1e-9)
        vals.append(s / max(p, s))
    return float(np.mean(vals)) if vals else 0.0


def collision_rate(results: Sequence[dict]) -> float:
    """Fraction of episodes that ended in a collision-kill. Lower better."""
    return float(np.mean([bool(_get(r, "collision_terminated", False)) for r in results]))


def path_length_m(results: Sequence[dict]) -> float:
    return float(np.mean([_get(r, "path_length_m") for r in results]))


def mean_inference_latency(results: Sequence[dict], p: float = 50.0) -> float:
    key = "infer_latency_s_med" if p == 50 else "infer_latency_s_mean"
    return float(np.nanmean([_get(r, key, np.nan) for r in results]))


def mean_inference_calls(results: Sequence[dict]) -> float:
    return float(np.mean([_get(r, "n_inferences") for r in results]))


def n_replans_total(results: Sequence[dict]) -> int:
    return int(np.sum([_get(r, "n_replans_triggered", 0) for r in results]))


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def ci_success(results, z: float = 1.96) -> Tuple[float, float]:
    k = int(np.sum([bool(_get(r, "success", False)) for r in results]))
    return wilson_ci(k, len(results), z=z)


def paired_bootstrap_ci(a: Sequence[bool], b: Sequence[bool],
                        n_boot: int = 10_000, seed: int = 0,
                        alpha: float = 0.05) -> Dict[str, float]:
    """Bootstrap CI for (mean_a - mean_b) over paired episode outcomes.

    Pair order is fixed: a[i] and b[i] are the same episode under methods A
    and B. Randomness only from resampling (seeded).
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    assert a.shape == b.shape
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "boot_ci_lo": float(lo),
        "boot_ci_hi": float(hi),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, float]:
    """Exact McNemar's test for paired binary outcomes (discords only).

    Returns {"n_discordant", "b_only_success", "a_only_success",
             "p_value", "odds_ratio"}.
    Implemented with no scipy dependency.
    """
    n01 = sum(1 for x, y in zip(a, b) if (not x) and y)   # a fail, b pass
    n10 = sum(1 for x, y in zip(a, b) if x and (not y))  # a pass, b fail
    nd = n01 + n10
    # two-sided binomial test
    p = 0.0
    if nd:
        from math import comb
        k = min(n01, n10)
        for i in range(0, k + 1):
            p += comb(nd, i) * (0.5**nd)
        p *= 2
        p = min(1.0, p)
    return {
        "n_discordant": nd, "a_only_success": n10, "b_only_success": n01,
        "p_value": float(p), "odds_ratio": float(0.0) if p == 0.0 else float(n01),
    }


def cohens_g(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Effect size for paired binary outcomes (like Cohen's d for paired diffs)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Aggregation / comparison table construction
# ---------------------------------------------------------------------------

METRIC_DEFS = {
    "success_rate": dict(higher_is_better=True, unit="fraction",
                         definition="fraction of episodes reaching the goal within radius then terminating"),
    "spl": dict(higher_is_better=True, unit="dimensionless",
                definition="Anderson et al. 2018 SPL with straight-line shortest-path normalization"),
    "collision_rate": dict(higher_is_better=False, unit="fraction",
                           definition="fraction of episodes terminated by the max-collision rule"),
    "path_length_m": dict(higher_is_better=False, unit="meters",
                          definition="mean total odometry path length"),
    "n_steps": dict(higher_is_better=False, unit="ticks",
                    definition="mean number of control ticks"),
    "n_inferences": dict(higher_is_better=False, unit="count",
                         definition="VLA inference calls per episode"),
    "infer_latency_s_med": dict(higher_is_better=False, unit="seconds",
                                definition="median single-inference wall time"),
}
