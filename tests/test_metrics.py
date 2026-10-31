"""Metric, statistics, and logging correctness tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from vlanav.evaluation.metrics import (
    success_rate, spl, wilson_ci, paired_bootstrap_ci, mcnemar_exact, cohens_g,
    collision_rate, mean_inference_calls,
)


def _res(n, succ_n, path=10.0, dist=8.0, coll=0):
    return [
        dict(success=i < succ_n, path_length_m=path, distance_start_to_goal_m=dist,
             collision_terminated=i < coll, n_inferences=3,
             infer_latency_s_med=0.5, infer_latency_s_mean=0.5)
        for i in range(n)
    ]


def test_spl_formula_hand_case():
    # perfect path: straight-line = path length -> SPL = success rate
    r = _res(4, 2, path=8.0, dist=8.0)
    assert spl(r) == pytest.approx(0.5)
    # path 2x longer -> SPL smeared by 8/10 = 0.8 -> total 2*0.8/4 = 0.4
    r = _res(4, 2, path=10.0, dist=8.0)
    assert spl(r) == pytest.approx(0.4)
    # failed episodes must contribute exactly 0
    r = _res(2, 0)
    assert spl(r) == 0.0


def test_success_and_collision_rates():
    r = _res(10, 4, coll=3)
    assert success_rate(r) == pytest.approx(0.4)
    assert collision_rate(r) == pytest.approx(0.3)


def test_wilson_ci_known_value():
    # Wilson for 8/10 with z=1.96 -> approx [0.4901, 0.9433]
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4901, abs=0.005)
    assert hi == pytest.approx(0.9433, abs=0.005)


def test_paired_bootstrap_extremes():
    a = [True] * 10 + [False] * 10
    b = [False] * 20
    out = paired_bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert out["mean_diff"] == pytest.approx(0.5)
    assert out["boot_ci_lo"] >= 0.25
    assert out["boot_ci_hi"] <= 0.75


def test_mcnemar_discords_exact():
    a = [True, False, True, False, True]
    b = [False, True, True, False, False]  # discords: (T,F),(F,T),(T,F)
    m = mcnemar_exact(a, b)
    assert m["n_discordant"] == 3
    assert m["a_only_success"] == 2
    assert m["b_only_success"] == 1
    # exact two-sided binomial for nd=3: 2*[C(3,0)+C(3,1)]*0.5^3 = 1.0
    assert m["p_value"] == pytest.approx(1.0, abs=1e-9)
    # no-discordant-imbalance case (2 vs 2): p = 1.0
    m2 = mcnemar_exact([True, True, False, False, True], [True, False, True, True, False])
    # (T,T) same; (T,F)→a_only; (F,T)→b_only; (F,T)→b_only; (T,F)→a_only
    assert m2["n_discordant"] == 4
    assert m2["a_only_success"] == 2 and m2["b_only_success"] == 2
    assert m2["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_cohens_g_zero_when_identical():
    g = cohens_g([True] * 4, [True] * 4)
    assert np.isnan(g)


def test_mean_inference_calls():
    r = _res(5, 3)
    r[0]["n_inferences"] = 10
    assert mean_inference_calls(r) == pytest.approx((10 + 4 * 3) / 5)
