"""Unit tests for the analysis helpers used in the results documents.

These are the numbers quoted in docs/results.md, so the statistics functions
themselves are tested (not just the notebook cells).
"""
from __future__ import annotations

import importlib.util
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str):
    path = os.path.join(HERE, "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location("vlanav_script_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


full = _load("analyze_full")
pilot = _load("analyze_pilot")


def test_wilson_bounds_are_sane():
    lo, hi = full.wilson(0, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert abs(hi - 0.2775) < 5e-3          # classic Wilson upper bound
    lo2, hi2 = full.wilson(10, 10)
    assert abs(lo2 - 0.7225) < 5e-3 and hi2 <= 1.0
    assert full.wilson(0, 0) == (0.0, 0.0)


def test_wilson_matches_pilot_implementation():
    for k, n in ((3, 10), (0, 6), (6, 6), (17, 40)):
        a = full.wilson(k, n)
        b = pilot.wilson(k, n)
        assert abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12


def test_mcnemar_exact_known_values():
    assert full.mcnemar_exact(0, 0) == 1.0
    # b=1, c=9 -> 2*(1+10)/2^10
    assert abs(full.mcnemar_exact(1, 9) - 0.021484375) < 1e-12
    # symmetric in (b, c)
    assert abs(full.mcnemar_exact(7, 2) - full.mcnemar_exact(2, 7)) < 1e-12
    # balanced discordance -> p == 1
    assert full.mcnemar_exact(5, 5) == 1.0


def test_spl_proxy_ignores_failures_and_short_paths():
    ok = {"success": True, "distance_start_to_goal_m": 4.0, "path_length_m": 8.0}
    short = {"success": True, "distance_start_to_goal_m": 4.0, "path_length_m": 2.0}
    fail = {"success": False, "distance_start_to_goal_m": 4.0, "path_length_m": 8.0}
    assert abs(full.spl_proxy(ok) - 0.5) < 1e-12
    assert full.spl_proxy(short) == 1.0     # never above 1 (max(path, straight))
    assert full.spl_proxy(fail) == 0.0


def test_leak_guard_flags_test_threshold_source():
    metas = {
        "exp-ok": {"method": "reliability_triggered",
                   "extra": {"threshold_source": "/kaggle/input/t/results/tuning/x.json"}},
        "exp-bad": {"method": "reliability_triggered",
                    "extra": {"threshold_source": "test split sweep"}},
        "exp-b1": {"method": "chunked_frozen", "extra": {}},
    }
    problems = full.check_provenance(metas)
    assert problems == ["exp-bad: threshold source mentions test split: test split sweep"]


def test_method_labels_cover_all_three_regimes():
    assert set(full.METHOD_LABELS) == {
        "chunked_frozen", "fixed_frequency", "reliability_triggered"}


def test_canon_method_fixes_auto_derived_b2_name():
    # dry-run regression (2026-10-30): FixedFrequencyReplan derives
    # "fixed_frequency_replan", which the analyzers do not key on.
    assert full.canon_method("fixed_frequency_replan") == "fixed_frequency"
    assert full.canon_method("fixed_frequency") == "fixed_frequency"
    assert full.canon_method("chunked_frozen") == "chunked_frozen"
    assert full.canon_method("reliability_triggered") == "reliability_triggered"
