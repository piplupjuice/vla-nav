"""Guards that must exist in the experiment scripts that ran on Kaggle.

Each assertion here corresponds to a real failure that cost GPU hours
(risk register R13-R25). The scripts under experiments/ are the exact code
that executed on the Kaggle workers, so testing them is testing the run.
"""
from __future__ import annotations

import json
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENTS = [
    "experiments/run_01_thor.py",
    "experiments/run_02_thor_pilot.py",
    "experiments/run_03_tuning.py",
    "experiments/run_04a_test_ab.py",
    "experiments/run_04b_test_cd.py",
    "experiments/run_06_demo.py",
    "experiments/run_07_thor_diag.py",
]

#: scripts that load the 15 GB checkpoint
MODEL_NOTEBOOKS = [n for n in EXPERIMENTS
                   if any(k in n for k in ("02_thor_pilot", "03_tuning",
                                           "04a", "04b", "06_demo"))]


def _text(path: str) -> str:
    return open(os.path.join(HERE, path), encoding="utf-8").read()


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_every_notebook_has_a_resolver_that_refuses_ambiguity(path):
    txt = _text(path)
    if "vlanav" not in txt:              # nb01 / nb07 do not import the package
        pytest.skip("does not use the dataset package")
    assert "/kaggle/input" in txt
    assert "ambiguous" in txt or "cands" in txt


@pytest.mark.parametrize("path", MODEL_NOTEBOOKS)
def test_model_notebooks_verify_the_checkpoint_and_retry_once(path):
    txt = _text(path)
    assert "force_download=(_attempt == 2)" in txt      # R23
    assert "os.path.getsize" in txt                     # size check, not just isfile


@pytest.mark.parametrize("path", MODEL_NOTEBOOKS)
def test_model_notebooks_refuse_a_stale_dataset(path):
    txt = _text(path)
    assert "contains_RotateLeft=True" in txt            # R21
    assert "contains_D_recovery_yaw=True" in txt


@pytest.mark.parametrize("path", ["experiments/run_04a_test_ab.py",
                                  "experiments/run_04b_test_cd.py",
                                  "experiments/run_06_demo.py"])
def test_long_runs_gate_on_real_movement(path):
    txt = _text(path)
    assert "_smoke.path_length_m > 0.02" in txt          # R22


@pytest.mark.parametrize("path", ["experiments/run_04a_test_ab.py",
                                  "experiments/run_04b_test_cd.py"])
def test_test_matrix_logs_the_instruction_and_scene_per_row(path):
    txt = _text(path)
    assert 'row["instruction"] = spec.instruction' in txt
    assert 'row["scene"] = spec.extra.get("scene")' in txt


def test_controller_is_always_built_with_the_actuation_grid():
    # nb07 evidence: THOR's default 0.25 m grid makes MoveAhead(0.1) a no-op
    for path in MODEL_NOTEBOOKS:
        txt = _text(path)
        if "gridSize" in txt:
            continue
        pytest.fail(f"{path}: controller build lacks gridSize=0.05")


def test_test_matrix_uses_canonical_regime_names():
    # dry-run regression: class-derived names silently dropped B2 from summaries
    for path in ("experiments/run_04a_test_ab.py",
                 "experiments/run_04b_test_cd.py"):
        txt = _text(path)
        assert '"fixed_frequency"' in txt
        assert "FixedFrequencyReplan(policy, \"fixed_frequency\")" in txt
