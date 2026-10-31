"""Integration test: full episode loop with the MOCK policy (NOT a VLA).

Marked with `mock` so it can never be confused with evidence that the real
pretrained VLA works. See the `is_model` flags in all results.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from vlanav.model.mock_policy import MockNavPolicy
from vlanav.environment.mininav_env import MiniNavEnv
from vlanav.environment.episodes import make_episode, list_episodes
from vlanav.reliability.replanning import (
    ChunkedFrozen, FixedFrequencyReplan, ReliabilityTriggered,
)
from vlanav.evaluation.logging import EpisodeLogger, ExperimentMeta
from vlanav.evaluation.metrics import success_rate, spl


@pytest.mark.mock
def test_full_episode_mock_end_to_end(tmp_path):
    env = MiniNavEnv(img_size=64)
    spec = make_episode("A_in_dist", "val", 0)
    goal = spec.goal_xy
    policy = MockNavPolicy(goal_xy=goal, clearance_env=env)   # privileged goal (mock!)
    assert policy.is_model is False
    regime = ReliabilityTriggered(policy, threshold_m=0.08, stall_limit=2)
    res = regime.run_episode(env, spec)
    assert res.success, f"mock should succeed on shipped val episode 0: {res}"
    assert res.method == "reliability_triggered"
    assert res.policy_is_model is False


@pytest.mark.mock
def test_val_split_pilot_all_conditions(tmp_path):
    """Three regimes over the full val split with identical episodes."""
    conds = ["A_in_dist", "B_env_shift", "C_instr_shift"]
    specs = list_episodes(conds, split="val")
    env = MiniNavEnv(img_size=64)
    results = {m: [] for m in ("chunked_frozen", "fixed_frequency", "reliability_triggered")}
    for spec in specs:
        env = MiniNavEnv(img_size=64)
        policy = MockNavPolicy(goal_xy=spec.goal_xy, clearance_env=env)
        for name, regime in [
            ("chunked_frozen", ChunkedFrozen(policy)),
            ("fixed_frequency", FixedFrequencyReplan(policy)),
            ("reliability_triggered", ReliabilityTriggered(policy, threshold_m=0.08, stall_limit=2)),
        ]:
            run_env = MiniNavEnv(64)
            r = regime.run_episode(run_env, spec)
            assert r.episode_id == spec.episode_id
            results[name].append(r)
    # the goal-aware mock with the privileged goal should mostly succeed
    # (B_env_shift has deliberate extra clutter, which can stall straight paths)
    for name, rows in results.items():
        assert success_rate(rows) >= 0.5, name
        assert spl(rows) >= 0.3, name
    # fairness invariant: same episode set for every regime
    ids = {name: [r.episode_id for r in rows] for name, rows in results.items()}
    assert ids["chunked_frozen"] == ids["fixed_frequency"] == ids["reliability_triggered"]
    # logging round trip, refuse-overwrite rule
    meta = ExperimentMeta(experiment_id="unit-mock-val", method="mock",
                          env_family="mininav", conditions=conds, split="val")
    log = EpisodeLogger(meta, str(tmp_path / "results" / "raw" / "unit-mock-val"))
    for r in results["chunked_frozen"]:
        log.log_episode(r)
    recs = [json.loads(l) for l in open(log.path, encoding="utf-8")]
    assert len(recs) == len(specs)
    assert all(x["policy_is_model"] is False for x in recs)
    with pytest.raises(FileExistsError):
        EpisodeLogger(meta, str(tmp_path / "results" / "raw" / "unit-mock-val"))


@pytest.mark.mock
def test_seeded_determinism_across_runs():
    """Identical spec + seed yields identical trajectory and metrics."""
    spec = make_episode("A_in_dist", "val", 0)
    # NOTE: mock policy also depends on floating point deterministic ops
    runs = []
    for _ in range(3):
        env = MiniNavEnv(img_size=64)
        reg = ChunkedFrozen(MockNavPolicy(goal_xy=spec.goal_xy))
        res = reg.run_episode(env, spec)
        runs.append((res.step, res.path_length_m, res.n_inferences))
    assert runs[0] == runs[1] == runs[2]
