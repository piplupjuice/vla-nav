"""RAAR (P1) + budget-matched baselines: mock-driven tests, no real model.

These tests prove the MECHANICS (signals, cooldown, caps, budget matching,
timing/recovery bookkeeping) on MiniNav + MockNavPolicy. They are labeled
mock: their results are never evidence about a VLA.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math

import numpy as np
import pytest

from vlanav.reliability.raar import (
    RAARConfig, ReliabilityMonitor, RAARTriggered, RandomBudgetReplan,
    FixedBudgetReplan, CollisionOracleTriggered, calibrate_budget_match,
)
from vlanav.reliability.replanning import (
    ChunkedFrozen, FixedFrequencyReplan, ReliabilityTriggered,
)
from vlanav.model.mock_policy import MockNavPolicy
from vlanav.environment.mininav_env import MiniNavEnv
from vlanav.environment.episodes import make_episode

pytestmark = pytest.mark.mock


# ----------------------------------------------------------------------
# ReliabilityMonitor unit behavior
# ----------------------------------------------------------------------
def test_monitor_motion_error_rises_when_blocked():
    m = ReliabilityMonitor(RAARConfig(motion_scale=0.1))
    s = m.update(intended_m=0.1, observed_m=0.0)
    assert s > 0.0                      # blocked step -> positive score
    for _ in range(5):
        s = m.update(intended_m=0.1, observed_m=0.0)
    assert m.last_e_raw == pytest.approx(0.1)
    assert s >= 1.0                     # normalized error saturates at 1


def test_monitor_quiet_when_moving():
    m = ReliabilityMonitor(RAARConfig(motion_scale=0.1))
    for _ in range(10):
        s = m.update(intended_m=0.1, observed_m=0.1)
    assert s < 0.05 and m._stall_run == 0


def test_monitor_stall_indicator_needs_consecutive_ticks():
    cfg = RAARConfig(stall_limit=2)
    m = ReliabilityMonitor(cfg)
    m.update(0.1, 0.0)                 # EMA(0.5): ema=0.05 -> S1=0.5, S2=0
    assert m.last_score == pytest.approx(0.5)
    m.update(0.1, 0.0)                 # ema=0.075 -> S1=0.75, S2=1
    assert m.last_score == pytest.approx(1.75)


def test_monitor_progress_signal_requires_goal_distances():
    cfg = RAARConfig(w_p=1.0)
    with pytest.raises(ValueError):
        RAARTriggered(policy=None, config=cfg)  # no goal_distance_fn hook
    m = ReliabilityMonitor(cfg)
    for _ in range(5):
        m.update(0.1, 0.1, goal_dist=3.0)   # goal distance frozen -> no progress
    assert m.last_score >= 1.0              # S3 = 1 fires
    m2 = ReliabilityMonitor(cfg)
    for i in range(5):
        m2.update(0.1, 0.1, goal_dist=3.0 - 0.05 * (i + 1))  # real progress
    assert m2.last_score < 0.5              # S3 = 0


# ----------------------------------------------------------------------
# Trigger mechanics on real episode loops
# ----------------------------------------------------------------------
def _run(raar_cls, **kw):
    env = MiniNavEnv(64)
    spec = make_episode("A_in_dist", "val", 0)
    policy = MockNavPolicy(goal_xy=spec.goal_xy, fail_from_call=3)
    reg = raar_cls(policy, **kw)
    return reg.run_episode(env, spec), reg


def test_raar_triggers_and_respects_cap():
    cfg = RAARConfig(tau=0.1, cooldown=1, max_replans_per_episode=6)
    res, _ = _run(RAARTriggered, config=cfg)
    assert res.n_replans_triggered > 0
    assert res.n_replans_triggered <= 6


def test_raar_cooldown_enforces_min_spacing():
    # tau tiny -> every eligible tick wants to trigger; cooldown must space
    # triggered replans out (asserted via the monitor's replay below).
    cfg = RAARConfig(tau=-1.0, cooldown=3, max_replans_per_episode=10 ** 6)
    env = MiniNavEnv(64)
    spec = make_episode("A_in_dist", "val", 0)
    policy = MockNavPolicy(goal_xy=spec.goal_xy, fail_from_call=1)
    reg = RAARTriggered(policy, config=cfg)
    res = reg.run_episode(env, spec)
    assert res.n_replans_triggered > 0
    # replay the loop length: with cooldown c, max possible triggered
    # replans over res.step ticks is ~ step / (c + 1) + 1
    assert res.n_replans_triggered <= res.step // cfg.cooldown + 1


def test_raar_budget_between_chunked_and_every_step():
    spec = make_episode("A_in_dist", "val", 0)
    env = MiniNavEnv(64)
    policy = MockNavPolicy(goal_xy=spec.goal_xy, clearance_env=env)
    b1 = ChunkedFrozen(policy).run_episode(MiniNavEnv(64), spec)
    raar = RAARTriggered(policy, config=RAARConfig()).run_episode(
        MiniNavEnv(64), spec)
    b2 = FixedFrequencyReplan(policy).run_episode(MiniNavEnv(64), spec)
    assert b1.n_inferences <= raar.n_inferences <= b2.n_inferences


def test_random_budget_deterministic_and_capped():
    r1, _ = _run(RandomBudgetReplan, fire_p=0.5, max_replans_per_episode=5)
    r2, _ = _run(RandomBudgetReplan, fire_p=0.5, max_replans_per_episode=5)
    assert r1.n_inferences == r2.n_inferences       # seeded per episode
    assert r1.n_replans_triggered <= 5              # cap honored
    with pytest.raises(ValueError):
        RandomBudgetReplan(MockNavPolicy(goal_xy=[0, 0]), fire_p=1.5)


def test_fixed_budget_interval():
    res, _ = _run(FixedBudgetReplan, interval_ticks=3)
    assert res.n_inferences >= res.step / 3 - 1
    with pytest.raises(ValueError):
        FixedBudgetReplan(MockNavPolicy(goal_xy=[0, 0]), interval_ticks=0)


def test_oracle_fires_on_collision():
    # A mock WITHOUT clearance checks aims straight at the goal; in a
    # cluttered (B_env_shift) scene the straight segment is usually blocked,
    # so the robot really collides and the oracle must re-query.
    found = False
    candidates = [("val", i) for i in range(12)] +                  [("test", i) for i in range(32)]
    for split, idx in candidates:
        env = MiniNavEnv(64)
        spec = make_episode("B_env_shift", split, idx)
        policy = MockNavPolicy(goal_xy=spec.goal_xy)   # clearance_env=None
        res = CollisionOracleTriggered(policy).run_episode(env, spec)
        if res.n_collisions > 0:
            assert res.n_replans_triggered > 0
            found = True
            break
    assert found, "no cluttered B_env_shift episode produced a collision"


# ----------------------------------------------------------------------
# Budget calibration arithmetic (validation-only, frozen before test)
# ----------------------------------------------------------------------
def test_calibrate_budget_match():
    out = calibrate_budget_match(target_queries_per_ep=16.0,
                                 mean_steps=48.0, chunk_ticks=8)
    assert out["fixed_interval_ticks"] == 3          # 48/16
    assert 0.0 <= out["random_fire_p"] <= 1.0
    out2 = calibrate_budget_match(target_queries_per_ep=200.0,
                                  mean_steps=48.0)
    assert out2["random_fire_p"] == 1.0              # clipped, never >1
    out3 = calibrate_budget_match(target_queries_per_ep=2.0, mean_steps=48.0)
    assert out3["fixed_interval_ticks"] == 24


# ----------------------------------------------------------------------
# New timing / recovery / SPL bookkeeping in the shared loop
# ----------------------------------------------------------------------
def test_timing_split_and_recovery_fields_logged():
    res, _ = _run(RAARTriggered, config=RAARConfig())
    assert res.vla_query_count == res.n_inferences
    assert res.vla_latency_seconds == pytest.approx(res.wall_time_s)
    assert res.total_wall_time_s >= res.simulator_time_s + res.vla_latency_seconds - 1e-6
    assert res.controller_time_s >= 0.0
    assert res.n_stall_onsets >= 0
    assert math.isnan(res.recovery_time_steps_mean) or res.n_recoveries > 0


def test_spl_standard_nan_on_mininav():
    # MiniNav has no shortest-path provider -> spl_standard must stay NaN,
    # never silently fall back to the straight-line proxy
    res, _ = _run(RAARTriggered, config=RAARConfig())
    assert math.isnan(res.shortest_path_length_m)
    assert math.isnan(res.spl_standard)


def test_legacy_b3_still_works_after_loop_changes():
    res, _ = _run(ReliabilityTriggered, threshold_m=0.08, stall_limit=2)
    assert res.n_replans_triggered > 0
    assert res.vla_query_count == res.n_inferences
