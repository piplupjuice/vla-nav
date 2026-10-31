"""Reliability signal + trigger logic tests (mock-driven; clearly labeled)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from vlanav.reliability.progress import ProgressTracker, intended_displacement_m
from vlanav.reliability.replanning import (
    ChunkedFrozen, FixedFrequencyReplan, ReliabilityTriggered,
)
from vlanav.model.mock_policy import MockNavPolicy
from vlanav.environment.mininav_env import MiniNavEnv
from vlanav.environment.episodes import make_episode


def test_progress_discrepancy_math():
    chunks = np.zeros((8, 4))
    chunks[4, 0] = 4.0  # 0.4 m intended
    intended = intended_displacement_m(chunks[4])
    assert intended == pytest.approx(0.4)
    t = ProgressTracker()
    d = t.update(intended, observed_m=0.25)
    assert d == pytest.approx(0.15)
    d = t.update(intended, observed_m=0.0)
    assert d == pytest.approx(0.4)
    assert t.stall_run == 1
    d = t.update(intended, observed_m=0.0)
    assert t.stall_run == 2


def test_trigger_fires_on_repeated_stall():
    env = MiniNavEnv(img_size=64)
    spec = make_episode("A_in_dist", "val", 0)
    goal = spec.goal_xy
    # Mock the "policy goes broken after 5 inferences" — stalled trigger should cap it
    policy = MockNavPolicy(goal_xy=goal, fail_from_call=3)
    r = ReliabilityTriggered(policy, threshold_m=0.08, stall_limit=2,
                             max_replans_per_episode=4)
    res = r.run_episode(env, spec)
    # after call #3 the mock stops progressing → trigger must fire, capped
    assert res.n_replans_triggered > 0
    assert res.n_replans_triggered <= 4


def test_inference_budget_ordering():
    """B1 infers roughly every-8 ticks; B2 infers every tick; B3 between."""
    env = MiniNavEnv(64)
    goal = make_episode("A_in_dist", "val", 0).goal_xy
    policy = MockNavPolicy(goal_xy=goal, clearance_env=env)
    b1 = ChunkedFrozen(policy).run_episode(MiniNavEnv(64), make_episode("A_in_dist", "val", 0))
    b2 = FixedFrequencyReplan(policy).run_episode(MiniNavEnv(64), make_episode("A_in_dist", "val", 0))
    b3 = ReliabilityTriggered(policy, threshold_m=0.08, stall_limit=2).run_episode(
        MiniNavEnv(64), make_episode("A_in_dist", "val", 0))
    assert b2.n_inferences >= b1.n_inferences  # B2 is the strict upper bound
    assert b3.n_inferences >= b1.n_inferences
    # B3 must not exceed B2 (implied: B3 ≤ B2 by construction, unless max replan cap)
    assert b3.n_inferences <= b2.n_inferences * 1.00001
    assert b1.n_inferences <= np.ceil(b1.step / 8) or b1.n_inferences == np.ceil(b1.step / 8)
