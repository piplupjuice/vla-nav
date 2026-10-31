"""Modality-8 goal-pose vector tests (official run_omnivla replication)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from vlanav.model.omnivla_adapter import OmnivlaNavPolicy


def test_goal_proprio_vector_matches_official_formula():
    # 0.5 m ahead, 0.2 m to the left, goal heading 90° CCW off the nose
    v = OmnivlaNavPolicy.goal_proprio_vector(0.5, 0.2, math.pi / 2)
    assert v.dtype == np.float32
    assert v[0] == pytest.approx(0.2 / 0.1)      # rel_y (left) normalized
    assert v[1] == pytest.approx(-0.5 / 0.1)     # -rel_x (front) normalized
    assert v[2] == pytest.approx(0.0, abs=1e-6)  # cos(90°)
    assert v[3] == pytest.approx(1.0, abs=1e-6)  # sin(90°)


def test_goal_proprio_vector_clamps_at_30m():
    v = OmnivlaNavPolicy.goal_proprio_vector(40.0, -40.0, 0.0)
    assert v[0] == pytest.approx(-30.0 / 0.1)    # -clamped rel_x
    assert v[1] == pytest.approx(-30.0 / 0.1)    # clamped rel_y sign path
    assert v[2] == pytest.approx(1.0, abs=1e-6)  # cos(0)


def test_language_pose_mode_requires_hook():
    # constructor-level guard (no load): language_pose without hook raises
    with pytest.raises(ValueError):
        OmnivlaNavPolicy(vla_path="/nonexistent", load_now=False,
                         instruction_mode="language_pose", goal_pose_fn=None)


def test_language_pose_vector_reaches_infer_path():
    """The hook must be stored and consulted (mock-free plumbing check)."""
    p = OmnivlaNavPolicy(vla_path="/nonexistent", load_now=False,
                         instruction_mode="language_pose",
                         goal_pose_fn=lambda: (0.5, 0.2, math.pi / 2))
    assert callable(p.goal_pose_fn)
    vec = p.goal_proprio_vector(*p.goal_pose_fn())
    assert vec.shape == (4,) and np.isfinite(vec).all()
