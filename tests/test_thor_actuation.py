"""Tests for the AI2-THOR actuation conventions fixed on 2026-10-30.

nb07 diagnostic evidence:
  * default movement grid 0.25 m -> `MoveAhead(0.1)` moved 0.0 m (stall bug);
    `gridSize=0.05` moves 0.10 m exactly.
  * THOR yaw is 0 at +z and grows clockwise (RotateLeft -> dyaw negative),
    while the pipeline uses 0 = +x, CCW.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

from vlanav.environment.ai2thor_env import (Ai2ThorNavEnv, thor_yaw_from_math_deg)


class _FakeEvent:
    def __init__(self, x, z, yaw_deg):
        self.metadata = {"agent": {"position": {"x": x, "y": 0.9, "z": z},
                                   "rotation": {"x": 0.0, "y": yaw_deg, "z": 0.0}},
                         "lastActionSuccess": True}


def _pose(yaw_deg):
    env = Ai2ThorNavEnv(controller=None)
    return env._pose_from(_FakeEvent(1.0, 2.0, yaw_deg))


def test_thor_yaw_zero_points_along_plus_z():
    # THOR yaw 0 = facing +z; math heading for +z is +pi/2
    _, _, yaw = _pose(0.0)
    assert abs(yaw - math.pi / 2) < 1e-9


def test_thor_yaw_90_points_along_plus_x():
    _, _, yaw = _pose(90.0)
    assert abs(yaw - 0.0) < 1e-9


def test_thor_yaw_180_points_along_minus_z():
    # math heading for -z is -pi/2 (sin = -1)
    _, _, yaw = _pose(180.0)
    assert abs(yaw + math.pi / 2) < 1e-9


def test_position_mapping_is_x_and_z():
    x, z, _ = _pose(0.0)
    assert (x, z) == (1.0, 2.0)


def test_math_to_thor_yaw_roundtrip():
    for math_deg in (0.0, 45.0, 90.0, 123.4, 270.0, 359.0):
        thor = thor_yaw_from_math_deg(math_deg)
        # converting that THOR yaw back to math must reproduce the heading
        back = (90.0 - thor) % 360.0
        assert abs(((back - math_deg + 180.0) % 360.0) - 180.0) < 1e-9


def test_recovery_heading_is_opposite_the_goal():
    # math heading toward the goal is 0 deg (+x); D_recovery must face 180 deg
    thor = thor_yaw_from_math_deg(180.0)
    math_back = (90.0 - thor) % 360.0
    assert abs(math_back - 180.0) < 1e-9


def test_actuation_grid_requirement_is_documented():
    env = Ai2ThorNavEnv(controller=None)
    props = env.properties()
    assert "gridSize=0.05" in props["actuation_grid_note"]
    assert props["control_dt_s"] > 0
