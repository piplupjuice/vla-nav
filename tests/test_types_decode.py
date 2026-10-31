"""Unit tests for the shared action-decoding chain (no model needed)."""
import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlanav.environment.types import (
    Cmd, WaypointChunk, NUM_ACTIONS_CHUNK, WAYPOINT_SPACING_M, PD_DT,
    MAX_LINEAR_V, MAX_ANGULAR_V, clip_angle,
)


def test_chunk_shape_validation():
    with pytest.raises(ValueError):
        WaypointChunk(np.zeros((4, 4)))
    WaypointChunk(np.zeros((8, 4)))  # OK


def test_pd_controller_straight_line():
    # waypoint 5 steps forward along +x (front axis) in 0.1m units
    chunk = np.zeros((8, 4))
    chunk[:, 0] = 0.0
    chunk[6, 0] = 5.0  # idx 6 chosen => 0.5 m ahead
    c = WaypointChunk(chunk).decode(waypoint_select=6)
    # official: linear = dx/DT = 1.5 -> clip 0.5 -> velocity limitation
    # (maxv=0.3, |angular|<=0.001) -> linear = 0.3, angular = 0
    assert c.linear_v == pytest.approx(0.3)
    assert c.angular_v == pytest.approx(0.0)


def test_pd_controller_pure_rotation():
    chunk = np.zeros((8, 4))
    chunk[:, 0] = 0.0
    chunk[:, 1] = 2.0  # 0.2 m to the LEFT, along +y (left axis)
    chunk[2, 2] = 0.0
    chunk[2, 3] = 1.0
    c = WaypointChunk(chunk).decode(waypoint_select=2)
    # official: dx=0 branch -> linear 0, angular clipped to 1.0, then the
    # velocity-limitation stage caps |angular| at DEPLOY_MAX_ANGULAR_V = 0.3
    assert c.linear_v == 0.0
    assert c.angular_v == pytest.approx(0.3)
    # left = positive angular (counterclockwise, robot convention)


def test_pd_controller_backward_position_turns_in_place():
    chunk = np.zeros((8, 4))
    chunk[3, 0] = -1.0  # behind robot (dx negative)
    chunk[3, 1] = 0.3
    chunk[3, 2] = math.cos(math.pi)
    chunk[3, 3] = 0.0
    c = WaypointChunk(chunk).decode(waypoint_select=3)
    # dx<0 -> linear clipped to 0; official uses plain arctan(dy/dx):
    # atan(0.03 / -0.1) / DT ~ -0.8745 (within +-1), then velocity
    # limitation: |linear|=0 <= maxv, |angular|>0.3 -> rd=0, angular=-0.3
    assert c.linear_v == 0.0
    assert c.angular_v == pytest.approx(0.3 * np.sign(math.atan(0.3 / -1.0)))


def test_clip_angle_range():
    rng = np.random.default_rng(0)
    for th in rng.uniform(-10, 10, 200):
        v = clip_angle(float(th))
        assert -math.pi <= v <= math.pi


def test_constants_are_official():
    # These constants are copied verbatim from OmniVLA run_omnivla.py.
    from vlanav.environment.types import DEPLOY_MAX_LINEAR_V, DEPLOY_MAX_ANGULAR_V
    assert NUM_ACTIONS_CHUNK == 8
    assert WAYPOINT_SPACING_M == 0.1
    assert PD_DT == 1 / 3
    assert MAX_LINEAR_V == 0.5
    assert MAX_ANGULAR_V == 1.0
    assert DEPLOY_MAX_LINEAR_V == 0.3     # "Velocity limitation" maxv
    assert DEPLOY_MAX_ANGULAR_V == 0.3    # "Velocity limitation" maxw
