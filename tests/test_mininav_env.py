"""MiniNav environment determinism, kinematics, collision and benchmark tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from vlanav.environment.mininav_env import MiniNavEnv, AGENT_RADIUS_M, WORLD_SIZE_M
from vlanav.environment.episodes import make_episode, list_episodes
from vlanav.environment.types import Cmd


def _spec(cond="A_in_dist", split="val", idx=0, seed=None):
    s = make_episode(cond, split, idx)
    return s


def test_reset_deterministic_image_and_pose(seed=12345):
    e1, e2 = MiniNavEnv(img_size=64), MiniNavEnv(img_size=64)
    s1, s2 = _spec(), _spec()
    o1, o2 = e1.reset(s1), e2.reset(s2)
    assert o1.image.shape == o2.image.shape
    assert np.array_equal(o1.image, o2.image)
    assert np.allclose(o1.robot_pose, o2.robot_pose)


def test_forward_motion_and_blocked_collision():
    env = MiniNavEnv(img_size=64)
    spec = _spec()
    env.reset(spec)
    yaw = env._p[2]
    info = env.step(Cmd(linear_v=0.3, angular_v=0.0))
    moved = info.moved_distance_m
    assert moved == pytest.approx(0.3 * (1 / 3))
    # colliding with a wall: run toward boundary
    for _ in range(200):
        info = env.step(Cmd(linear_v=0.4, angular_v=0.0))
    assert env._collision_total > 0


def test_turn_moves_direction():
    env = MiniNavEnv(img_size=64)
    env.reset(_spec())
    x0, y0, _ = env._p
    env.step(Cmd(0.3, 1.0))
    x1, y1, yaw = env._p
    assert yaw > 0  # left turn counterclockwise


def test_success_check():
    env = MiniNavEnv(img_size=64)
    spec = _spec()
    env.reset(spec)
    # teleport agent onto goal (test hook, not a policy behavior)
    env._p = (spec.goal_xy[0], spec.goal_xy[1], 0.0)
    assert env.is_success()


def test_benchmark_splits_disjoint_and_frozen():
    specs_val = list_episodes(["A_in_dist"], split="val")
    specs_test = list_episodes(["A_in_dist"], split="test")
    val_seeds = {s.seed for s in specs_val}
    test_seeds = {s.seed for s in specs_test}
    assert len(val_seeds & test_seeds) == 0, "val and test seeds must be disjoint"
    assert len(specs_val) == 12 and len(specs_test) == 32
    # same configuration is reproducible across calls
    again = make_episode("A_in_dist", "val", 0)
    assert again.seed == specs_val[0].seed
    assert again.instruction == specs_val[0].instruction
    assert again.goal_xy == specs_val[0].goal_xy


def test_conditions_exist_and_have_unique_ids():
    conds = ["A_in_dist", "B_env_shift", "C_instr_shift", "D_recovery"]
    for split in ("val", "test"):
        eps = list_episodes(conds, split)
        ids = [e.episode_id for e in eps]
        assert len(ids) == len(set(ids))
        assert len(eps) == sum(
            {"A_in_dist": 12, "B_env_shift": 12, "C_instr_shift": 12,
             "D_recovery": 12}.values()) if split == "val" else True
    # recovery condition starts facing away from goal
    rec = make_episode("D_recovery", "test", 0)
    d = np.hypot(rec.goal_xy[0] - rec.start_xy[0], rec.goal_xy[1] - rec.start_xy[1])
    ang = np.arctan2(rec.goal_xy[1] - rec.start_xy[1], rec.goal_xy[0] - rec.start_xy[0])
    assert np.abs((rec.start_yaw_rad - (ang + np.pi))) < 1e-6, "must start facing away"
