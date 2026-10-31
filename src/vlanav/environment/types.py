"""Core data types and the shared PD controller for VLA-NAV.

The PD controller here is a faithful re-implementation of the controller in
OmniVLA's official inference script (`inference/run_omnivla.py` in
github.com/NHirose/OmniVLA, accessed 2026-10-28). Any model adapter must use
*this single implementation* so that the frozen baseline and all replanning
variants use identical actuation.

Conventions
-----------
Waypoint chunk: array of shape (NUM_ACTIONS_CHUNK=8, 4):
    (dx, dy, cos(heading), sin(heading)) in units of `metric_waypoint_spacing`
    meters. Robot frame: x = front, y = left.
Cmd: (linear_vel, angular_vel) with v >= 0 (no reverse; controller turns
     in place instead) and angular_vel clipped to [-1.0, 1.0] rad/s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

#: Official OmniVLA constants (see its inference script).
NUM_ACTIONS_CHUNK = 8
WAYPOINT_SPACING_M = 0.1
OFFICIAL_WAYPOINT_SELECT = 4  # waypoint index used by official controller
PER_STEP_WAYPOINT_SELECT = 0  # waypoint used when replanning every step
PD_DT = 1 / 3  # control period of the official PD controller

#: Velocity limits of the official controller.
MAX_LINEAR_V = 0.5
MAX_ANGULAR_V = 1.0

#: The hard sub-limits of the official *deployment* script's velocity
#: limitation block (`run_omnivla.py`, "Velocity limitation": maxv, maxw
#: = 0.3). Replicated verbatim; the curvature rules below can reduce speeds
#: FURTHER than these clips, so reliability `d_t` uses cmd.linear_v * PD_DT.
DEPLOY_MAX_LINEAR_V = 0.3
DEPLOY_MAX_ANGULAR_V = 0.3


@dataclass
class Cmd:
    """A single differential-drive velocity command."""

    linear_v: float   # m/s, >= 0
    angular_v: float  # rad/s

    def as_tuple(self) -> Tuple[float, float]:
        return (float(self.linear_v), float(self.angular_v))


@dataclass
class WaypointChunk:
    """A model-produced chunk of 8 waypoints in the normalized 0.1 m space."""

    data: np.ndarray  # shape (8, 4) float

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.shape != (NUM_ACTIONS_CHUNK, 4):
            raise ValueError(
                f"WaypointChunk must be {(NUM_ACTIONS_CHUNK, 4)}, got {self.data.shape}"
            )

    # ------------------------------------------------------------------
    # Decoding to a differential-drive command via the official controller
    # ------------------------------------------------------------------
    def decode(self, waypoint_select: int = OFFICIAL_WAYPOINT_SELECT) -> Cmd:
        """Decode with the official OmniVLA PD controller (fresh observation)."""
        chosen = self.data[waypoint_select].copy() * WAYPOINT_SPACING_M
        dx, dy, hx, hy = (float(c) for c in chosen)
        return _pd_to_cmd(dx, dy, hx, hy)

    # ------------------------------------------------------------------
    # Plan following: execute a held chunk with ODOMETRY ONLY re-aiming.
    # (No re-observation of the world: the plan positions were fixed at
    # query time; only the robot's own pose is used to transform them.)
    # ------------------------------------------------------------------
    def follow(self, pose_at_query: np.ndarray, pose_now: np.ndarray,
               waypoint_select: int = OFFICIAL_WAYPOINT_SELECT) -> Cmd:
        """Residual-form decode of the held plan in the current robot frame."""
        chosen = self.data[waypoint_select].copy() * WAYPOINT_SPACING_M
        dx_q, dy_q = float(chosen[0]), float(chosen[1])
        yq = float(pose_at_query[2])
        # plan point in the world frame
        wx = float(pose_at_query[0]) + dx_q * math.cos(yq) - dy_q * math.sin(yq)
        wy = float(pose_at_query[1]) + dx_q * math.sin(yq) + dy_q * math.cos(yq)
        # residual in the current robot frame
        ex, ey = wx - float(pose_now[0]), wy - float(pose_now[1])
        ynow = float(pose_now[2])
        dx = ex * math.cos(ynow) + ey * math.sin(ynow)
        dy = -ex * math.sin(ynow) + ey * math.cos(ynow)
        hx, hy = float(chosen[2]), float(chosen[3])
        return _pd_to_cmd(dx, dy, hx, hy)


def _pd_to_cmd(dx: float, dy: float, hx: float, hy: float) -> Cmd:
    """The official PD controller body (shared by decode() and follow()).

    Replicates `run_omnivla.py` verbatim: PD decode → clip to (0, 0.5) m/s
    and ±1.0 rad/s → curvature-based velocity limitation (maxv=maxw=0.3).
    NOTE: the official steering uses plain `arctan(dy/dx)` (not atan2), so a
    negative-dx waypoint steers by the *acute* angle — faithful even when it
    looks wrong, because the deployment behaves that way.
    """
    dt = PD_DT
    eps = 1e-8
    if abs(dx) < eps and abs(dy) < eps:
        linear = 0.0
        angular = 1.0 * clip_angle(math.atan2(hy, hx)) / dt
    elif abs(dx) < eps:
        linear = 0.0
        angular = 1.0 * np.sign(dy) * np.pi / (2.0 * dt)
    else:
        linear = dx / dt
        angular = math.atan(dy / dx) / dt
    linear = float(np.clip(linear, 0.0, MAX_LINEAR_V))
    angular = float(np.clip(angular, -MAX_ANGULAR_V, MAX_ANGULAR_V))

    # ---- Velocity limitation (official block, threshold values 0.3/0.3) ----
    maxv, maxw = DEPLOY_MAX_LINEAR_V, DEPLOY_MAX_ANGULAR_V
    if abs(linear) <= maxv:
        if abs(angular) <= maxw:
            linear_l = linear
            angular_l = angular
        else:
            rd = linear / angular
            linear_l = maxw * np.sign(linear) * abs(rd)
            angular_l = maxw * np.sign(angular)
    else:
        if abs(angular) <= 0.001:
            linear_l = maxv * np.sign(linear)
            angular_l = 0.0
        else:
            rd = linear / angular
            if abs(rd) >= maxv / maxw:
                linear_l = maxv * np.sign(linear)
                angular_l = maxv * np.sign(angular) / abs(rd)
            else:
                linear_l = maxw * np.sign(linear) * abs(rd)
                angular_l = maxw * np.sign(angular)
    return Cmd(linear_v=float(linear_l), angular_v=float(angular_l))


def clip_angle(theta: float) -> float:
    """Clip an angle to [-pi, pi]."""
    return (theta + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Observation:
    """What the robot can actually perceive at a timestep.

    `image` is the egocentric camera frame. `robot_pose` is the robot's own
    odometry (available on real deployments), NOT privileged simulator state.
    Goal information deliberately lives only in the EpisodeSpec; whether the
    policy sees any goal signal is decided by the policy adapter.
    """

    image: np.ndarray          # (H, W, 3) uint8
    robot_pose: np.ndarray     # (x, y, yaw) odometry frame [= world when clean]
    step: int

    def copy(self) -> "Observation":
        return Observation(self.image.copy(), self.robot_pose.copy(), self.step)


@dataclass
class EpisodeSpec:
    """A fully specified navigation episode configuration.

    `goal_world_xy` is evaluation bookkeeping; it is handed to the simulator
    (which places the beacon), but given to the policy only through whatever
    goal modality the adapter elects (e.g., language instruction, goal image).
    """

    episode_id: str
    condition: str            # A_in_dist | B_env_shift | C_instr_shift | D_recovery
    split: str                # val | test
    seed: int
    world_id: int             # seed for scene generation
    start_xy: Tuple[float, float]
    start_yaw_rad: float
    goal_xy: Tuple[float, float]
    instruction: str
    max_steps: int = 200
    goal_radius_m: float = 0.6
    extra: dict = field(default_factory=dict)


@dataclass
class StepInfo:
    """Per-step simulator bookkeeping (non-privileged; logged anyway)."""

    collision: bool
    moved_distance_m: float
    step: int
