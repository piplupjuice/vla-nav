"""Research policy interfaces + a clearly-labeled MOCK policy for tests only.

The real model integration lives in `model/omnivla_adapter.py` (Kaggle).
Everything here defines the *contract* a navigation policy must satisfy so
that the frozen baseline and the replanning variants are guaranteed fair.

`MockNavPolicy` exists ONLY to unit-test the pipeline on MiniNav without a
GPU. It uses the privileged goal position and the running environment's
static collision geometry — that is the point of a mock. Its outputs are
tagged with `is_model=False` in every log line, and no result produced by
it may be cited as evidence about a VLA.

Semantics (mirroring the official OmniVLA contract):
    - waypoint k of the chunk = robot-frame position after (k+1) executed
      sub-steps, in units of 0.1 m (cumulative, NOT per-step increments);
    - the official PD controller decodes the selected waypoint into
      (linear_v, angular_v).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..environment.types import Cmd, Observation, WaypointChunk
from ..environment.types import NUM_ACTIONS_CHUNK, WAYPOINT_SPACING_M, OFFICIAL_WAYPOINT_SELECT
from ..environment.mininav_env import AGENT_RADIUS_M


class NavPolicy(ABC):
    """Contract for any mobile-navigation policy used by the runner."""

    is_model: bool = False

    @abstractmethod
    def infer(self, obs: Observation, instruction: str) -> WaypointChunk:
        """Produce a full 8-waypoint chunk from the current observation.

        Latency measurement is handled by the runner, not the policy.
        """

    def infer_step(self, obs: Observation, instruction: str,
                   waypoint_select: int = OFFICIAL_WAYPOINT_SELECT) -> Cmd:
        """Decode the command for a fresh observation."""
        return self.infer(obs, instruction).decode(waypoint_select=waypoint_select)

    def bind_env(self, env) -> None:  # noqa: B027
        """Optional hook: give the policy access to the *actually running* env.

        Called by `BaseRegime.run_episode` right after `env.reset(spec)`.
        Real VLA policies must leave this as a no-op (they never touch
        privileged state). The test-only mock overrides it so its clearance
        checks are measured against the same scene the trajectory runs in
        — a policy bound to a stale env instance would be silently unfair.
        """


@dataclass
class MockNavPolicy(NavPolicy):
    """Tangent-bug mock policy (TEST-ONLY; privileged goal + scene geometry).

    Strategy:
        - pick the candidate aim direction whose **inflated-footprint
          straight path** (radius + margin, so grazing paths are rejected)
          stays collision-free the longest, preferring directions near the
          goal bearing (front-cone tangent arc when the goal is behind);
        - final approach (goal within 0.8 m) aims straight at the goal;
        - if the odometry shows ~no motion for `_STUCK_ESC_AT` consecutive
          forward attempts, escalate a persistent (sign-fixed-per-episode)
          sidestep: first a wide sidestep, then further, then an about-face;
          escalation clears as soon as real motion resumes.

    Exercises the same `NavPolicy` interface a real VLA must satisfy.
    """

    goal_xy: Optional[list] = None
    clearance_env: object = None
    stall_pos_tol_m: float = 0.03
    #: when set, the mock "breaks" after N inference calls (emulates a policy
    #: whose predictions stop producing usable motion — used to test triggers)
    fail_from_call: Optional[int] = None

    _AIMS = np.concatenate([np.linspace(-1.7, -0.15, 12),
                            np.array([0.0]),
                            np.linspace(0.15, 1.7, 12)])
    _LOOKAHEAD_M = 2.0      # straight-path collision-check horizon
    _STUCK_ESC_AT = 3       # consecutive stalled forward attempts → escalate
    _ESC_ANGLES = (0.6 * math.pi, 0.8 * math.pi, math.pi)  # L1 sidestep … L3 about-face

    def __post_init__(self) -> None:
        self._last_xy: Optional[np.ndarray] = None
        self._stuck = 0
        self._stuck_sign: Optional[float] = None  # fixed on first stall
        self._prev_cmd_v = 0.0
        self._calls = 0
        self._esc = 0  # escalation level 1..3 (0 = not escalating)

    def bind_env(self, env) -> None:
        """Track the running env so clearance checks use its live scene."""
        self.clearance_env = env

    # ------------------------------------------------------------------
    def _path_hit_m(self, x: float, y: float, yaw: float, aim: float,
                    length: float) -> float:
        """Distance along the straight world-frame path before the inflated
        robot footprint (radius + margin) would collide. Privileged,
        TEST-ONLY: uses the running env's static collision geometry.
        Returns `length` when the whole path is free."""
        if self.clearance_env is None:
            return length
        wa = yaw + aim
        dx, dy = math.cos(wa), math.sin(wa)
        obstacles = getattr(self.clearance_env, "_scene", {}).get("obstacles", [])
        hits = self.clearance_env._circle_hits_any
        margin = 0.05
        for s in np.arange(0.05, length + 1e-9, 0.05):
            if hits(x + dx * float(s), y + dy * float(s),
                    AGENT_RADIUS_M + margin, obstacles):
                return float(s)
        return length

    def _choose_free_aim(self, x: float, y: float, yaw: float,
                         bearing: float) -> tuple[Optional[float], float]:
        """Best candidate aim; returns (aim or None, free-path meters).

        Scoring: among collision-free candidates prefer one near the desired
        direction, with a mild preference for a wider free path. Candidates
        whose inflated footprint collides are excluded entirely. Returns
        None when every candidate collides (caller must escalate).
        """
        half = math.pi / 2
        if abs(bearing) <= half:
            prefer = bearing
        else:  # out-of-cone goal: approach by front-cone tangent arc
            prefer = math.copysign(half, bearing)
        best: Optional[tuple[float, float]] = None
        for aim in self._AIMS:
            hit = self._path_hit_m(x, y, yaw, aim, self._LOOKAHEAD_M)
            if hit < self._LOOKAHEAD_M:
                continue  # candidate collides within the horizon: rejected
            score = 10.0 * abs(aim - prefer) + 3.0 * (1.0 - hit / self._LOOKAHEAD_M)
            if best is None or score < best[0]:
                best = (score, float(aim))
        return (None, 0.0) if best is None else (best[1], self._LOOKAHEAD_M)

    # ------------------------------------------------------------------
    def infer(self, obs: Observation, instruction: str) -> WaypointChunk:
        assert self.goal_xy is not None, "mock requires the privileged goal_xy"
        goal = np.asarray(self.goal_xy, dtype=np.float64)
        px, py, yaw = obs.robot_pose
        delta = goal - np.array([px, py])
        dist = float(np.linalg.norm(delta[:2]))
        self._calls += 1

        if (self.fail_from_call is not None
                and self._calls > self.fail_from_call):
            dist = 0.0  # emitted motion collapses to zero after N calls

        bearing = (math.atan2(delta[1], delta[0]) - yaw + math.pi) % (2 * math.pi) - math.pi

        # ---- stuck counter (odometry + previous forward attempt only) ----
        if self._last_xy is not None:
            moved = float(np.linalg.norm(np.array([px, py]) - self._last_xy))
            if moved < self.stall_pos_tol_m and self._prev_cmd_v > 0.05:
                self._stuck += 1
                if self._stuck_sign is None:
                    self._stuck_sign = 1.0  # fixed for the whole episode
            elif moved >= self.stall_pos_tol_m:
                self._stuck = 0
                self._esc = 0  # real motion resumed: clear escalation
        self._last_xy = np.array([px, py])

        # ---- escalation: each run of `_STUCK_ESC_AT` stalls advances one level
        if self._stuck >= self._STUCK_ESC_AT and self._esc < len(self._ESC_ANGLES):
            self._esc += 1
            self._stuck = 0

        if self._esc > 0:
            # Persistent escape maneuver: aim far off the blocked direction.
            aim = bearing + self._stuck_sign * self._ESC_ANGLES[self._esc - 1]
            aim = (aim + math.pi) % (2 * math.pi) - math.pi
            hit = self._path_hit_m(px, py, yaw, aim, self._LOOKAHEAD_M)
            if hit < self._LOOKAHEAD_M:
                # escalation aim also blocked: try the mirrored sidestep
                alt = (bearing - self._stuck_sign * self._ESC_ANGLES[self._esc - 1]
                       + math.pi) % (2 * math.pi) - math.pi
                if self._path_hit_m(px, py, yaw, alt, self._LOOKAHEAD_M) >= self._LOOKAHEAD_M:
                    aim = alt
            reach = min(dist, 0.5)  # short escape; re-evaluate soon
        elif dist <= 0.8:
            aim = bearing  # final approach: aim exactly at the goal
            hit = self._path_hit_m(px, py, yaw, aim, 0.8)
            reach = min(dist, max(hit, 0.1))
        else:
            aim, hit = self._choose_free_aim(px, py, yaw, bearing)
            if aim is None:
                # every candidate collides: begin escalation immediately
                self._esc = 1
                self._stuck = 0
                if self._stuck_sign is None:
                    self._stuck_sign = 1.0
                aim = bearing + self._stuck_sign * self._ESC_ANGLES[0]
                aim = (aim + math.pi) % (2 * math.pi) - math.pi
                if self._path_hit_m(px, py, yaw, aim, self._LOOKAHEAD_M) < self._LOOKAHEAD_M:
                    aim = ((bearing + self._stuck_sign * math.pi / 2
                            + math.pi) % (2 * math.pi) - math.pi)
                reach = min(dist, 0.5)
            else:
                reach = min(dist, max(hit, 0.2))

        # planned reach: full chunk covers up to 0.8 m; trim to goal distance
        reach = min(reach, NUM_ACTIONS_CHUNK * WAYPOINT_SPACING_M)
        chunk = np.zeros((NUM_ACTIONS_CHUNK, 4), dtype=np.float64)
        step_norm = reach / NUM_ACTIONS_CHUNK / WAYPOINT_SPACING_M
        for k in range(NUM_ACTIONS_CHUNK):
            frac = (k + 1) * step_norm * WAYPOINT_SPACING_M
            chunk[k, 0] = frac * math.cos(aim) / WAYPOINT_SPACING_M
            chunk[k, 1] = frac * math.sin(aim) / WAYPOINT_SPACING_M
            chunk[k, 2] = math.cos(aim)
            chunk[k, 3] = math.sin(aim)
        self._prev_cmd_v = WaypointChunk(chunk).decode(4).linear_v
        return WaypointChunk(chunk)
