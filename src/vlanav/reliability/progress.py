"""PID-light progress tracking and the reliability signal (pre-threshold).

Main reliability signal (pre-registered in configs/reliability.yaml):
    d_t = || intended displacement of the chosen waypoint ||_2
          - observed displacement (robot odometry)

Both quantities are onboard quantities a deployed robot can measure; no
privileged simulator state is used. A positive d_t is a *progress bug*.
We also track a stale-odometry ledger: consecutive steps with ~zero motion
(`stall`). Threshold selection belongs to the caller; this module just
produces the raw signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def intended_displacement_m(chosen_waypoint_norm: np.ndarray,
                            waypoint_spacing_m: float = 0.1) -> float:
    """Euclidean magnitude of the selected waypoint's (dx, dy) in meters."""
    dx, dy = float(chosen_waypoint_norm[0]), float(chosen_waypoint_norm[1])
    return math.hypot(dx, dy) * waypoint_spacing_m


@dataclass
class ProgressTracker:
    """Accumulates stepwise progress signals for one episode."""

    stall_threshold_m: float = 0.01  # below this, a step counts as stalled
    stall_consecutive_limit: int = 2
    _stall_run: int = 0
    _d_history: list = field(default_factory=list)

    def reset(self) -> None:
        self._stall_run = 0
        self._d_history = []

    def update(self, intended_m: float, observed_m: float) -> float:
        """Register one executed step; returns the discrepancy d_t (meters)."""
        d = float(intended_m) - float(observed_m)
        self._d_history.append(d)
        if observed_m < self.stall_threshold_m:
            self._stall_run += 1
        else:
            self._stall_run = 0
        return d

    @property
    def stall_run(self) -> int:
        return self._stall_run

    @property
    def history(self) -> list:
        return list(self._d_history)


@dataclass
class ReliabilitySignal:
    """The decision statistic produced each step."""

    d_t: float          # meters, positive => worse than predicted
    stall_run: int      # consecutive stalled steps
    trigger: bool       # raw decision (thresholds resolved by caller)
