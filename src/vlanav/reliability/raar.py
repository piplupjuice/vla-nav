"""RAAR: Reliability-Aware Adaptive Replanning (P1) + matched baselines.

Next-cycle methods (research plan 2026-10-30, docs/raar_research_plan.md).
Everything here plugs into the same BaseRegime loop as B1/B2/B3, so any
comparison shares policy, env, budget, termination and logging.

Signals (P1 main configuration is ONBOARD-ONLY, deployable on a real robot):
  S1 motion execution error  e_t = |intended_disp_t - observed_disp_t|,
     EMA-smoothed and normalized by `motion_scale` (calibrated on
     validation episodes, frozen before test);
  S2 stall indicator  s_t = 1[observed motion < stall_eps_m for
     `stall_limit` consecutive ticks].

OPTIONAL (ablation ONLY — uses simulator ground-truth goal distance,
which a real robot does not have; never part of the deployable P1):
  S3 goal-progress indicator  q_t = 1[goal-distance reduction over the
     last `progress_window` ticks < progress_min_m], supplied through a
     `goal_distance_fn` hook; the regime raises if w_p > 0 without one.

Score:  R_t = w_m * S1_t + w_s * S2_t (+ w_p * S3_t)
Trigger:  R_t > tau  =>  abort the held chunk and re-query.

Pre-registered safety rules (research plan §2.2):
  cooldown        — no TRIGGERED replan within `cooldown` ticks of the
                    previous one (prevents per-tick re-query on one failure);
  chunk horizon   — a held plan never executes beyond `max_chunk_ticks`;
  episode cap     — at most `max_replans_per_episode` triggered replans.

Matched baselines (research plan §Phase C) — the fair-comparison controls:
  B4 RandomBudgetReplan  — chunked execution + per-tick early re-query with
        probability p; p is calibrated ON VALIDATION so the mean
        queries/episode matches RAAR's, then frozen for test. Answers
        "is any improvement just from querying more?".
  B5 FixedBudgetReplan   — fixed query interval k (validation-calibrated to
        the same budget). Answers "is adaptivity better than a schedule?".
  O1 CollisionOracleTriggered — diagnostic only: re-queries whenever the
        simulator reports a collision since the last query. Uses privileged
        simulator events; it bounds what a *reactive* trigger could achieve
        and is never presented as a deployable method.

Budget calibration (`calibrate_budget_match`) is validation-only arithmetic;
its outputs (p, k) are recorded in meta.extra for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np

from ..environment.types import Cmd, EpisodeSpec, OFFICIAL_WAYPOINT_SELECT
from ..model.mock_policy import NavPolicy
from .replanning import BaseRegime, ChunkedFrozen, _residual_magn


@dataclass
class RAARConfig:
    """Weights, threshold and safety rules. Defaults = plan §2.1/§2.2."""

    w_m: float = 1.0          # weight, normalized motion execution error
    w_s: float = 1.0          # weight, stall indicator (0/1)
    w_p: float = 0.0          # weight, goal-progress indicator (ablation ONLY)
    tau: float = 1.0          # replan when R_t > tau
    cooldown: int = 3         # ticks before the next TRIGGERED replan
    stall_limit: int = 2      # consecutive stalled ticks that set S2
    stall_eps_m: float = 0.01
    motion_scale: float = 0.1  # normalizer; calibrate on validation
    motion_ema: float = 0.5    # EMA factor of the raw motion error
    progress_window: int = 5   # ticks for S3 (ablation only)
    progress_min_m: float = 0.05
    max_chunk_ticks: int = 8
    max_replans_per_episode: int = 24

    def sanitized(self) -> "RAARConfig":
        if self.w_p > 0 and self.w_s < 0:
            raise ValueError("negative weights")
        return self


class ReliabilityMonitor:
    """Computes the RAAR score from executed-step feedback only.

    Feed one `update(...)` per executed tick, in execution order. The score
    returned by `update` reflects ONLY information available at that tick
    (the step that just finished) — never the future — so a trigger built on
    it cannot peek.
    """

    def __init__(self, cfg: RAARConfig) -> None:
        self.cfg = cfg.sanitized()
        self._ema_err = 0.0
        self._stall_run = 0
        self._goal_hist: deque = deque(maxlen=max(2, cfg.progress_window))
        self.last_score = 0.0
        self.last_e_raw = 0.0

    def reset(self) -> None:
        self._ema_err = 0.0
        self._stall_run = 0
        self._goal_hist.clear()
        self.last_score = 0.0
        self.last_e_raw = 0.0

    def update(self, intended_m: float, observed_m: float,
               goal_dist: float | None = None) -> float:
        cfg = self.cfg
        e_raw = abs(float(intended_m) - float(observed_m))
        self._ema_err = (cfg.motion_ema * e_raw
                         + (1.0 - cfg.motion_ema) * self._ema_err)
        self.last_e_raw = e_raw
        s1 = min(1.0, self._ema_err / max(cfg.motion_scale, 1e-9))

        if observed_m < cfg.stall_eps_m:
            self._stall_run += 1
        else:
            self._stall_run = 0
        s2 = 1.0 if self._stall_run >= cfg.stall_limit else 0.0

        s3 = 0.0
        if cfg.w_p > 0.0:
            # ablation-only signal: needs privileged goal distances
            self._goal_hist.append(float(goal_dist) if goal_dist is not None
                                   else np.nan)
            if len(self._goal_hist) == self._goal_hist.maxlen:
                drop = self._goal_hist[0] - self._goal_hist[-1]
                s3 = 1.0 if not (drop >= cfg.progress_min_m) else 0.0

        self.last_score = cfg.w_m * s1 + cfg.w_s * s2 + cfg.w_p * s3
        return self.last_score


class RAARTriggered(BaseRegime):
    """P1: chunked plan-following with multi-signal adaptive re-query."""

    def __init__(self, policy: NavPolicy, name: str = "raar_triggered", *,
                 config: RAARConfig | None = None,
                 goal_distance_fn=None, **kw) -> None:
        super().__init__(policy, name, **kw)
        self.cfg = (config or RAARConfig()).sanitized()
        if self.cfg.w_p > 0 and goal_distance_fn is None:
            raise ValueError(
                "w_p > 0 (goal-progress signal) requires goal_distance_fn; "
                "that signal is simulator-privileged and ablation-only")
        self.goal_distance_fn = goal_distance_fn
        self.chunk_ticks = self.cfg.max_chunk_ticks
        self.plan_residual_eps_m = ChunkedFrozen.plan_residual_eps_m

    def reset_episode(self, spec: EpisodeSpec) -> None:
        self._chunk = None
        self._ticks_since_query = 0
        self._ticks_since_replan = 10 ** 6
        self._pose_at_query = None
        self._replan_count = 0
        self._monitor = ReliabilityMonitor(self.cfg)
        self._prev_goal_dist: float | None = None

    def on_step(self, intended_m: float, observed_m: float) -> None:  # noqa: D102
        goal_dist = None
        if self.goal_distance_fn is not None:
            goal_dist = float(self.goal_distance_fn())
        self._monitor.update(intended_m, observed_m, goal_dist)

    def _triggered(self) -> bool:
        cfg = self.cfg
        if self._replan_count >= cfg.max_replans_per_episode:
            return False
        if self._ticks_since_replan < cfg.cooldown:
            return False
        return self._monitor.last_score > cfg.tau

    def query_and_decode(self, obs, spec) -> Cmd:
        plan_stale = (self._chunk is None
                      or self._ticks_since_query >= self.chunk_ticks)
        if not plan_stale and self._triggered():
            self._replan_count += 1
            self._ticks_since_replan = 0
            self._chunk = None
            self._ticks_since_query = self.chunk_ticks
            plan_stale = True
        if not plan_stale:
            residual = _residual_magn(self._pose_at_query, obs.robot_pose,
                                      self._chunk)
            plan_stale = residual < self.plan_residual_eps_m
        if plan_stale:
            self._chunk = self._timed_query(obs, spec.instruction)
            self._ticks_since_query = 0
            self._pose_at_query = obs.robot_pose.copy()
        cmd = self._chunk.follow(self._pose_at_query, obs.robot_pose)
        self._ticks_since_query += 1
        self._ticks_since_replan += 1
        return cmd


class RandomBudgetReplan(BaseRegime):
    """B4 control: budget-matched RANDOM triggering.

    Executes chunks like B1 but may abort and re-query early with
    probability `p` per tick (fresh per-episode RNG derived from the episode
    seed, so runs replay bit-identically). `p` is calibrated on validation
    to match RAAR's mean queries/episode and then FROZEN — never tuned on
    test. If the trigger were informative only because it queries more,
    this control would match RAAR's success at the same budget.
    """

    def __init__(self, policy: NavPolicy, name: str = "random_budget", *,
                 fire_p: float, chunk_ticks: int = 8,
                 max_replans_per_episode: int = 24, **kw) -> None:
        super().__init__(policy, name, **kw)
        if not (0.0 <= fire_p <= 1.0):
            raise ValueError("fire_p must be in [0, 1]")
        self.fire_p = float(fire_p)
        self.chunk_ticks = int(chunk_ticks)
        self.max_replans = int(max_replans_per_episode)
        self.plan_residual_eps_m = ChunkedFrozen.plan_residual_eps_m

    def reset_episode(self, spec: EpisodeSpec) -> None:
        self._chunk = None
        self._ticks_since_query = 0
        self._pose_at_query = None
        self._replan_count = 0
        # method-tagged stream keeps episodes deterministic and independent
        self._rng = np.random.default_rng((int(spec.seed) * 1_000_003 + 7)
                                          % (2 ** 31))

    def query_and_decode(self, obs, spec) -> Cmd:
        plan_stale = (self._chunk is None
                      or self._ticks_since_query >= self.chunk_ticks)
        if (not plan_stale
                and self._replan_count < self.max_replans
                and self._rng.random() < self.fire_p):
            self._replan_count += 1
            self._chunk = None
            self._ticks_since_query = self.chunk_ticks
            plan_stale = True
        if not plan_stale:
            residual = _residual_magn(self._pose_at_query, obs.robot_pose,
                                      self._chunk)
            plan_stale = residual < self.plan_residual_eps_m
        if plan_stale:
            self._chunk = self._timed_query(obs, spec.instruction)
            self._ticks_since_query = 0
            self._pose_at_query = obs.robot_pose.copy()
        cmd = self._chunk.follow(self._pose_at_query, obs.robot_pose)
        self._ticks_since_query += 1
        return cmd


class FixedBudgetReplan(BaseRegime):
    """B5 control: FIXED query interval k, validation-calibrated to match
    RAAR's budget. Identical actuation to B1; only the interval differs."""

    def __init__(self, policy: NavPolicy, name: str = "fixed_budget", *,
                 interval_ticks: int, **kw) -> None:
        super().__init__(policy, name, **kw)
        if interval_ticks < 1:
            raise ValueError("interval_ticks must be >= 1")
        self.chunk_ticks = int(interval_ticks)
        self.plan_residual_eps_m = ChunkedFrozen.plan_residual_eps_m

    def reset_episode(self, spec: EpisodeSpec) -> None:
        self._chunk = None
        self._ticks_since_query = 0
        self._pose_at_query = None

    def query_and_decode(self, obs, spec) -> Cmd:
        plan_stale = (self._chunk is None
                      or self._ticks_since_query >= self.chunk_ticks)
        if not plan_stale:
            residual = _residual_magn(self._pose_at_query, obs.robot_pose,
                                      self._chunk)
            plan_stale = residual < self.plan_residual_eps_m
        if plan_stale:
            self._chunk = self._timed_query(obs, spec.instruction)
            self._ticks_since_query = 0
            self._pose_at_query = obs.robot_pose.copy()
        cmd = self._chunk.follow(self._pose_at_query, obs.robot_pose)
        self._ticks_since_query += 1
        return cmd


class CollisionOracleTriggered(BaseRegime):
    """O1 DIAGNOSTIC (never a deployable baseline): re-queries whenever the
    simulator reported a collision on the previous executed step.

    Upper-bounds what a *reactive* trigger could do by using privileged
    simulator collision events. Research plan §Phase C: present as an
    oracle diagnostic only.
    """

    chunk_ticks: int = 8

    def __init__(self, policy: NavPolicy, name: str = "oracle_collision",
                 **kw) -> None:
        super().__init__(policy, name, **kw)
        self.plan_residual_eps_m = ChunkedFrozen.plan_residual_eps_m

    def reset_episode(self, spec: EpisodeSpec) -> None:
        self._chunk = None
        self._ticks_since_query = 0
        self._pose_at_query = None
        self._collided = False
        self._replan_count = 0

    def mark_collision(self) -> None:
        self._collided = True

    def query_and_decode(self, obs, spec) -> Cmd:
        plan_stale = (self._chunk is None
                      or self._ticks_since_query >= self.chunk_ticks)
        if not plan_stale and self._collided:
            self._collided = False
            self._replan_count += 1
            self._chunk = None
            self._ticks_since_query = self.chunk_ticks
            plan_stale = True
        if not plan_stale:
            residual = _residual_magn(self._pose_at_query, obs.robot_pose,
                                      self._chunk)
            plan_stale = residual < self.plan_residual_eps_m
        if plan_stale:
            self._chunk = self._timed_query(obs, spec.instruction)
            self._ticks_since_query = 0
            self._pose_at_query = obs.robot_pose.copy()
        cmd = self._chunk.follow(self._pose_at_query, obs.robot_pose)
        self._ticks_since_query += 1
        return cmd


def calibrate_budget_match(target_queries_per_ep: float, mean_steps: float,
                           chunk_ticks: int = 8) -> dict:
    """Validation-only arithmetic that matches B4/B5 to a target budget.

    B5: queries/ep ≈ mean_steps / k  =>  k = clip(round(mean_steps/Q), 1, ..)
    B4: per tick not already fresh, fire with prob p. Expected extra fires
        per episode ≈ p * (mean_steps - Q_base) where Q_base is the chunk
        cadence alone; solving p so total ≈ Q:
            p = (Q - Q_base) / max(mean_steps - Q_base, 1), clipped to [0,1].
    Both outputs are FROZEN into meta.extra before any test episode runs.
    """
    q = float(target_queries_per_ep)
    t = float(mean_steps)
    k = int(np.clip(round(t / max(q, 1e-9)), 1, 64))
    q_base = t / chunk_ticks
    p = float(np.clip((q - q_base) / max(t - q_base, 1.0), 0.0, 1.0))
    return {"fixed_interval_ticks": k, "random_fire_p": round(p, 4),
            "target_queries_per_ep": round(q, 3),
            "mean_steps_used": round(t, 2),
            "source": "calibrated on validation only; frozen before test"}
