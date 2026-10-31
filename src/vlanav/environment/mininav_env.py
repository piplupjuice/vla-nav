"""MiniNav: a deterministic lightweight 2D mobile-navigation environment.

STATUS AND ROLE (read before using any VLA on it):
    MiniNav is a *pipeline-only* fallback simulator. Its rendering is a
    synthetic depth-shaded raycast view, which is far outside the
    distribution of photorealistic data any pretrained VLA was trained on.
    It exists so that the reliability, evaluation, logging and demo machinery
    can be developed and unit-tested without GPU access.

    Results of a VLA on MiniNav must never be presented as evidence that the
    VLA "works"; they are labeled as `sim_family=mininav` in every log line.
    The renderer draws an explicit goal beacon (a colored striped column),
    which makes it a *visually-marked-goal* task; the real VLA integration
    target (AI2-THOR / ProcTHOR houses) is language-goal, not beacon-goal.

Determinism: given the same EpisodeSpec.seed, scene layout, start/goal and
instruction are bit-identical across machines (numpy default_rng).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .types import Cmd, EpisodeSpec, Observation, StepInfo
from .types import MAX_LINEAR_V, MAX_ANGULAR_V, PD_DT

AGENT_RADIUS_M = 0.25
FOV_DEG = 90.0
WORLD_SIZE_M = 12.0
CAMERA_HEIGHT = 1.4  # only affects rendering aesthetics


class MiniNavEnv:
    """Continuous 2D differential-drive navigation with a rendered beacon goal."""

    sim_family = "mininav"
    sim_semantics = "differential-drive; goal beacon marked in RGB"

    def __init__(self, img_size: int = 256, fov_deg: float = FOV_DEG) -> None:
        self.img_size = img_size
        self.fov_deg = fov_deg
        self.spec: EpisodeSpec | None = None
        self._rng: np.random.Generator | None = None
        self._poses: List[Tuple[float, float, float]] = []  # trajectory log
        self._step_count = 0
        self._collision_total = 0
        self._path_length = 0.0

    # ------------------------------------------------------------------
    # Scene generation (deterministic from seed)
    # ------------------------------------------------------------------
    def _make_scene(self, seed: int, require_clear_path: bool = True) -> dict:
        rng = np.random.default_rng(seed)
        # A/C/D scenes reject layouts whose straight start→goal segment is
        # blocked (feasible standard scenes) — in MiniNav the straight path
        # being clear is a property of the *scene*, sampled with rejection;
        # B_env_shift drops that requirement, which is exactly the "increase
        # clutter / obstacle-shift" condition (documented in episodes.py).
        n_range = (4, 9) if not require_clear_path else (3, 7)
        for attempt in range(200):
            n_obs = int(rng.integers(*n_range))
            obstacles: List[Tuple[float, float, float, float]] = []
            margin = 1.0
            for _ in range(n_obs):
                cx = float(rng.uniform(-WORLD_SIZE_M / 2 + margin, WORLD_SIZE_M / 2 - margin))
                cy = float(rng.uniform(-WORLD_SIZE_M / 2 + margin, WORLD_SIZE_M / 2 - margin))
                w = float(rng.uniform(0.8, 2.2))
                h = float(rng.uniform(0.8, 2.2))
                obstacles.append((cx, cy, w, h))

            def free_point(min_clearance: float) -> Tuple[float, float]:
                for _ in range(200):
                    x = float(rng.uniform(-WORLD_SIZE_M / 2 + 0.6, WORLD_SIZE_M / 2 - 0.6))
                    y = float(rng.uniform(-WORLD_SIZE_M / 2 + 0.6, WORLD_SIZE_M / 2 - 0.6))
                    if all(
                        abs(x - cx) > w / 2 + min_clearance
                        and abs(y - cy) > h / 2 + min_clearance
                        for cx, cy, w, h in obstacles
                    ):
                        return (x, y)
                return (0.0, 0.0)

            start = free_point(0.8)
            goal = free_point(0.6)
            tries = 0
            while math.hypot(start[0] - goal[0], start[1] - goal[1]) < 4.0 and tries < 50:
                goal = free_point(0.6)
                tries += 1
                if tries == 50:
                    start = free_point(0.8)
            if not require_clear_path:
                break
            # straight-segment clearance (inflated by the agent radius)
            def seg_hits(px, py, qx, qy, cx, cy, w, h) -> bool:
                seg = np.array([qx - px, qy - py])
                rel = np.array([cx - px, cy - py])
                t0 = max(0.0, min(1.0, float(np.dot(seg, rel) / max(seg @ seg, 1e-9))))
                closest = np.array([px, py]) + t0 * seg
                dx = abs(closest[0] - cx) - w / 2
                dy = abs(closest[1] - cy) - h / 2
                dd = math.hypot(max(dx, 0.0), max(dy, 0.0)) - (AGENT_RADIUS_M + 0.2)
                return dd < 0
            clear = all(
                not seg_hits(start[0], start[1], goal[0], goal[1], *o)
                for o in obstacles
            )
            if clear:
                break
        obstacles.sort(key=lambda o: (o[0], o[1]))
        return {"obstacles": obstacles, "start": start, "goal": goal, "seed": seed}

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _circle_hits_any(x: float, y: float, r: float, obstacles) -> bool:
        for cx, cy, w, h in obstacles:
            dx = max(cx - w / 2, min(x, cx + w / 2)) - x
            dy = max(cy - h / 2, min(y, cy + h / 2)) - y
            if dx * dx + dy * dy < r * r:
                return True
        # walls
        lim = WORLD_SIZE_M / 2
        if abs(x) > lim - r or abs(y) > lim - r:
            return True
        return False

    def _raycast_depth(self, pos, yaw, rel_angle: float, max_depth: float) -> float:
        """Depth to nearest hit along a ray from (x, y, yaw+rel_angle)."""
        a = yaw + rel_angle
        x0, y0 = pos
        dx, dy = math.cos(a), math.sin(a)
        best = max_depth
        # walls (axis-aligned box)
        if dx > 0:
            best = min(best, max(0.0, (WORLD_SIZE_M / 2 - x0) / dx))
        elif dx < 0:
            best = min(best, max(0.0, (-WORLD_SIZE_M / 2 - x0) / dx))
        if dy > 0:
            best = min(best, max(0.0, (WORLD_SIZE_M / 2 - y0) / dy))
        elif dy < 0:
            best = min(best, max(0.0, (-WORLD_SIZE_M / 2 - y0) / dy))
        # obstacles (AABB ray clip)
        for cx, cy, w, h in self._scene["obstacles"]:
            t = _ray_aabb(x0, y0, dx, dy, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if t is not None:
                best = min(best, t)
        return float(min(best, max_depth))

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self, spec: EpisodeSpec) -> Observation:
        assert spec.split in ("val", "test")
        self.spec = spec
        require_clear = spec.extra.get("clutter", "standard") != "high"
        self._scene = self._make_scene(spec.world_id, require_clear_path=require_clear)
        rng = np.random.default_rng(spec.seed)
        # small deterministic start-yaw jitter (docs: scene is seed-full)
        self._p: Tuple[float, float, float] = (
            float(spec.start_xy[0]),
            float(spec.start_xy[1]),
            float(spec.start_yaw_rad),
        )
        self._rng = rng
        self._poses = [self._p]
        self._step_count = 0
        self._collision_total = 0
        self._path_length = 0.0
        return self.observe()

    def observe(self) -> Observation:
        x, y, yaw = self._p
        return Observation(
            image=self.render(),
            robot_pose=np.array([x, y, yaw]),
            step=self._step_count,
        )

    def step(self, cmd: Cmd) -> StepInfo:
        assert self.spec is not None, "call reset() first"
        v = float(np.clip(cmd.linear_v, 0.0, MAX_LINEAR_V))
        w = float(np.clip(cmd.angular_v, -MAX_ANGULAR_V, MAX_ANGULAR_V))
        dt = PD_DT
        x, y, yaw = self._p
        nx, ny, nyaw = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, (yaw + w * dt)
        nyaw = (nyaw + math.pi) % (2 * math.pi) - math.pi
        collision = self._circle_hits_any(nx, ny, AGENT_RADIUS_M, self._scene["obstacles"])
        if collision:
            nx, ny = x, y  # blocked: no motion, collision logged
        self._p = (nx, ny, nyaw)
        moved = 0.0 if collision else math.hypot(nx - x, ny - y)
        self._path_length += moved
        self._step_count += 1
        if collision:
            self._collision_total += 1
        self._poses.append(self._p)
        return StepInfo(collision=bool(collision), moved_distance_m=float(moved),
                        step=self._step_count)

    # ------------------------------------------------------------------
    # Rendering: first-person depth-shaded view + goal beacon stripe.
    # ------------------------------------------------------------------
    def render(self) -> np.ndarray:
        assert self.spec is not None
        W = self.img_size
        H = self.img_size // 2
        x, y, yaw = self._p
        fov = math.radians(self.fov_deg)
        angles = np.linspace(-fov / 2, fov / 2, W)
        max_depth = 15.0
        depths = np.array([self._raycast_depth((x, y), yaw, a, max_depth) for a in angles])
        # depth shading
        shade = 1.0 - (depths / max_depth)
        shade = np.clip(shade, 0.05, 1.0) ** 1.4
        img = np.tile((shade * 200).clip(0, 255).astype(np.uint8)[None, :, None], (H, 1, 3))

        # goal beacon: render if within FOV and line-of-sight
        gx, gy = self.spec.goal_xy
        dxg, dyg = gx - x, gy - y
        dist_g = math.hypot(dxg, dyg)
        bearing = math.atan2(dyg, dxg) - yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        if abs(bearing) < fov / 2 and dist_g < max_depth:
            # visible only if beacon ray is unobstructed
            depth_along_bearing = self._raycast_depth((x, y), yaw, bearing, max_depth)
            if depth_along_bearing >= dist_g - 0.05:
                col = int((0.5 + bearing / fov) * W)
                col = int(np.clip(col, 1, W - 2))
                stripe_h = int(H * 0.45)
                top = H // 2 - stripe_h // 2 - int(0.6 * (1 - min(dist_g, max_depth) / max_depth) * 0)
                img[:, col - 1 : col + 2] = np.array([255, 60, 40], dtype=np.uint8)  # beacon
        # ground/ceiling gradient for depth cue
        grad = np.linspace(160, 90, H).astype(np.uint8)[:, None]
        img = (img * (grad[:, :, None] / 128.0)).clip(0, 255).astype(np.uint8)
        return img

    # ------------------------------------------------------------------
    # Termination / episode bookkeeping (evaluation rules; one place only)
    # ------------------------------------------------------------------
    def is_success(self, pose: np.ndarray | None = None) -> bool:
        assert self.spec is not None
        if pose is None:
            pose = np.array(self._p)
        d = math.hypot(pose[0] - self.spec.goal_xy[0], pose[1] - self.spec.goal_xy[1])
        return bool(d <= self.spec.goal_radius_m)

    def properties(self) -> dict:
        return {
            "sim_family": self.sim_family,
            "semantics": self.sim_semantics,
            "world_size_m": WORLD_SIZE_M,
            "agent_radius_m": AGENT_RADIUS_M,
            "fov_deg": self.fov_deg,
            "control_dt_s": PD_DT,
            "max_v": MAX_LINEAR_V,
            "max_w": MAX_ANGULAR_V,
        }


def _ray_aabb(px, py, dx, dy, xmin, ymin, xmax, ymax):
    if dx == 0.0:
        if xmin <= px <= xmax:
            txmin, txmax = -math.inf, math.inf
        else:
            return None
    else:
        t1, t2 = (xmin - px) / dx, (xmax - px) / dx
        txmin, txmax = min(t1, t2), max(t1, t2)
    if dy == 0.0:
        if ymin <= py <= ymax:
            tymin, tymax = -math.inf, math.inf
        else:
            return None
    else:
        t1, t2 = (ymin - py) / dy, (ymax - py) / dy
        tymin, tymax = min(t1, t2), max(t1, t2)
    tmin, tmax = max(txmin, tymin), min(txmax, tymax)
    if tmax >= max(tmin, 0.0) and tmin > 0:
        return float(tmin)
    return None
