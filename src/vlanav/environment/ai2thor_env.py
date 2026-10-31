"""AI2-THOR navigation environment conforming to the MiniNav env contract.

STATUS AND ROLE:
    This IS the paper-claims environment (`sim_family=ai2thor`): real
    photorealistic Unity rendering, real collision physics, intact objects.
    The VLA (OmniVLA) navigates through egocentric RGB + a natural-language
    goal instruction — its nominal domain.

Honesty notes (documented design constraints):
    - The official OmniVLA deployment emits (linear_v, angular_v) at
      dt = 1/3 s. AI2-THOR is a discrete-action simulator, so each tick maps
      the command deterministically to `Rotate(degrees=omega*dt)`,
      `MoveAhead(moveMagnitude=v*dt)`; the mapping is identical for every
      regime and logged in `properties()`. There is no invention of reward
      or motion here: motion comes from THOR physics (MoveAhead fails and
      stands still when blocked; the pose delta before/after is the truth).
    - Goal conditioning is LANGUAGE-ONLY (official modality 7). The
      environment never draws a beacon/arrow into the frame: `frames` are
      exactly what Unity renders; goal positions exist only in
      EpisodeSpec bookkeeping (logged) and in the instruction text handed
      to the policy.
    - Determinism: scene choice, spawn pose, target object, and instruction
      template are all chosen from one numpy Generator seeded by
      `spec.seed`, so runs replay bit-identically (THOR object randomization
      is NOT used — it is not seedable).
    - The Controller is created and owned by the caller (the Kaggle
      notebook), so GPU/rendering mode configuration lives in run code, not
      here.
"""
from __future__ import annotations

import math
import numpy as np

from .types import Cmd, EpisodeSpec, Observation, StepInfo
from .types import MAX_LINEAR_V, MAX_ANGULAR_V, PD_DT

#: Disjoint scene-ID blocks (documented convention; scenes are numbered and
#: each number maps to a distinct Unity floor plan).
SCENES_VAL = [f"FloorPlan{n}" for n in range(1, 31)]        # FloorPlan1..30
SCENES_TEST = [f"FloorPlan{n}" for n in range(401, 431)]    # FloorPlan401..430
SCENES_ENV_SHIFT = [f"FloorPlan{n}" for n in range(301, 331)]

_GOAL_RADIUS_M = 0.6


def thor_yaw_from_math_deg(math_deg: float) -> float:
    """Convert a math heading (0 = +x, CCW) to AI2-THOR's yaw in degrees.

    nb07 evidence: THOR yaw is 0 at +z and grows clockwise, so
    `thor = 90 - math`. Used for the deterministic spawn heading (condition D).
    """
    return float((90.0 - float(math_deg)) % 360.0)

# natural-language instruction templates. A = canonical phrasing (nominal
# distribution); C_instr_shift = deliberate paraphrases of the SAME goal.
_TPL_A = [
    "Navigate to the {target} and stop there.",
    "Go to the {target}.",
]
_TPL_C = [
    "Find the {target} in the scene and stand next to it.",
    "Where is the {target}? Move until you reach it.",
    "Head over to the {target} and stop when you're close.",
    "Your goal: reach the {target}.",
]


class Ai2ThorNavEnv:
    """Language-goal navigation in AI2-THOR under the NavPolicy-API contract."""

    sim_family = "ai2thor"
    sim_semantics = "photorealistic RGB (Unity); goal via language only; THOR physics"

    def __init__(self, controller) -> None:   # ai2thor.controller.Controller
        self.controller = controller
        self.spec: EpisodeSpec | None = None
        self._pose: tuple[float, float, float] | None = None  # (x, z, yaw_rad)
        self._step_count = 0
        self._collision_total = 0
        self._path_length = 0.0
        self._target_object_id: str | None = None
        self._room_name: str | None = None
        self._trace: list[tuple[float, float, float]] = []

    # BaseRegime.run_episode reads env._p for the final distance metric.
    @property
    def _p(self) -> tuple[float, float, float]:
        return self._pose

    # ------------------------------------------------------------------
    def _scene_name(self, spec: EpisodeSpec) -> str:
        pool = SCENES_TEST if spec.split == "test" else SCENES_VAL
        if spec.condition == "B_env_shift":
            pool = SCENES_ENV_SHIFT
        h = (int(spec.world_id) * 2654435761 + int(spec.seed)) % (2**31)
        return pool[h % len(pool)]

    def _pick_target(self, objects: list[dict], rng: np.random.Generator) -> dict:
        """Deterministic language-goable target: a visible, pickup-less,
        med-section furniture-like object far enough from the spawn."""
        OK_TYPES = {
            "CounterTop", "DiningTable", "Sofa", "ArmChair", "Bed", "Bathtub",
            "Toaster", "Microwave", "Fridge", "Television", "DeskLamp",
            "PottedPlant", "GarbageCan", "Shelf", "Drawer", "StoveBurner",
            "Window", "Faucet", "LightSwitch", "Chair", "SideTable", "CoffeeMachine",
        }
        cands = [o for o in objects
                 if o.get("breakable") is False
                 and o["objectId"] and "|" in o["objectId"]
                 and o["objectType"] in OK_TYPES
                 and o.get("visible")]
        # fall back to any non-floor object if nothing matched
        if not cands:
            cands = [o for o in objects if o["objectId"] and "|" in o["objectId"]]
        assert cands, "no usable target objects in the scene"
        return cands[int(rng.integers(0, len(cands)))]

    def reset(self, spec: EpisodeSpec) -> Observation:
        scene = self._scene_name(spec)
        rng = np.random.default_rng(int(spec.seed))
        ev = self.controller.reset(scene=scene)
        ev = self.controller.step(dict(action="Done"))  # settle

        m = ev.metadata
        self._room_name = m.get("sceneName")

        ag = m["agent"]
        x, z = ag["position"]["x"], ag["position"]["z"]
        start_xy = (float(x), float(z))

        # target selection + instruction are DETERMINISTIC given the seed;
        # they are recorded INTO spec.extra (the spec is the provenance).
        target = self._pick_target(m["objects"], rng)
        self._target_object_id = target["objectId"]
        tx, tz = float(target["position"]["x"]), float(target["position"]["z"])

        # deterministic spawn yaw from the seed (keep THOR's default cell).
        # D_recovery semantics (same as `episodes.make_episode`): the robot
        # starts facing DIRECTLY AWAY from the goal, so it must turn around /
        # recover rather than walk a mostly-clear path.
        if spec.condition == "D_recovery":
            math_deg = (math.degrees(math.atan2(tz - start_xy[1], tx - start_xy[0]))
                        + 180.0) % 360.0
            yaw_deg = thor_yaw_from_math_deg(math_deg)
        else:
            yaw_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0]))

        if spec.condition == "C_instr_shift":
            tpl = _TPL_C[int(rng.integers(0, len(_TPL_C)))]
        else:
            tpl = _TPL_A[int(rng.integers(0, len(_TPL_A)))]
        instruction = tpl.format(target=target["objectType"].lower())

        spec.extra.update({
            "scene": scene,
            "target_object_id": self._target_object_id,
            "target_type": target["objectType"],
            "instruction": instruction,
            "start_xy_actual": [round(start_xy[0], 3), round(start_xy[1], 3)],
            "goal_xy_actual": [round(tx, 3), round(tz, 3)],
            "start_yaw_deg": yaw_deg,
        })
        # the carriers the regimes/logger read:
        spec.goal_xy = (tx, tz)
        spec.start_xy = start_xy
        spec.instruction = instruction

        # teleport on the spot with the chosen heading (walkable cell).
        # v5 evidence (2026-10-30): AI2-THOR 5.0 `Teleport` accepts only
        # position/rotation/horizon/standing/forceAction -- `rotateOnTeleport`
        # raises ValueError and killed the whole run. `rotation` IS the yaw.
        ev = self.controller.step(dict(
            action="Teleport",
            position=dict(x=start_xy[0], y=ag["position"]["y"], z=start_xy[1]),
            rotation=float(yaw_deg), forceAction=True))
        if not ev.metadata["lastActionSuccess"]:
            # stationary teleport, then rotate in place to the chosen heading
            ev = self.controller.step(dict(
                action="Teleport",
                position=dict(x=start_xy[0], y=ag["position"]["y"], z=start_xy[1]),
                forceAction=True))
            cur_deg = float(ev.metadata["agent"]["rotation"]["y"])
            delta = (float(yaw_deg) - cur_deg + 180.0) % 360.0 - 180.0
            if abs(delta) >= 1.0:
                act = "RotateLeft" if delta > 0 else "RotateRight"
                ev = self.controller.step(dict(action=act, degrees=abs(delta)))

        self._pose = self._pose_from(ev)
        self.spec = spec
        self._step_count = 0
        self._collision_total = 0
        self._path_length = 0.0
        self._trace = [self._pose]
        return self.observe()

    def _pose_from(self, event) -> tuple[float, float, float]:
        ag = event.metadata["agent"]
        p = ag["position"]
        # nb07 evidence (2026-10-30): THOR's `rotation.y` is 0 at +z and grows
        # CLOCKWISE (RotateRight +30, RotateLeft -30), while the pipeline (and
        # the official PD decode) uses the math convention in the (x, z) plane:
        # 0 = +x, growing counter-clockwise. `yaw_math = 90deg - yaw_thor`.
        thor_deg = float(ag["rotation"]["y"])
        yaw = math.radians(90.0 - thor_deg)
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        return (float(p["x"]), float(p["z"]), float(yaw))

    def observe(self) -> Observation:
        x, z, yaw = self._pose
        frame = self.controller.last_event.frame  # HxWx3 uint8 (real render)
        return Observation(image=np.asarray(frame, dtype=np.uint8),
                           robot_pose=np.array([x, z, yaw]),
                           step=self._step_count)

    def step(self, cmd: Cmd) -> StepInfo:
        assert self.spec is not None, "call reset() first"
        v = float(np.clip(cmd.linear_v, 0.0, MAX_LINEAR_V))
        w = float(np.clip(cmd.angular_v, -MAX_ANGULAR_V, MAX_ANGULAR_V))
        dt = PD_DT

        x0, z0, yaw0 = self._pose
        moved, collision = 0.0, False

        rot_deg = w * dt * 180.0 / math.pi
        if abs(rot_deg) >= 1e-2:
            # positive w = counterclockwise (left); ai2thor's valid actions
            # are RotateLeft/RotateRight (+ "Rotate" is not portable).
            act = ("RotateLeft", abs(rot_deg)) if rot_deg > 0 else \
                  ("RotateRight", abs(rot_deg))
            ev = self.controller.step(dict(action=act[0], degrees=act[1]))
            if ev.metadata["lastActionSuccess"]:
                self._pose = self._pose_from(ev)

        trans = min(v * dt, 0.25)  # THOR moveMagnitude bound
        if trans >= 1e-2:
            ev = self.controller.step(dict(action="MoveAhead", moveMagnitude=trans))
            p = self._pose_from(ev)
            d = math.hypot(p[0] - x0, p[1] - z0)
            moved = d
            # collision: MoveAhead reported success but produced < half the
            # requested displacement (THOR stops against objects)
            collision = (moved + 1e-6) < 0.5 * trans
            self._pose = p

        yaw = self._pose[2]
        self._path_length += moved
        self._step_count += 1
        if collision:
            self._collision_total += 1
        self._trace.append(self._pose)
        return StepInfo(collision=bool(collision), moved_distance_m=float(moved),
                        step=self._step_count)

    # ------------------------------------------------------------------
    def shortest_path_length_m(self) -> float | None:
        """Standard-SPL denominator: shortest walkable path start -> goal.

        Uses AI2-THOR's `GetShortestPath` metadata action (no agent motion,
        safe to call right after reset). Returns None when THOR refuses or
        yields no usable path — the episode then logs shortest_path_length_m
        = NaN and spl_standard = NaN, and the straight-line proxy stays the
        only (optimistic, labelled) denominator. Never silently substituted.
        """
        assert self.spec is not None and self._pose is not None
        sx, sz, _ = self._pose
        gx, gz = self.spec.goal_xy
        try:
            ev = self.controller.step(
                action="GetShortestPath",
                x=float(sx), y=0.0, z=float(sz),
                position={"x": float(gx), "y": 0.0, "z": float(gz)})
        except Exception:
            return None
        if not ev.metadata.get("lastActionSuccess", False):
            return None
        path = ev.metadata.get("action") or ev.metadata.get("shortest_path") or []
        try:
            pts = [(float(p["x"]), float(p["z"])) for p in path]
        except (TypeError, KeyError, ValueError):
            return None
        if len(pts) < 2:
            return None
        total = 0.0
        for (ax, az), (bx, bz) in zip(pts, pts[1:]):
            total += math.hypot(bx - ax, bz - az)
        return total if total > 0 else None

    # ------------------------------------------------------------------
    def is_success(self, pose: np.ndarray | None = None) -> bool:
        assert self.spec is not None
        if pose is None:
            pose = np.array(self._pose)
        gx, gz = self.spec.goal_xy
        d = math.hypot(pose[0] - gx, pose[1] - gz)
        return bool(d <= _GOAL_RADIUS_M)

    def properties(self) -> dict:
        return {
            "sim_family": self.sim_family,
            "semantics": self.sim_semantics,
            "goal_radius_m": _GOAL_RADIUS_M,
            "control_dt_s": PD_DT,
            "max_v": MAX_LINEAR_V,
            "max_w": MAX_ANGULAR_V,
            "actuation": "Rotate(degrees=omega*dt) then MoveAhead(moveMagnitude=v*dt)",
            "actuation_grid_note": (
                "the Controller MUST be built with gridSize=0.05: with THOR's "
                "default 0.25 m grid, MoveAhead(0.1 m) moves 0.0 m (nb07 "
                "evidence 2026-10-30) and every episode stalls"),
            "goal_conditioning": "language only (official modality 7)",
            "scenes_val": SCENES_VAL[0] + ".." + SCENES_VAL[-1],
            "scenes_test": SCENES_TEST[0] + ".." + SCENES_TEST[-1],
            "scenes_env_shift": SCENES_ENV_SHIFT[0] + ".." + SCENES_ENV_SHIFT[-1],
        }
