"""Benchmark episode generation: deterministic splits, seeds, and shifts.

Naming and structure follow the project protocol (`docs/project_report.md`
section "Experimental protocol"): the SAME scene seed family generates a
`val` split (used for threshold tuning + method selection) and a strictly
separate `test` split frozen before any method comparison. World seeds differ
across conditions; instruction rephrasings for condition C are enumerated.

Non-goals: no training happens in this project, so "val/test" refers to what
is *allowed to influence offline decisions* (thresholds, model choice,
episode-count checks) vs what is reserved for the final reported numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .types import EpisodeSpec

CONFIRMED_SEED_EQUIVALENCE = "numpy default_rng(PCG64); identical on any platform"

#: Scene seed ranges per condition-domain (fixed constants, versioned).
SEED_RANGES: Dict[str, Tuple[int, int]] = {
    "A_in_dist":   (10000, 10199),
    "B_env_shift": (20000, 20199),
    "C_instr_shift": (30000, 30199),
    "D_recovery":  (40000, 40199),
}

#: Number of episodes per condition per split (small enough for a GPU pilot).
N_EPISODES: Dict[str, int] = {
    "A_in_dist": 32,
    "B_env_shift": 32,
    "C_instr_shift": 24,
    "D_recovery": 24,
}
N_VAL_TARGET: Dict[str, int] = {
    "A_in_dist": 12, "B_env_shift": 12, "C_instr_shift": 12, "D_recovery": 12,
}

INSTRUCTIONS_A = [
    "Navigate to the marked beacon.",
    "Walk straight ahead until you reach the beacon.",
]
INSTRUCTIONS_C = [
    # Paraphrase / ambiguity / conflict battery — same underlying goal.
    "Please headwhay to the beacon.",          # typo parody (mild)
    "Make your way toward the flashing target.",
    "Go to the beacon, but walking is optional.",
    "Reach the beacon and stop exactly on it.",
    # Ambiguous
    "Move forward toward something.",
    "Find the target and get there.",
    # Conflicting (safety-relevant, clearly documented)
    "Go to the beacon, but do not move at all.",
    "Head to the beacon away from it.",
]
INSTRUCTIONS_B = INSTRUCTIONS_A  # environmental shift reuses clear wording


def _scene_params(cond: str, seed: int) -> dict:
    """Scenario configuration that differs per condition (explicit, logged)."""
    rng = np.random.default_rng(seed)
    if cond == "A_in_dist":
        return dict(clutter="standard", width_factor=1.0)
    if cond == "B_env_shift":
        # shift = increased clutter + narrower corridors among obstacles
        return dict(clutter="high", width_factor=0.75)
    if cond == "C_instr_shift":
        return dict(clutter="standard", width_factor=1.0)
    if cond == "D_recovery":
        return dict(clutter="standard", width_factor=1.0)
    raise ValueError(cond)


def make_episode(cond: str, split: str, idx: int, *, version: str = "v1") -> EpisodeSpec:
    """One reproducible episode. Callers must not mutate the returned spec."""
    lo, hi = SEED_RANGES[cond]
    # Splits are *different seeds* in the same family; test seeds are never
    # used for threshold tuning. Seeds are versioned so logs tag them.
    if split == "val":
        seed = lo + idx            # val uses the FIRST n seeds
    elif split == "test":
        seed = hi - 1 - (idx % ((hi - lo) // 2))  # test seeds come from the upper half
    else:
        raise ValueError(split)
    n = N_VAL_TARGET[cond] if split == "val" else N_EPISODES[cond]
    if idx >= n:
        raise IndexError(f"{cond}/{split}: episode index {idx} >= {n}")
    rng = np.random.default_rng(seed)
    params = _scene_params(cond, seed)
    world_id = seed  # world seed == episode seed (documented; not reused)
    start = (float(rng.uniform(-4.5, 4.5)), float(rng.uniform(-4.5, 4.5)))
    # goal: deterministic relative placement, min separation, INSIDE the world
    # (world is the 12x12 box; MiniNav walls live at +-6 m, so any goal must
    # satisfy |coord| <= 5.0 to keep the straight segment inside the box).
    ang = float(rng.uniform(0, 2 * np.pi))
    dist = float(rng.uniform(4, 9))
    goal_lim = 5.0
    gx, gy = start[0] + dist * np.cos(ang), start[1] + dist * np.sin(ang)
    gx, gy = float(np.clip(gx, -goal_lim, goal_lim)), float(np.clip(gy, -goal_lim, goal_lim))
    # enforce minimum separation after clipping (grow radially if needed)
    sep = np.hypot(gx - start[0], gy - start[1])
    if sep < 4.0:
        scale = 4.0 / max(sep, 1e-9)
        gx = float(np.clip(start[0] + scale * (gx - start[0]), -goal_lim, goal_lim))
        gy = float(np.clip(start[1] + scale * (gy - start[1]), -goal_lim, goal_lim))
    goal = (gx, gy)
    if cond == "C_instr_shift":
        instr_idx = (idx + version.count("v")) % len(INSTRUCTIONS_C)
        instr = INSTRUCTIONS_C[instr_idx]
    else:
        instr = INSTRUCTIONS_A[idx % len(INSTRUCTIONS_A)]
    if cond == "D_recovery":
        # D_recovery = "lost start": yaw points directly AWAY from the goal
        dxg, dyg = goal[0] - start[0], goal[1] - start[1]
        start_yaw = float(np.arctan2(dyg, dxg) + np.pi)
    else:
        start_yaw = 0.0
    return EpisodeSpec(
        episode_id=f"{cond}/{split}/{version}/{idx:03d}",
        condition=cond, split=split, seed=seed, world_id=world_id,
        start_xy=start, start_yaw_rad=start_yaw, goal_xy=goal,
        instruction=instr, max_steps=steps_for(cond),
        goal_radius_m=goal_radius_for(cond),
        extra={"clutter": params["clutter"], "width_factor": params["width_factor"],
               "benchmark_version": version},
    )


def steps_for(cond: str) -> int:
    return {"A_in_dist": 200, "B_env_shift": 260, "C_instr_shift": 200,
            "D_recovery": 240}[cond]


def goal_radius_for(cond: str) -> float:
    return {"A_in_dist": 0.6, "B_env_shift": 0.6, "C_instr_shift": 0.7,
            "D_recovery": 0.6}[cond]


def list_episodes(conds: List[str], split: str, version: str = "v1") -> List[EpisodeSpec]:
    out: List[EpisodeSpec] = []
    for c in conds:
        n = N_VAL_TARGET[c] if split == "val" else N_EPISODES[c]
        out.extend(make_episode(c, split, i, version=version) for i in range(n))
    return out
