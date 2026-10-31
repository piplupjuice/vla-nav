"""Demo rendering: top-down trajectories, first-person strips, and MP4 export.

Used by the demo notebook and by tests. Only depends on matplotlib/PIL so it
runs locally without GPU. Records the method, model name and condition ON
the figure so a video cannot be mistaken for a different experimental setup.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:  # optional at import time so tests can run without matplotlib display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

from ..environment.mininav_env import MiniNavEnv
from ..environment.types import EpisodeSpec


def render_topdown(env: MiniNavEnv, spec: EpisodeSpec, method: str = "",
                   model_name: str = "", save_path: Optional[str] = None,
                   view_first_person: bool = False) -> "plt.Figure":
    """Top-down trajectory figure with first-person inset (optional)."""
    assert _HAS_MPL, "matplotlib is required for demo rendering"
    assert env.spec is spec, "env must be reset with the same spec first"
    poses = np.asarray(env._poses)
    fig, ax = plt.subplots(figsize=(6, 6))
    for cx, cy, w, h in env._scene["obstacles"]:
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                   color="0.6", alpha=0.85))
    lim = 6.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.plot(*spec.goal_xy, "*" if env.is_success() else "o", color="tab:red",
            markersize=18, label="goal")
    ax.plot(*poses[0, :2], "s", color="tab:green", markersize=10, label="start")
    ax.plot(poses[:, 0], poses[:, 1], "-", color="tab:blue", lw=2,
            label=f"trajectory ({method})")
    ax.set_title(f"{spec.condition} | {spec.episode_id}\n"
                 f"model={model_name or 'mock'} | success={env.is_success()}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if view_first_person:
        obs = env.observe()
        im = Image.fromarray(obs.image)
        ax_in = fig.add_axes([0.7, 0.72, 0.28, 0.25])
        ax_in.imshow(im)
        ax_in.axis("off")
        ax_in.set_title("first-person", fontsize=7)
    return fig


def render_frames_video(env: MiniNavEnv, spec: EpisodeSpec, driver, out_path: str,
                        fps: int = 6, max_frames: int = 240) -> Optional[str]:
    """Render an MP4 from a run by re-driving the environment frame-by-frame.

    `driver(env, obs, spec) -> Cmd` re-uses the same regime as the runner in
    a read-only way so the video shows exactly what was measured. Uses
    matplotlib animation -> mp4 if a writer exists, else falls back to a
    strip of PNG frames with the same path prefix and returns the frames dir.
    """
    assert _HAS_MPL
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    obs = env.reset(spec)
    frames = [obs.image.copy()]
    done = False
    t = 0
    while not done and t < spec.max_steps and t < max_frames:
        cmd = driver(env, obs, spec)
        env.step(cmd)
        obs = env.observe()
        frames.append(obs.image.copy())
        t += 1
        done = env.is_success()
    # build figure anim
    fig, ax = plt.subplots(figsize=(4, 2.2), dpi=110)
    ax.set_axis_off()
    im = ax.imshow(frames[0])
    ax.set_title(spec.instruction, fontsize=7)

    def draw(i):
        im.set_data(frames[i])
        return (im,)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps,
                             blit=True)
        writer = FFMpegWriter(fps=fps, metadata={
            "model": str(env.spec.instruction),
            "env": env.sim_family,
        })
        anim.save(out_path, writer=writer)
        return out_path
    except Exception:
        frames_dir = out_path.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        step = max(1, len(frames) // 24)
        for i, fr in enumerate(frames[::step]):
            Image.fromarray(fr).save(os.path.join(frames_dir, f"{i:03d}.png"))
        return frames_dir
    finally:
        import matplotlib.pyplot as _p
        _p.close(fig)
