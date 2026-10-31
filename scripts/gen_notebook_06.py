"""Generate notebook 06_demo.ipynb: real THOR frames -> MP4 demo (3 regimes).

Honest demo rules:
  * SAME two val episodes, SAME seeds, recorded for B1, B2 and B3 -> the
    viewer sees the cost/behaviour difference, not a curated winner.
  * EVERY tick's real 400x400 frame is kept (no frame cherry-picking).
  * Overlay values (step, v, w, distance-to-goal, instruction) are the logged
    values of that exact tick; the pose is captured WITH the frame (no
    end-of-episode pose reused for earlier frames).
  * B3 threshold comes from the val-only tuning artifact when available,
    otherwise 0.2 with the fallback logged in demo_meta.json.
Output: /kaggle/working/demo/*.mp4 (H.264, 8 fps), keyframes, demo_meta.json.

Run:  python scripts/gen_notebook_06.py
Then: kaggle kernels push -p kaggle_notebooks/06_demo
"""
import ast
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "kaggle_notebooks", "06_demo")
os.makedirs(OUT, exist_ok=True)

md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t}
code = lambda t: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t}

cells = [
    md("""# VLA-NAV - Notebook 06: real-frame MP4 demo (B1 vs B2 vs B3)

Two fixed val episodes, each recorded for all three regimes with the real
OmniVLA checkpoint in real AI2-THOR. Every control tick is kept; the overlay
shows that tick's logged command and true distance-to-goal.

**Honest:** no re-runs for a prettier take, no dropped frames, no edited
overlays. If the robot stalls, spins or collides, it is in the video."""),

    code(r'''%%time
import json, os, platform, subprocess, time

os.makedirs("/kaggle/tmp", exist_ok=True)
os.makedirs("/kaggle/working/demo", exist_ok=True)
os.environ["AI2THOR_DOWNLOAD_DIR"] = "/kaggle/tmp/aa"
os.environ["OBJECT_CACHE_DIR"] = "/kaggle/tmp/thor_objcache"
os.environ["XDG_CACHE_HOME"] = "/kaggle/tmp/xdgcache"

import torch
print("torch", torch.__version__, "| gpus",
      [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())])

def cmd(s, must=True):
    print("$", s)
    r = subprocess.run(s, shell=True, capture_output=True, text=True)
    print(r.stdout[-800:]); print(r.stderr[-800:])
    assert (r.returncode == 0) if must else True
    return r.returncode

cmd("pip install -q 'timm==0.9.16' 'peft==0.11.1' 'draccus==0.8.0' 'utm' "
    "'efficientnet_pytorch' 'ai2thor' 'imageio' 'imageio-ffmpeg'")
cmd("apt-get update -qq", must=False)
for _p in ("xvfb", "x11-utils", "mesa-utils", "libgl1", "libglu1-mesa",
           "libegl1", "libxcursor1", "libxinerama1", "libxrandr2",
           "libxi6", "libxss1", "libnss3", "libvulkan1", "vulkan-tools"):
    if cmd(f"apt-get install -y -qq --no-install-recommends {_p}", must=False) != 0:
        cmd(f"apt-get install -y -qq --no-install-recommends {_p}t64", must=False)

cmd("rm -rf /kaggle/tmp/OmniVLA", must=False)
cmd("git clone --depth 1 https://github.com/NHirose/OmniVLA.git /kaggle/tmp/OmniVLA")
cmd("pip install -q --no-deps 'transformers==4.40.1' 'huggingface_hub==0.29.1'")

import importlib.util
_tf_root = os.path.dirname(importlib.util.find_spec("transformers").origin)
_tbl = os.path.join(_tf_root, "dependency_versions_table.py")
src = open(_tbl, encoding="utf-8").read()
needle = '"tokenizers": "tokenizers>=0.19,<0.20"'
assert needle in src, "unexpected tokenizers spec"
open(_tbl, "w", encoding="utf-8").write(
    src.replace(needle, '"tokenizers": "tokenizers>=0.19"'))

import sys
sys.path.insert(0, "/kaggle/tmp/OmniVLA")
from huggingface_hub import snapshot_download
# UNIQUE per-run directory (2026-10-30 evidence: two concurrent kernels that
# shared /kaggle/tmp/omnivla-original corrupted each other's checkpoint).
import uuid as _uuid
_RUN = (os.environ.get("KAGGLE_KERNEL_SLUG", "run") + "-"
        + _uuid.uuid4().hex[:6]).replace("/", "_")
VLA_PATH = "/kaggle/tmp/omnivla-" + _RUN
CACHE_DIR = "/kaggle/tmp/hfcache-" + _RUN
print("VLA_PATH =", VLA_PATH)
NEED = [f"model-0000{i}-of-00004.safetensors" for i in (1, 2, 3, 4)] + [
    "action_head--120000_checkpoint.pt", "proprio_projector--120000_checkpoint.pt",
    "config.json", "processing_prismatic.py", "modeling_prismatic.py",
    "configuration_prismatic.py"]

def _missing(root):
    bad = []
    for f in NEED:
        p = os.path.join(root, f)
        if not os.path.isfile(p) or os.path.getsize(p) < 1000:
            bad.append(f)
    return bad

# v3 evidence: verify sizes and allow one forced retry instead of failing later
for _attempt in (1, 2):
    snapshot_download(repo_id="NHirose/omnivla-original", local_dir=VLA_PATH,
                      cache_dir=CACHE_DIR,
                      force_download=(_attempt == 2))
    _miss = _missing(VLA_PATH)
    print("download attempt", _attempt, "| missing:", _miss)
    if not _miss:
        break
if _miss:
    raise RuntimeError(f"checkpoint incomplete after retry: {_miss}")
print("setup OK | head sizes MB:", {f: round(os.path.getsize(os.path.join(VLA_PATH, f)) / 1e6, 1)
                                      for f in NEED if f.endswith((".pt", ".json"))})'''),

    code(r'''import sys, os
cands = []
for root, dirs, _files in os.walk("/kaggle/input"):
    if "vlanav" in dirs and os.path.isfile(os.path.join(root, "vlanav", "__init__.py")):
        cands.append(root)
if len(cands) != 1:
    raise RuntimeError(f"expected exactly one vlanav package, got {cands}")
SRC_ROOT = cands[0]
sys.path.insert(0, SRC_ROOT)
print("SRC_ROOT =", SRC_ROOT)
_stamp = os.path.join(os.path.dirname(SRC_ROOT), "BUILD.txt")
_stamp_txt = open(_stamp).read() if os.path.isfile(_stamp) else ""
print("BUILD:", _stamp_txt or "(no stamp)")
# stale-dataset guard (2026-10-30): a mount without the actuation fixes makes
# MoveAhead(0.1 m) a silent no-op and the demo records a robot that never moves.
for _need in ("contains_RotateLeft=True", "contains_D_recovery_yaw=True"):
    assert _need in _stamp_txt, f"STALE DATASET: BUILD.txt lacks {_need}"

from vlanav.model.omnivla_adapter import OmnivlaNavPolicy
from vlanav.environment.ai2thor_env import Ai2ThorNavEnv
from vlanav.reliability.replanning import (ChunkedFrozen, FixedFrequencyReplan,
                                           ReliabilityTriggered)
from vlanav.environment.types import EpisodeSpec, Cmd

policy = OmnivlaNavPolicy(VLA_PATH,
                          prismatic_repo="/kaggle/tmp/OmniVLA", load_now=False)
# pre-load re-verification: /kaggle/tmp is shared with any other running
# kernel, so a file present in the setup cell can be gone by the load cell.
_vanish = [f for f in NEED
           if not (os.path.isfile(os.path.join(VLA_PATH, f))
                   and os.path.getsize(os.path.join(VLA_PATH, f)) > 1000)]
assert not _vanish, f"checkpoint files missing before load: {_vanish}"
policy.load()
assert policy.is_model
print("policy loaded:", policy.provenance.get("load_seconds"), "s")'''),

    code(r'''%%time
import os, subprocess, time
os.environ["DISPLAY"] = ":0"
subprocess.run("pkill Xvfb || true", shell=True)
subprocess.Popen(["Xvfb", ":0", "-screen", "0", "1024x768x24", "-ac",
                  "+extension", "GLX", "+render", "-noreset"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 start_new_session=True)
time.sleep(4)
from ai2thor.controller import Controller
controller = Controller(scene="FloorPlan3", width=400, height=400,
                        platform="Linux64", server_type="Linux64", gridSize=0.05)
ev = controller.reset(scene="FloorPlan3")
print("controller OK | frame", ev.frame.shape)'''),

    code(r'''%%time
# 3-tick smoke episode (action/argument gate; v5 evidence)
from vlanav.environment.types import EpisodeSpec as _Spec
env_smoke = Ai2ThorNavEnv(controller)
spec_smoke = _Spec(episode_id="smoke-000", condition="A_in_dist", split="val",
                   seed=991, world_id=991, start_xy=(0.0, 0.0),
                   start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                   instruction="(set at reset)", max_steps=3,
                   goal_radius_m=0.6, extra={"smoke": True})
_smoke = ChunkedFrozen(policy, "chunked_frozen").run_episode(env_smoke, spec_smoke)
print("SMOKE OK | steps", _smoke.step, "| scene", spec_smoke.extra.get("scene"),
      "| moved", round(_smoke.path_length_m, 3), "m")
assert _smoke.step >= 1
# movement gate (nb07): a stalled robot would make the demo a video of nothing
assert _smoke.path_length_m > 0.02, (
    "SMOKE FAILED: robot moved %.3f m in 3 ticks -> actuation broken"
    " (grid/yaw); refusing to record an empty demo" % _smoke.path_length_m)'''),

    code(r'''# B3 threshold: val-tuning artifact only, else documented fallback
import glob as _g
thr_cfg, thr_source = None, None
for p in _g.glob("/kaggle/input/**/threshold_choice.json", recursive=True):
    try:
        d = json.load(open(p))
        if "threshold_m" in d:
            thr_cfg, thr_source = d, p
            break
    except Exception:
        continue
if thr_cfg is None:
    thr_cfg = {"threshold_m": 0.2, "stall_limit": 2,
               "selection_rule": "FALLBACK: tuning artifact absent; 0.2 used"}
    thr_source = "(none)"
THRESHOLD = float(thr_cfg["threshold_m"])
STALL_LIMIT = int(thr_cfg.get("stall_limit", 2))
print("threshold", THRESHOLD, "stall", STALL_LIMIT, "from", thr_source)'''),

    md("## Record every tick of every regime on the same two episodes"),

    code(r'''%%time
import math, os, time
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

env = Ai2ThorNavEnv(controller)
SEEDS = [30001, 30002]          # fixed val demo seeds (documented)
MAX_STEPS = 60
COND = "A_in_dist"

def spec_for(i, seed):
    return EpisodeSpec(episode_id=f"demo-{i}-{seed}", condition=COND, split="val",
                       seed=seed, world_id=200 + i,
                       start_xy=(0.0, 0.0), start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                       instruction="(filled at reset from THOR target)",
                       max_steps=MAX_STEPS, goal_radius_m=0.6,
                       extra={"demo": True, "screen": "recorded"})

def regime_for(name):
    if name == "chunked_frozen":
        return ChunkedFrozen(policy, "chunked_frozen")
    if name == "fixed_frequency":
        return FixedFrequencyReplan(policy, "fixed_frequency")
    return ReliabilityTriggered(policy, "reliability_triggered",
                                threshold_m=THRESHOLD,
                                stall_limit=STALL_LIMIT,
                                max_replans_per_episode=24)

def overlay(img, spec, step, cmd, dist, regime_name):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, im.width, 66], fill=(0, 0, 0, 160))
    d.text((6, 3), f"{regime_name}  step {step}/" + str(spec.max_steps),
           fill=(255, 255, 0))
    d.text((6, 19), f"v={cmd.linear_v:.2f} m/s  w={cmd.angular_v:+.2f} rad/s",
           fill=(0, 255, 255))
    d.text((6, 35), f"dist_to_goal={dist:.2f} m", fill=(0, 255, 120))
    d.text((6, 51), spec.instruction[:52], fill=(255, 255, 255))
    return np.asarray(im)

def title_card(text, size=(400, 400)):
    im = Image.new("RGB", size, (16, 16, 16))
    d = ImageDraw.Draw(im)
    d.text((10, size[1] // 2 - 20), text, fill=(240, 240, 240))
    return np.asarray(im)

# monkeypatch capture: frame + pose + command are stored TOGETHER per tick
def record_episode(spec, regime_name):
    frames = []
    orig_reset, orig_step = env.reset, env.step

    def rec_reset(s, _o=orig_reset):
        obs = _o(s)
        frames.append((obs.image.copy(), obs.robot_pose.copy(), 0, Cmd(0.0, 0.0)))
        return obs

    def rec_step(c, _o=orig_step):
        info = _o(c)
        obs = env.observe()
        frames.append((obs.image.copy(), obs.robot_pose.copy(),
                       env._step_count, c))
        return info

    env.reset, env.step = rec_reset, rec_step
    try:
        res = regime_for(regime_name).run_episode(env, spec)
    finally:
        env.reset, env.step = orig_reset, orig_step
    return res, frames

REGIMES = ["chunked_frozen", "fixed_frequency", "reliability_triggered"]
all_meta = []
for i, seed in enumerate(SEEDS):
    spec = spec_for(i, seed)
    per_regime_frames = {}
    for rname in REGIMES:
        res, frames = record_episode(spec, rname)
        gx, gz = spec.goal_xy
        rendered = []
        for img, pose, step, c in frames:
            dist = math.hypot(float(pose[0]) - gx, float(pose[1]) - gz)
            rendered.append(overlay(img, spec, step, c, dist, rname))
        per_regime_frames[rname] = rendered
        mp4 = f"/kaggle/working/demo/ep{i}_{seed}_{rname}.mp4"
        imageio.mimwrite(mp4, rendered, fps=8, codec="libx264",
                         macro_block_size=None)
        Image.fromarray(rendered[-1]).save(
            f"/kaggle/working/demo/ep{i}_{seed}_{rname}_last.jpg")
        meta = {"episode_index": i, "seed": seed, "world_id": spec.world_id,
                "regime": rname, "success": bool(res.success),
                "steps": int(res.step), "inferences": int(res.n_inferences),
                "replans": int(res.n_replans_triggered),
                "collisions": int(res.n_collisions),
                "dist_final_m": round(float(res.distance_to_goal_m), 3),
                "instruction": spec.instruction,
                "scene": spec.extra.get("scene"),
                "frames": len(rendered),
                "mp4": os.path.basename(mp4),
                "mp4_kb": os.path.getsize(mp4) // 1024}
        all_meta.append(meta)
        print(json.dumps(meta))

    # one combined video per episode: title card + B1 + B2 + B3 segments
    combo = []
    for rname in REGIMES:
        combo.append(title_card(f"ep{i} seed {seed}  |  {rname}  |  "
                                f"thr={THRESHOLD} m"))
        combo.extend(per_regime_frames[rname])
    combo_mp4 = f"/kaggle/working/demo/ep{i}_{seed}_B1_B2_B3.mp4"
    imageio.mimwrite(combo_mp4, combo, fps=8, codec="libx264",
                     macro_block_size=None)
    print("combined:", combo_mp4, os.path.getsize(combo_mp4) // 1024, "KB")

json.dump({"threshold_m": THRESHOLD, "stall_limit": STALL_LIMIT,
           "threshold_source": thr_source, "max_steps": MAX_STEPS,
           "condition": COND, "seeds": SEEDS, "episodes": all_meta,
           "note": ("every tick recorded; regimes run on the same episodes; "
                    "no cherry-picking or frame edits")},
          open("/kaggle/working/demo/demo_meta.json", "w"), indent=2)
controller.stop()
print("DEMO DONE:", len(all_meta), "clips")'''),

    md("""## What the video shows / does not show

Shows: every tick of three real regimes on two identical episodes, with the
actual decoded (v, ω), the live distance-to-goal, and the language instruction.
Does not show: edited frames, a re-run for a nicer take, or any claim beyond
the logged episode result.""")
]

for i, c in enumerate(cells):
    c.setdefault("id", f"cell-{i:02d}")
    if c["cell_type"] == "code":
        plain = "\n".join(l for l in "".join(c["source"]).split("\n")
                          if not l.lstrip().startswith(("%%", "%")))
        ast.parse(plain)

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(os.path.join(OUT, "06_demo.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    print("wrote", f.name)

meta = {"id": "pikanigga/vlanav-06-demo", "title": "vlanav-06-demo",
        "code_file": "06_demo.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
        "enable_t4x2": "true", "enable_internet": "true",
        "dataset_sources": ["pikanigga/vlanav-src"],
        "kernel_sources": ["pikanigga/vlanav-03-tuning"],
        "competition_sources": []}
with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    print("wrote", f.name)
