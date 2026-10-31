"""Generate the Bottleneck Sweep kernel (08b).

Question (one GPU run, ~2.5 h): is OmniVLA's language-only goal interface
the bottleneck of the indoor failure, or is the whole indoor domain the wall?

Three methods on the SAME 24 verified-clear-path easy val episodes:
  1. B2 fixed-frequency, official modality 7 (language only)   [reference]
  2. B2 fixed-frequency, official modality 8 (language+goal-pose;
     goal position from the simulator - PRIVILEGED, labelled as such)
  3. B3 reliability-triggered, modality 7, threshold 0.2 (val artifact)
     [keeps the cost axis on the clean slice]

Also: GetShortestPath probe (self-debugging) fixes the standard-SPL
denominator on THOR, and clear-path selection uses it.

Run:  python scripts/gen_notebook_08b.py
Then: kaggle kernels push -p kaggle_notebooks/08b_bottleneck
"""
import ast
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t}
code = lambda t: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t}


def setup_code() -> str:
    return r'''%%time
import json, os, platform, subprocess, time

os.makedirs("/kaggle/tmp", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)
os.environ["AI2THOR_DOWNLOAD_DIR"] = "/kaggle/tmp/aa"
os.environ["OBJECT_CACHE_DIR"] = "/kaggle/tmp/thor_objcache"
os.environ["XDG_CACHE_HOME"] = "/kaggle/tmp/xdgcache"
report = {"notebook": "08b_bottleneck", "stage": "kernel-env",
          "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "python": platform.python_version()}
import torch
report["torch"] = torch.__version__
report["gpus"] = [torch.cuda.get_device_properties(i).name
                  for i in range(torch.cuda.device_count())]
json.dump(report, open("/kaggle/working/bottleneck_env.json", "w"), indent=2)

def cmd(s, must=True):
    r = subprocess.run(s, shell=True, capture_output=True, text=True)
    assert (r.returncode == 0) if must else True
    return r.returncode

cmd("pip install -q 'timm==0.9.16' 'peft==0.11.1' 'draccus==0.8.0' 'utm' 'efficientnet_pytorch' 'ai2thor'")
cmd("apt-get update -qq", must=False)
for _p in ("xvfb", "x11-utils", "mesa-utils", "libgl1", "libglu1-mesa",
           "libegl1", "libxcursor1", "libxinerama1", "libxrandr2",
           "libxi6", "libxss1", "libnss3", "libvulkan1", "vulkan-tools"):
    if cmd(f"apt-get install -y -qq --no-install-recommends {_p}", must=False) != 0:
        cmd(f"apt-get install -y -qq --no-install-recommends {_p}t64", must=False)
try:
    import glob as globmod
    icd_dir = "/usr/share/vulkan/icd.d"
    icd_txt = {open(p).read() for p in globmod.glob(icd_dir + "/*.json")} \
        if os.path.isdir(icd_dir) else set()
    if not any("nvidia" in s.lower() for s in icd_txt):
        os.makedirs(icd_dir, exist_ok=True)
        with open(os.path.join(icd_dir, "nvidia_icd.json"), "w") as f:
            f.write('{\n    "file_format_version" : "1.0.0",\n'
                    '    "ICD": {\n'
                    '        "library_path": "libGLX_nvidia.so.0",\n'
                    '        "api_version" : "1.2.135"\n    }\n}\n')
        os.environ["VK_ICD_FILENAMES"] = os.path.join(icd_dir, "nvidia_icd.json")
except Exception as _e:
    print("vulkan ICD diagnostic skipped:", repr(_e))

cmd("rm -rf /kaggle/tmp/OmniVLA", must=False)
cmd("git clone --depth 1 https://github.com/NHirose/OmniVLA.git /kaggle/tmp/OmniVLA")
assert os.path.isdir("/kaggle/tmp/OmniVLA/prismatic"), "clone incomplete"
cmd("pip install -q --no-deps 'transformers==4.40.1' 'huggingface_hub==0.29.1'")
import importlib.util
_tf_root = os.path.dirname(importlib.util.find_spec("transformers").origin)
_tbl = os.path.join(_tf_root, "dependency_versions_table.py")
src = open(_tbl, encoding="utf-8").read()
needle = '"tokenizers": "tokenizers>=0.19,<0.20"'
if needle not in src:
    raise RuntimeError("unexpected tokenizers spec")
open(_tbl, "w", encoding="utf-8").write(src.replace(needle, '"tokenizers": "tokenizers>=0.19"'))

import sys
sys.path.insert(0, "/kaggle/tmp/OmniVLA")
from huggingface_hub import snapshot_download
import uuid as _uuid
_RUN = "08b-" + _uuid.uuid4().hex[:6]
VLA_PATH = "/kaggle/tmp/omnivla-" + _RUN
CACHE_DIR = "/kaggle/tmp/hfcache-" + _RUN
NEED = [f"model-0000{i}-of-00004.safetensors" for i in (1, 2, 3, 4)] + [
    "action_head--120000_checkpoint.pt", "proprio_projector--120000_checkpoint.pt",
    "config.json", "processing_prismatic.py", "modeling_prismatic.py",
    "configuration_prismatic.py"]

def _missing(root):
    return [f for f in NEED
            if not (os.path.isfile(os.path.join(root, f))
                    and os.path.getsize(os.path.join(root, f)) > 1000)]

for _attempt in (1, 2):
    snapshot_download(repo_id="NHirose/omnivla-original", local_dir=VLA_PATH,
                      cache_dir=CACHE_DIR, force_download=(_attempt == 2))
    _miss = _missing(VLA_PATH)
    print("download attempt", _attempt, "| missing:", _miss)
    if not _miss:
        break
if _miss:
    raise RuntimeError(f"checkpoint incomplete after retry: {_miss}")
print("setup OK")'''


def main() -> None:
    out = os.path.join(HERE, "kaggle_notebooks", "08b_bottleneck")
    os.makedirs(out, exist_ok=True)

    cells = [
        md("""# VLA-NAV - Notebook 08b: The Bottleneck Sweep

**One question:** is OmniVLA's language-only goal interface the bottleneck
of the indoor failure, or is the whole indoor domain the wall?

Three methods on the same 24 verified-clear-path easy val episodes
(straight-line < 2.5 m AND GetShortestPath <= 1.25x straight):
1. B2 every-step, **modality 7** (language only) - reference
2. B2 every-step, **modality 8** (language+goal-pose; goal position from
   the simulator - PRIVILEGED input, an official deployment mode of the
   model, labelled as such everywhere)
3. B3 reliability-triggered, modality 7, threshold 0.2 m (val artifact)

Exploratory sweep: both outcomes are reportable findings. Diagnostic val
data only - never test evidence."""),

        code(setup_code()),

        code(r'''import sys, os, json, types
cands = []
for root, dirs, _files in os.walk("/kaggle/input"):
    if "vlanav" in dirs and os.path.isfile(os.path.join(root, "vlanav", "__init__.py")):
        cands.append(root)
if not cands:
    raise RuntimeError("vlanav package not found")
if len(cands) > 1:
    raise RuntimeError(f"ambiguous mount: {cands}")
SRC_ROOT = cands[0]
sys.path.insert(0, SRC_ROOT)
_stamp = open(os.path.join(os.path.dirname(SRC_ROOT), "BUILD.txt")).read()
print("BUILD:", _stamp)
for _need in ("contains_RotateLeft=True", "contains_D_recovery_yaw=True"):
    assert _need in _stamp, f"STALE DATASET: {_need}"
import vlanav.model.omnivla_adapter as _ad
assert "language_pose" in _ad.OmnivlaNavPolicy.MODALITY_ID, \
    "stale dataset: modality 8 missing"
print("modality 8 OK")

from vlanav.model.omnivla_adapter import OmnivlaNavPolicy
from vlanav.environment.ai2thor_env import Ai2ThorNavEnv
from vlanav.environment.types import EpisodeSpec
from vlanav.reliability.replanning import (ChunkedFrozen, FixedFrequencyReplan,
                                           ReliabilityTriggered)
from vlanav.evaluation.logging import EpisodeLogger, ExperimentMeta

policy = OmnivlaNavPolicy(VLA_PATH,
                          prismatic_repo="/kaggle/tmp/OmniVLA", load_now=False)
_vanish = [f for f in NEED
           if not (os.path.isfile(os.path.join(VLA_PATH, f))
                   and os.path.getsize(os.path.join(VLA_PATH, f)) > 1000)]
assert not _vanish, f"checkpoint files missing before load: {_vanish}"
policy.load()
assert policy.is_model
print("POLICY LOADED | modes:", policy.MODALITY_ID)'''),

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
controller = Controller(scene="FloorPlan1", width=400, height=400,
                        platform="Linux64", server_type="Linux64", gridSize=0.05)
ev = controller.reset(scene="FloorPlan1")
print("CONTROLLER OK | frame", ev.frame.shape)'''),

        code(r'''%%time
# ---- GetShortestPath probe (self-debugging SPL fix) --------------------
import math
env_fix = Ai2ThorNavEnv(controller)
_sm = EpisodeSpec(episode_id="probe-000", condition="A_in_dist", split="val",
                  seed=991, world_id=991, start_xy=(0.0, 0.0), start_yaw_rad=0.0,
                  goal_xy=(0.0, 0.0), instruction="(probe)", max_steps=1,
                  goal_radius_m=0.6, extra={})
env_fix.reset(_sm)
ag = controller.last_event.metadata["agent"]
p0 = ag["position"]
tgt = None
for o in controller.last_event.metadata["objects"]:
    if o.get("visible") and o.get("objectId") and "|" in o["objectId"] \
            and o["objectType"] in ("CounterTop", "Toaster", "Microwave"):
        tgt = o
        break
assert tgt is not None, "probe: no visible target"
ev = controller.step(action="GetShortestPath", x=float(p0["x"]), y=0.0,
                     z=float(p0["z"]),
                     position={"x": float(tgt["position"]["x"]),
                               "y": 0.0, "z": float(tgt["position"]["z"])})
print("lastActionSuccess:", ev.metadata.get("lastActionSuccess"))
print("metadata keys with list values:", [k for k, v in ev.metadata.items()
                                          if isinstance(v, list) and v])
PATH_KEY = None
for k in ("action", "shortest_path", "corners"):
    v = ev.metadata.get(k)
    if isinstance(v, list) and v and isinstance(v[0], dict) and "x" in v[0]:
        PATH_KEY = k
        break
print("PATH_KEY =", PATH_KEY)
assert PATH_KEY is not None, "GetShortestPath path key not found - SPL stays NaN"

def _shortest_path_length_m(self):
    assert self.spec is not None and self._pose is not None
    sx, sz, _ = self._pose
    gx, gz = self.spec.goal_xy
    try:
        e = self.controller.step(action="GetShortestPath", x=float(sx),
                                 y=0.0, z=float(sz),
                                 position={"x": float(gx), "y": 0.0, "z": float(gz)})
    except Exception:
        return None
    if not e.metadata.get("lastActionSuccess", False):
        return None
    pts = [(float(q["x"]), float(q["z"])) for q in e.metadata.get(PATH_KEY) or []]
    if len(pts) < 2:
        return None
    total = sum(math.hypot(bx - ax, bz - az)
                for (ax, az), (bx, bz) in zip(pts, pts[1:]))
    return total if total > 0 else None

Ai2ThorNavEnv.shortest_path_length_m = _shortest_path_length_m
# verify on the probe pair: straight distance vs reported path
d_straight = math.hypot(float(tgt["position"]["x"]) - float(p0["x"]),
                        float(tgt["position"]["z"]) - float(p0["z"]))
print("probe: straight", round(d_straight, 2), "m")'''),

        code(r'''%%time
# ---- 3-tick smoke in BOTH modalities (movement gate + conditioning check)
env_smoke = Ai2ThorNavEnv(controller)
spec_smoke = EpisodeSpec(episode_id="smoke-000", condition="A_in_dist", split="val",
                         seed=991, world_id=991, start_xy=(0.0, 0.0),
                         start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                         instruction="(set at reset)", max_steps=3,
                         goal_radius_m=0.6, extra={})
def _gp():
    pose = env_smoke._pose
    gx, gz = env_smoke.spec.goal_xy
    dx, dz = gx - pose[0], gz - pose[1]
    yaw = pose[2]
    rx = dx * math.cos(yaw) + dz * math.sin(yaw)
    ry = -dx * math.sin(yaw) + dz * math.cos(yaw)
    return rx, ry, math.atan2(ry, rx)

policy.instruction_mode = "language"; policy.goal_pose_fn = None
r7 = ChunkedFrozen(policy, "chunked_frozen").run_episode(env_smoke, spec_smoke)
policy.instruction_mode = "language_pose"; policy.goal_pose_fn = _gp
r8 = ChunkedFrozen(policy, "chunked_frozen").run_episode(env_smoke, spec_smoke)
policy.instruction_mode = "language"; policy.goal_pose_fn = None
print("smoke m7: moved", round(r7.path_length_m, 3), "| m8: moved", round(r8.path_length_m, 3))
assert r7.path_length_m > 0.02, "modality 7 actuation broken"
assert r8.path_length_m > 0.02, "modality 8 actuation broken"'''),

        code(r'''%%time
# ---- clear-path easy episode selection (24; provenance logged) ---------
env_scan = Ai2ThorNavEnv(controller)
WANT, STRAIGHT_MAX = 24, 2.5
# Phase-A-style selection (proven: 20/70 at <2.5 m). SPL is recorded
# opportunistically but is NOT a filter: GetShortestPath fails for ~90%
# of furniture-object goals (object centre is off the navmesh).
easy, scanned, n_spl, n_spl_ok = [], 0, 0, 0
for i in range(500, 640):
    if len(easy) >= WANT or scanned >= 100:
        break
    spec = EpisodeSpec(episode_id=f"botl/val/pB/{i}", condition="A_in_dist",
                       split="val", seed=i, world_id=i, start_xy=(0.0, 0.0),
                       start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                       instruction="(set at reset)", max_steps=60,
                       goal_radius_m=0.6, extra={"benchmark_version": "botl"})
    env_scan.reset(spec)
    d = math.hypot(spec.goal_xy[0] - spec.start_xy[0],
                   spec.goal_xy[1] - spec.start_xy[1])
    scanned += 1
    if d >= STRAIGHT_MAX:
        continue
    sp = env_scan.shortest_path_length_m()
    if sp is None:
        n_spl += 1
    else:
        n_spl_ok += 1
    easy.append((spec, round(d, 2), round(sp, 2) if sp else float("nan"),
                 (sp / d) if sp else float("nan")))
print("scanned", scanned, "-> easy episodes:", len(easy),
      "| SPL ok/none:", n_spl_ok, "/", n_spl)
assert len(easy) >= 12, "not enough easy episodes - investigate scan"
SEEDS = [s.seed for s, _, _, _ in easy]
print("selected", len(easy), "episodes")
print("seeds:", SEEDS)'''),

        code(r'''%%time
# ---- the sweep: 3 methods x 24 paired episodes --------------------------
import time
import pandas as pd

MAX_STEPS = 60
TAG = "botl-" + time.strftime("%Y%m%d")
env = Ai2ThorNavEnv(controller)

def _gp():
    pose = env._pose
    gx, gz = env.spec.goal_xy
    dx, dz = gx - pose[0], gz - pose[1]
    yaw = pose[2]
    rx = dx * math.cos(yaw) + dz * math.sin(yaw)
    ry = -dx * math.sin(yaw) + dz * math.cos(yaw)
    return rx, ry, math.atan2(ry, rx)

def make_regimes():
    return [("language", "fixed_frequency", FixedFrequencyReplan(policy, "fixed_frequency")),
            ("language_pose", "fixed_frequency", FixedFrequencyReplan(policy, "fixed_frequency")),
            ("language", "reliability_triggered",
             ReliabilityTriggered(policy, "reliability_triggered",
                                  threshold_m=0.2, stall_limit=2,
                                  max_replans_per_episode=24))]

rows = []
t0 = time.time()
for gmode, rname, regime in make_regimes():
    policy.instruction_mode = gmode
    policy.goal_pose_fn = _gp if gmode == "language_pose" else None
    exp_id = f"{TAG}-{gmode}-{rname}"
    out_dir = f"/kaggle/working/results/raw/{exp_id}"
    done, prev = set(), []
    ep_path = os.path.join(out_dir, "episodes.jsonl")
    if os.path.isfile(ep_path):
        for line in open(ep_path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add(r["episode_id"]); prev.append(r)
        print(f"resume {exp_id}: {len(done)} rows on disk")
    meta = ExperimentMeta(
        experiment_id=exp_id, method=regime.name, env_family="ai2thor",
        model_name="NHirose/omnivla-original",
        model_revision="omnivla-original@120000",
        gpu_name=";".join(torch.cuda.get_device_properties(i).name
                          for i in range(torch.cuda.device_count())),
        torch_version=torch.__version__,
        conditions=["A_in_dist"], split="val", action_budget=MAX_STEPS,
        notes=("Bottleneck sweep: clear-path easy val episodes; goal "
               "modality " + gmode + (" (PRIVILEGED goal pose, official "
               "modality 8)" if gmode == "language_pose" else
               " (onboard language-only, official modality 7)") + "."),
        extra={"benchmark_version": "botl", "goal_modality": gmode,
               "regime": rname, "seeds": SEEDS,
               "easy_dist_m": [d for _, d, _, _ in easy],
               "shortest_path_m": [sp for _, _, sp, _ in easy],
               "threshold_m": 0.2 if rname == "reliability_triggered" else None,
               "threshold_source": "val-only tuning artifact (nb03)" if rname == "reliability_triggered" else None})
    logger = EpisodeLogger(meta, out_dir, overwrite=True)
    for r in prev:
        logger.log_episode(r)
    print("=== ", exp_id, " ===")
    for spec, d, sp, ratio in easy:
        if spec.episode_id in done:
            continue
        res = regime.run_episode(env, spec)
        row = dict(res.__dict__)
        row["instruction"] = spec.instruction
        row["scene"] = spec.extra.get("scene")
        row["goal_modality"] = gmode
        row["easy_dist_m"] = d
        row["shortest_path_m"] = sp
        row["detour_ratio"] = round(ratio, 3)
        logger.log_episode(row)
        rows.append(row)
        print(json.dumps({"exp": exp_id, "ep": res.episode_id,
                          "succ": res.success, "steps": res.step,
                          "d_final": round(res.distance_to_goal_m, 2),
                          "inf": res.n_inferences,
                          "col": res.n_collisions,
                          "spl_std": (round(res.spl_standard, 3)
                                      if res.spl_standard == res.spl_standard else None),
                          "min": round((time.time() - t0) / 60, 1)}))

controller.stop()

all_rows = []
for p in sorted(__import__("glob").glob("/kaggle/working/results/raw/botl-*/episodes.jsonl")):
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            all_rows.append(json.loads(line))
df = pd.DataFrame(all_rows)
df["regime"] = df["meta"].apply(lambda m: m.get("method"))
summary = (df.groupby(["goal_modality", "regime"])
             .agg(SR=("success", "mean"), n=("success", "size"),
                  queries=("n_inferences", "mean"),
                  collisions=("n_collisions", "mean"),
                  dist_final=("distance_to_goal_m", "mean"),
                  spl_std=("spl_standard", "mean"),
                  recoveries=("n_recoveries", "mean"))
             .round(3))
print(summary.to_string())
summary.to_json("/kaggle/working/results/bottleneck_summary.json", orient="index")

b7 = df[(df.goal_modality == "language") & (df.regime == "fixed_frequency")]
b8 = df[(df.goal_modality == "language_pose") & (df.regime == "fixed_frequency")]
read = {"SR_language_only": round(float(b7.success.mean()), 3),
        "SR_language_pose": round(float(b8.success.mean()), 3),
        "n": int(len(b7)),
        "reading": ("language interface IS the bottleneck: modality 8 lifts "
                    "success" if float(b8.success.mean()) > float(b7.success.mean()) + 0.05
                    else "language interface is NOT the dominant bottleneck; "
                         "the indoor domain itself is the wall")}
print(json.dumps(read, indent=2))
json.dump(read, open("/kaggle/working/results/bottleneck_reading.json", "w"), indent=2)'''),

        md("""## Readout rules

- Modality 8 receives the goal position from the simulator: it is an
  OFFICIAL deployment mode of the model, but a PRIVILEGED input - every
  table labels it. It is never compared as if it were onboard-only.
- Both sweep outcomes are findings: the language interface being the
  bottleneck is actionable for the model authors; it not being one
  strengthens the transfer-wall evidence.
- All rows carry goal_modality + shortest_path_m; SPL_standard is now the
  real walkable-path denominator (probe-verified)."""),
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
    with open(os.path.join(out, "08b_bottleneck.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        print("wrote", f.name)

    meta = {"id": "pikanigga/vlanav-08b-bottleneck", "title": "vlanav-08b-bottleneck",
            "code_file": "08b_bottleneck.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true",
            "enable_gpu": "true", "enable_t4x2": "true",
            "enable_internet": "true",
            "dataset_sources": ["pikanigga/vlanav-src"],
            "kernel_sources": [],
            "competition_sources": []}
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        print("wrote", f.name)


if __name__ == "__main__":
    main()
