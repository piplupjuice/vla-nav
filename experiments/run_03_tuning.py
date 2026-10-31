"""03_tuning: the exact code that ran on Kaggle.

Extracted from the pushed notebook
(kaggle_notebooks/03_tuning/03_tuning.ipynb); markdown cells omitted, cell
boundaries marked. Cell magics are kept as comments so the file
stays plain Python."""


# ---- cell 2 ----------------------------------------
# %%time
import json, os, platform, subprocess, time

os.makedirs("/kaggle/tmp", exist_ok=True)
os.makedirs("/kaggle/working/results/tuning", exist_ok=True)
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
    if must:
        assert r.returncode == 0, f"FAILED: {s}"
    return r.returncode

cmd("pip install -q 'timm==0.9.16' 'peft==0.11.1' 'draccus==0.8.0' 'utm' 'efficientnet_pytorch' 'ai2thor'")
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
if needle not in src:
    raise RuntimeError("unexpected tokenizers spec - refusing blind patch")
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

# v3 evidence (2026-10-30): a "successful" download can still leave the two .pt
# heads missing, and the run then dies minutes later inside the load cell.
# Verify sizes here and allow exactly one forced retry.
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
                                      for f in NEED if f.endswith((".pt", ".json"))})

# ---- cell 3 ----------------------------------------
import sys, os
# path-agnostic resolver + ambiguity refusal (v1/v2/v4 evidence)
cands = []
for root, dirs, _files in os.walk("/kaggle/input"):
    if "vlanav" in dirs and os.path.isfile(os.path.join(root, "vlanav", "__init__.py")):
        cands.append(root)
assert cands, "vlanav package not found under /kaggle/input"
assert len(cands) == 1, f"ambiguous mount: {cands}"
SRC_ROOT = cands[0]
sys.path.insert(0, SRC_ROOT)
print("SRC_ROOT =", SRC_ROOT)
_stamp = os.path.join(os.path.dirname(SRC_ROOT), "BUILD.txt")
_stamp_txt = open(_stamp).read() if os.path.isfile(_stamp) else ""
print("BUILD:", _stamp_txt or "(no stamp)")
# stale-dataset guard (2026-10-30): a mount without the actuation fixes makes
# MoveAhead(0.1 m) a silent no-op and every episode stalls (nb07 evidence).
for _need in ("contains_RotateLeft=True", "contains_D_recovery_yaw=True"):
    assert _need in _stamp_txt, f"STALE DATASET: BUILD.txt lacks {_need}"

from vlanav.model.omnivla_adapter import OmnivlaNavPolicy
from vlanav.environment.ai2thor_env import Ai2ThorNavEnv
from vlanav.reliability.replanning import ReliabilityTriggered
from vlanav.environment.types import EpisodeSpec

policy = OmnivlaNavPolicy(VLA_PATH,
                          prismatic_repo="/kaggle/tmp/OmniVLA", load_now=False)
# pre-load re-verification: /kaggle/tmp is shared with any other running
# kernel, so a file present in the setup cell can be gone by the load cell.
_vanish = [f for f in NEED
           if not (os.path.isfile(os.path.join(VLA_PATH, f))
                   and os.path.getsize(os.path.join(VLA_PATH, f)) > 1000)]
assert not _vanish, f"checkpoint files missing before load: {_vanish}"
policy.load()
print("policy loaded:", json.dumps(policy.provenance, default=str))

# ---- cell 4 ----------------------------------------
# %%time
import subprocess, time
os.environ["DISPLAY"] = ":0"
subprocess.run("pkill Xvfb || true", shell=True)
subprocess.Popen(["/usr/bin/Xvfb", ":0", "-screen", "0", "1024x768x24", "-ac",
                  "+extension", "GLX", "+render", "-noreset"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 start_new_session=True)
time.sleep(4)

from ai2thor.controller import Controller
controller = Controller(scene="FloorPlan5", width=400, height=400,
                        platform="Linux64", server_type="Linux64", gridSize=0.05)
controller.reset(scene="FloorPlan5")
print("controller OK (linux64+Xvfb)")

# ---- cell 5 ----------------------------------------
# %%time
# 3-tick smoke episode: v5 evidence says a bad THOR action argument otherwise
# surfaces only after the 15 GB model load + controller build.
env_smoke = Ai2ThorNavEnv(controller)
spec_smoke = EpisodeSpec(episode_id="smoke-000", condition="A_in_dist", split="val",
                         seed=991, world_id=991, start_xy=(0.0, 0.0),
                         start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                         instruction="(set at reset)", max_steps=3,
                         goal_radius_m=0.6, extra={"smoke": True})
_smoke = ReliabilityTriggered(policy, threshold_m=0.05, stall_limit=2,
                              max_replans_per_episode=4).run_episode(
    env_smoke, spec_smoke)
print("SMOKE OK | steps", _smoke.step, "| scene", spec_smoke.extra.get("scene"))
assert _smoke.step >= 1

# ---- cell 6 ----------------------------------------
# %%time
import math, statistics
import pandas as pd
from vlanav.evaluation.logging import EpisodeLogger, ExperimentMeta

SEEDS = [20001 + 111 * i for i in range(6)]   # val tuning seeds (disjoint from pilot)
MAX_STEPS = 60                                # same budget as the test matrix

def make_spec(i):
    seed = SEEDS[i]
    return EpisodeSpec(
        episode_id=f"A_in_dist/val/v1/{i:03d}",
        condition="A_in_dist", split="val", seed=seed, world_id=100 + i,
        start_xy=(0.0, 0.0), start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
        instruction="(set at reset from THOR target)", max_steps=MAX_STEPS,
        goal_radius_m=0.6, extra={"tuning": True, "benchmark_version": "v1"})

env = Ai2ThorNavEnv(controller)
THRESHOLDS = [0.05, 0.10, 0.20, 0.40]
rows, all_rows = [], []
for thr in THRESHOLDS:
    exp_id = f"tune-thor-20261030-thr{thr:.2f}"
    meta = ExperimentMeta(
        experiment_id=exp_id, method="reliability_triggered",
        env_family="ai2thor", model_name="NHirose/omnivla-original",
        model_revision="omnivla-original@120000",
        gpu_name=";".join(torch.cuda.get_device_properties(i).name
                          for i in range(torch.cuda.device_count())),
        torch_version=torch.__version__, conditions=["A_in_dist"], split="val",
        action_budget=MAX_STEPS,
        notes=("VAL-ONLY threshold sweep; every row is logged so the table "
               "recomputes from episodes.jsonl; test split untouched."),
        extra={"threshold_m": thr, "stall_limit": 2, "seeds": SEEDS,
               "threshold_source": "this sweep (val only)"})
    logger = EpisodeLogger(meta, f"/kaggle/working/results/raw/{exp_id}")
    regime = ReliabilityTriggered(policy, threshold_m=thr, stall_limit=2,
                                  max_replans_per_episode=24)
    succ, infs, steps, dists = [], [], [], []
    for i in range(len(SEEDS)):
        spec = make_spec(i)
        res = regime.run_episode(env, spec)
        logger.log_episode(res)
        all_rows.append(res.__dict__)
        succ.append(res.success); infs.append(res.n_inferences)
        steps.append(res.step); dists.append(res.distance_to_goal_m)
        print(json.dumps({"thr": thr, "ep": i, "success": res.success,
                          "inf": res.n_inferences,
                          "d_final": round(res.distance_to_goal_m, 2)}))
    row = {"threshold_m": thr, "n": len(succ),
           "SR": round(sum(succ) / len(succ), 3),
           "inferences_mean": round(statistics.mean(infs), 1),
           "steps_mean": round(statistics.mean(steps), 1),
           "dist_final_mean": round(statistics.mean(dists), 2)}
    rows.append(row)
    print(row)

pd.DataFrame(all_rows).to_json("/kaggle/working/results/tuning_rows.json",
                               orient="records", indent=1)

# selection: highest SR, then fewest inferences
best = sorted(rows, key=lambda r: (-r["SR"], r["inferences_mean"]))[0]
choice = {"threshold_m": best["threshold_m"], "stall_limit": 2,
          "selection_rule": "max SR, tie -> min mean inferences (val only)",
          "table": rows,
          "seeds": SEEDS, "n_per_threshold": len(SEEDS),
          "val_scenes": "FloorPlan1..30 (env picks by seed)",
          "note": "test split untouched; this file is the ONLY threshold source"}
with open("/kaggle/working/results/tuning/threshold_choice.json", "w") as f:
    json.dump(choice, f, indent=2)
print("CHOICE:", json.dumps(choice["threshold_m"], indent=2))
print("saved threshold_choice.json")
