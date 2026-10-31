"""08_phase_a: the exact code that ran on Kaggle.

Extracted from the pushed notebook
(kaggle_notebooks/08_phase_a/08_phase_a.ipynb); markdown cells omitted, cell
boundaries marked. Cell magics are kept as comments so the file
stays plain Python."""


# ---- cell 2 ----------------------------------------
# %%time
import json, os, platform, subprocess, time

os.makedirs("/kaggle/tmp", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)
os.environ["AI2THOR_DOWNLOAD_DIR"] = "/kaggle/tmp/aa"
os.environ["OBJECT_CACHE_DIR"] = "/kaggle/tmp/thor_objcache"
os.environ["XDG_CACHE_HOME"] = "/kaggle/tmp/xdgcache"
report = {"notebook": "08_phase_a", "stage": "kernel-env",
          "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "python": platform.python_version()}
import torch
report["torch"] = torch.__version__
report["gpus"] = [torch.cuda.get_device_properties(i).name
                  for i in range(torch.cuda.device_count())]
print(json.dumps(report, indent=2))
json.dump(report, open("/kaggle/working/phase_a_env.json", "w"), indent=2)

def cmd(s, must=True):
    print("$", s)
    r = subprocess.run(s, shell=True, capture_output=True, text=True)
    print(r.stdout[-1000:]); print(r.stderr[-1000:])
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
    print("vulkan ICD ok")
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
    raise RuntimeError("unexpected tokenizers spec - refusing blind patch")
open(_tbl, "w", encoding="utf-8").write(
    src.replace(needle, '"tokenizers": "tokenizers>=0.19"'))
print("tokenizers spec patched (exact needle)")

import sys
sys.path.insert(0, "/kaggle/tmp/OmniVLA")
from huggingface_hub import snapshot_download
import uuid as _uuid
_RUN = ("08phasea-" + _uuid.uuid4().hex[:6])
VLA_PATH = "/kaggle/tmp/omnivla-" + _RUN
CACHE_DIR = "/kaggle/tmp/hfcache-" + _RUN
print("VLA_PATH =", VLA_PATH)
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
                      cache_dir=CACHE_DIR,
                      force_download=(_attempt == 2))
    _miss = _missing(VLA_PATH)
    print("download attempt", _attempt, "| missing:", _miss)
    if not _miss:
        break
if _miss:
    raise RuntimeError(f"checkpoint incomplete after retry: {_miss}")
print("setup OK")

# ---- cell 3 ----------------------------------------
import sys, os, json
cands = []
for root, dirs, _files in os.walk("/kaggle/input"):
    if "vlanav" in dirs and os.path.isfile(os.path.join(root, "vlanav", "__init__.py")):
        cands.append(root)
if not cands:
    raise RuntimeError("vlanav package not found; /kaggle/input: "
                       + repr(os.listdir("/kaggle/input")))
if len(cands) > 1:
    raise RuntimeError(f"ambiguous mount: {cands}")
SRC_ROOT = cands[0]
sys.path.insert(0, SRC_ROOT)
print("SRC_ROOT =", SRC_ROOT)
_stamp = os.path.join(os.path.dirname(SRC_ROOT), "BUILD.txt")
_stamp_txt = open(_stamp).read() if os.path.isfile(_stamp) else ""
print("BUILD:", _stamp_txt or "(no stamp)")
for _need in ("contains_RotateLeft=True", "contains_D_recovery_yaw=True"):
    assert _need in _stamp_txt, f"STALE DATASET: BUILD.txt lacks {_need}"
# RAAR must be present in the mounted package (new dataset version guard)
import vlanav.reliability.raar as _raar
assert hasattr(_raar, "RAARTriggered"), "stale dataset: RAAR module missing"
print("RAAR module OK")

from vlanav.model.omnivla_adapter import OmnivlaNavPolicy
from vlanav.environment.ai2thor_env import Ai2ThorNavEnv
from vlanav.reliability.replanning import (ChunkedFrozen, FixedFrequencyReplan)
from vlanav.reliability.raar import RAARTriggered, RAARConfig
from vlanav.environment.types import EpisodeSpec
from vlanav.evaluation.logging import EpisodeLogger, ExperimentMeta

policy = OmnivlaNavPolicy(VLA_PATH,
                          prismatic_repo="/kaggle/tmp/OmniVLA", load_now=False)
_vanish = [f for f in NEED
           if not (os.path.isfile(os.path.join(VLA_PATH, f))
                   and os.path.getsize(os.path.join(VLA_PATH, f)) > 1000)]
assert not _vanish, f"checkpoint files missing before load: {_vanish}"
policy.load()
assert policy.is_model, "refusing to run Phase A with a mock policy"
print("POLICY LOADED:", json.dumps(policy.provenance, default=str))

# ---- cell 4 ----------------------------------------
# %%time
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
print("CONTROLLER OK | frame", ev.frame.shape, "| val scene rendered")

# ---- cell 5 ----------------------------------------
# %%time
# 3-tick smoke on a VAL scene (movement gate; nb07 evidence)
env_smoke = Ai2ThorNavEnv(controller)
spec_smoke = EpisodeSpec(episode_id="smoke-000", condition="A_in_dist", split="val",
                         seed=991, world_id=991, start_xy=(0.0, 0.0),
                         start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                         instruction="(set at reset)", max_steps=3,
                         goal_radius_m=0.6, extra={"smoke": True})
_smoke = ChunkedFrozen(policy, "chunked_frozen").run_episode(env_smoke, spec_smoke)
print("SMOKE OK | steps", _smoke.step, "| inf", _smoke.n_inferences,
      "| scene", spec_smoke.extra.get("scene"),
      "| moved", round(_smoke.path_length_m, 3), "m")
assert _smoke.step >= 1
assert _smoke.path_length_m > 0.02, (
    "SMOKE FAILED: robot moved %.3f m in 3 ticks -> actuation broken "
    "" % _smoke.path_length_m)

# ---- cell 6 ----------------------------------------
# %%time
# ---- easy-episode selection (deterministic; selection is logged) --------
# Candidate seeds 500..599 on the VAL A_in_dist block; keep the first 20
# whose STRAIGHT-LINE start->goal distance is < 2.5 m. reset() is metadata
# only (no VLA call), so scanning is cheap. Selection provenance goes into
# meta.extra so the set is reproducible.
import math
env_scan = Ai2ThorNavEnv(controller)
EASY_M, WANT = 2.5, 20
easy_specs = []
scanned = 0
for i in range(500, 600):
    if len(easy_specs) >= WANT or scanned >= 60:
        break
    spec = EpisodeSpec(episode_id=f"phaseA/val/pA/{i}",
                       condition="A_in_dist", split="val", seed=i,
                       world_id=i, start_xy=(0.0, 0.0), start_yaw_rad=0.0,
                       goal_xy=(0.0, 0.0), instruction="(set at reset)",
                       max_steps=60, goal_radius_m=0.6,
                       extra={"benchmark_version": "phaseA"})
    env_scan.reset(spec)
    d = math.hypot(spec.goal_xy[0] - spec.start_xy[0],
                   spec.goal_xy[1] - spec.start_xy[1])
    scanned += 1
    if d < EASY_M:
        easy_specs.append((spec, round(d, 2)))
print("scanned", scanned, "candidates ->", len(easy_specs), "easy episodes")
assert len(easy_specs) >= 12, "too few easy episodes found; raise seed range"
SEEDS = [s.seed for s, _ in easy_specs]
print("easy seeds:", SEEDS)

# ---- cell 7 ----------------------------------------
# %%time
# ---- run B2 + RAAR on the same easy episodes (paired) -------------------
import time
import pandas as pd

MAX_STEPS = 60
TAG = "phasea-thor-" + time.strftime("%Y%m%d")
env = Ai2ThorNavEnv(controller)

RAAR_CFG = RAARConfig()   # documented defaults (plan §2.1); NOT tuned here

def make_regimes():
    return [("fixed_frequency", FixedFrequencyReplan(policy, "fixed_frequency")),
            ("raar_triggered", RAARTriggered(policy, "raar_triggered",
                                             config=RAAR_CFG))]

rows = []
t_start = time.time()
for mname, regime in make_regimes():
    exp_id = f"{TAG}-{mname}"
    out_dir = f"/kaggle/working/results/raw/{exp_id}"
    done, prev = set(), []
    ep_path = os.path.join(out_dir, "episodes.jsonl")
    if os.path.isfile(ep_path):
        for line in open(ep_path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add(r["episode_id"]); prev.append(r)
        print(f"resume {exp_id}: {len(done)} rows already on disk")
    meta = ExperimentMeta(
        experiment_id=exp_id, method=regime.name, env_family="ai2thor",
        model_name="NHirose/omnivla-original",
        model_revision="omnivla-original@120000",
        gpu_name=";".join(torch.cuda.get_device_properties(i).name
                          for i in range(torch.cuda.device_count())),
        torch_version=torch.__version__,
        conditions=["A_in_dist"], split="val", action_budget=MAX_STEPS,
        notes=("Phase A smoke: 20 EASY val episodes (straight-line < 2.5 m), "
               "DIAGNOSTIC ONLY, never test evidence. RAAR config = "
               "documented defaults, not tuned. GO/NO-GO gate reads the "
               "B2 rows."),
        extra={"benchmark_version": "phaseA", "easy_threshold_m": 2.5,
               "seeds": SEEDS,
               "easy_dist_m": [d for _, d in easy_specs],
               "raar_config": RAAR_CFG.__dict__,
               "scanned_candidates": scanned,
               "goal_radius_m": 0.6})
    logger = EpisodeLogger(meta, out_dir, overwrite=True)
    for r in prev:
        logger.log_episode(r)
    print("=== ", exp_id, " ===")
    for spec, d in easy_specs:
        if spec.episode_id in done:
            continue
        res = regime.run_episode(env, spec)
        row = dict(res.__dict__)
        row["instruction"] = spec.instruction
        row["scene"] = spec.extra.get("scene")
        row["easy_dist_m"] = d
        logger.log_episode(row)
        rows.append(row)
        print(json.dumps({"exp": exp_id, "ep": res.episode_id,
                          "succ": res.success, "steps": res.step,
                          "d_final": round(res.distance_to_goal_m, 2),
                          "inf": res.n_inferences,
                          "replans": res.n_replans_triggered,
                          "stall_onsets": res.n_stall_onsets,
                          "recoveries": res.n_recoveries,
                          "spl_std": (round(res.spl_standard, 3)
                                      if res.spl_standard == res.spl_standard else None),
                          "elapsed_min": round((time.time() - t_start) / 60, 1)}))

controller.stop()

# ---- summary + pre-registered GO/NO-GO gate -----------------------------
all_rows = []
for p in sorted(__import__("glob").glob("/kaggle/working/results/raw/phasea-*/episodes.jsonl")):
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            all_rows.append(json.loads(line))
df = pd.DataFrame(all_rows)
summary = (df.groupby("method")
             .agg(SR=("success", "mean"), n=("success", "size"),
                  inference_mean=("n_inferences", "mean"),
                  replans_mean=("n_replans_triggered", "mean"),
                  recoveries_mean=("n_recoveries", "mean"),
                  steps_mean=("step", "mean"),
                  dist_final=("distance_to_goal_m", "mean"),
                  lat_mean_s=("infer_latency_s_mean", "mean"),
                  vla_s=("vla_latency_seconds", "mean"),
                  sim_s=("simulator_time_s", "mean"),
                  ctrl_s=("controller_time_s", "mean"))
             .round(3))
print(summary.to_string())
summary.to_json("/kaggle/working/results/phase_a_summary.json", orient="index")

b2 = df[(df.method == "fixed_frequency")]
sr = float(b2.success.mean()) if len(b2) else 0.0
gate = {"b2_easy_success_rate": round(sr, 3), "n": int(len(b2)),
        "gate": "GO" if sr >= 0.5 else "NO-GO",
        "rule": "B2 SR >= 0.5 on the 20 easy val episodes (pre-registered)"}
print(json.dumps(gate, indent=2))
json.dump(gate, open("/kaggle/working/results/phase_a_gate.json", "w"), indent=2)
