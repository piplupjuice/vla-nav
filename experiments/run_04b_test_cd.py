"""04b_test_cd: the exact code that ran on Kaggle.

Extracted from the pushed notebook
(kaggle_notebooks/04b_test_cd/04b_test.ipynb); markdown cells omitted, cell
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
report = {"notebook": "04_test_b", "stage": "kernel-env",
          "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "python": platform.python_version()}
import torch
report["torch"] = torch.__version__
report["gpus"] = [torch.cuda.get_device_properties(i).name
                  for i in range(torch.cuda.device_count())]
print(json.dumps(report, indent=2))
json.dump(report, open("/kaggle/working/full_env_b.json", "w"), indent=2)

def cmd(s, must=True):
    print("$", s)
    r = subprocess.run(s, shell=True, capture_output=True, text=True)
    print(r.stdout[-1000:]); print(r.stderr[-1000:])
    assert (r.returncode == 0) if must else True
    return r.returncode

# proven pins (nb00 v7 / nb01 v4 / nb02 evidence)
cmd("pip install -q 'timm==0.9.16' 'peft==0.11.1' 'draccus==0.8.0' 'utm' 'efficientnet_pytorch' 'ai2thor'")
cmd("apt-get update -qq", must=False)
for _p in ("xvfb", "x11-utils", "mesa-utils", "libgl1", "libglu1-mesa",
           "libegl1", "libxcursor1", "libxinerama1", "libxrandr2",
           "libxi6", "libxss1", "libnss3", "libvulkan1", "vulkan-tools"):
    if cmd(f"apt-get install -y -qq --no-install-recommends {_p}", must=False) != 0:
        cmd(f"apt-get install -y -qq --no-install-recommends {_p}t64", must=False)

try:  # diagnostics must never kill the setup cell (nb01 v4 evidence)
    import glob as globmod
    icd_dir = "/usr/share/vulkan/icd.d"
    icd_txt = {open(p).read() for p in globmod.glob(icd_dir + "/*.json")} \
        if os.path.isdir(icd_dir) else set()
    if not any("nvidia" in s.lower() for s in icd_txt):
        os.makedirs(icd_dir, exist_ok=True)
        nvidia_icd = os.path.join(icd_dir, "nvidia_icd.json")
        with open(nvidia_icd, "w") as f:
            f.write('{\n    "file_format_version" : "1.0.0",\n'
                    '    "ICD": {\n'
                    '        "library_path": "libGLX_nvidia.so.0",\n'
                    '        "api_version" : "1.2.135"\n    }\n}\n')
        os.environ["VK_ICD_FILENAMES"] = nvidia_icd
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
# heads missing and the run dies minutes later inside the load cell. Verify
# sizes here and allow exactly one forced retry.
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
import sys, os, json
# path-agnostic dataset resolver + ambiguity refusal (v1/v2/v4 evidence)
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
# stale-dataset guard (2026-10-30): a mount without the actuation fixes makes
# MoveAhead(0.1 m) a silent no-op and every episode stalls (nb07 evidence).
for _need in ("contains_RotateLeft=True", "contains_D_recovery_yaw=True"):
    assert _need in _stamp_txt, f"STALE DATASET: BUILD.txt lacks {_need}"

from vlanav.model.omnivla_adapter import OmnivlaNavPolicy
from vlanav.environment.ai2thor_env import Ai2ThorNavEnv
from vlanav.reliability.replanning import (ChunkedFrozen, FixedFrequencyReplan,
                                           ReliabilityTriggered)
from vlanav.environment.types import EpisodeSpec, Observation
from vlanav.evaluation.logging import EpisodeLogger, ExperimentMeta

policy = OmnivlaNavPolicy(VLA_PATH,
                          prismatic_repo="/kaggle/tmp/OmniVLA", load_now=False)
# pre-load re-verification: /kaggle/tmp is shared with any other running
# kernel, so a file present in the setup cell can be gone by the load cell.
_vanish = [f for f in NEED
           if not (os.path.isfile(os.path.join(VLA_PATH, f))
                   and os.path.getsize(os.path.join(VLA_PATH, f)) > 1000)]
assert not _vanish, f"checkpoint files missing before load: {_vanish}"
policy.load()
assert policy.is_model, "refusing to run the test matrix with a mock policy"
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
controller = Controller(scene="FloorPlan401", width=400, height=400,
                        platform="Linux64", server_type="Linux64", gridSize=0.05)
ev = controller.reset(scene="FloorPlan401")
print("CONTROLLER OK | frame", ev.frame.shape, "| test scene rendered")

# ---- cell 5 ----------------------------------------
# %%time
# 3-tick smoke episode: v5 evidence says a bad THOR action argument otherwise
# surfaces only after the 15 GB model load + controller build.
from vlanav.environment.types import EpisodeSpec as _Spec
env_smoke = Ai2ThorNavEnv(controller)
spec_smoke = _Spec(episode_id="smoke-000", condition="A_in_dist", split="test",
                   seed=991, world_id=991, start_xy=(0.0, 0.0),
                   start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                   instruction="(set at reset)", max_steps=3,
                   goal_radius_m=0.6, extra={"smoke": True})
_smoke = ChunkedFrozen(policy, "chunked_frozen").run_episode(env_smoke, spec_smoke)
print("SMOKE OK | steps", _smoke.step, "| inf", _smoke.n_inferences,
      "| scene", spec_smoke.extra.get("scene"),
      "| moved", round(_smoke.path_length_m, 3), "m")
assert _smoke.step >= 1
# movement gate (nb07 evidence): with the wrong grid/yaw the robot advances 0 m
# and every "result" would be an actuation artefact, not a model outcome.
assert _smoke.path_length_m > 0.02, (
    "SMOKE FAILED: robot moved %.3f m in 3 ticks -> actuation is broken "
    "(gridSize/yaw); refusing to spend GPU hours" % _smoke.path_length_m)

# ---- cell 6 ----------------------------------------
# ---- B3 threshold: val-tuning artifact ONLY (never test) -------------
import glob as _g

thr_source, thr_cfg = None, None
for p in _g.glob("/kaggle/input/**/threshold_choice.json", recursive=True):
    try:
        d = json.load(open(p))
        if "threshold_m" in d:
            thr_cfg, thr_source = d, p
            break
    except Exception:
        continue
if thr_cfg is None:
    fb = None
    for p in _g.glob("/kaggle/input/**/reliability.yaml", recursive=True):
        fb = p
        break
    if fb:
        import re
        txt = open(fb).read()
        mthr = re.search(r"threshold_m:\s*([0-9.]+)", txt)
        mstall = re.search(r"stall_limit:\s*([0-9]+)", txt)
        thr_cfg = {"threshold_m": float(mthr.group(1)) if mthr else 0.2,
                   "stall_limit": int(mstall.group(1)) if mstall else 2,
                   "selection_rule": "FALLBACK: tuning artifact absent; "
                                     "configs/reliability.yaml value used",
                   "table": []}
        thr_source = fb
    else:
        thr_cfg = {"threshold_m": 0.2, "stall_limit": 2,
                   "selection_rule": "FALLBACK: no tuning artifact and no "
                                     "config; documented default 0.2",
                   "table": []}
        thr_source = "(none)"
print("THRESHOLD SOURCE:", thr_source)
print(json.dumps(thr_cfg, indent=2)[:900])

THRESHOLD = float(thr_cfg["threshold_m"])
STALL_LIMIT = int(thr_cfg.get("stall_limit", 2))

# ---- cell 7 ----------------------------------------
# %%time
import json, time
import pandas as pd

CONDS = ['C_instr_shift', 'D_recovery']
N_PER_COND = 16
MAX_STEPS = 60
SEED_RANGES = {"A_in_dist": (10000, 10199), "B_env_shift": (20000, 20199),
               "C_instr_shift": (30000, 30199), "D_recovery": (40000, 40199)}
TAG = "test-thor-20261030"

def spec_for(cond, i):
    """Test seeds = upper half of the condition's seed range (episodes.py
    convention); shared by all three regimes -> paired comparison."""
    lo, hi = SEED_RANGES[cond]
    seed = hi - 1 - i
    return EpisodeSpec(episode_id=f"{cond}/test/v1/{i:03d}",
                       condition=cond, split="test", seed=seed, world_id=seed,
                       start_xy=(0.0, 0.0), start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                       instruction="(filled at reset from THOR target)",
                       max_steps=MAX_STEPS, goal_radius_m=0.6,
                       extra={"benchmark_version": "v1",
                              "max_steps_note": "full run: 60 ticks (~6 m)"})

def make_regimes():
    # explicit canonical names: class-name derivation produced
    # "fixed_frequency_replan", which the analyzers do not key on (dry-run bug)
    return [("chunked_frozen", ChunkedFrozen(policy, "chunked_frozen")),
            ("fixed_frequency", FixedFrequencyReplan(policy, "fixed_frequency")),
            ("reliability_triggered",
             ReliabilityTriggered(policy, "reliability_triggered",
                                  threshold_m=THRESHOLD,
                                  stall_limit=STALL_LIMIT,
                                  max_replans_per_episode=24))]

env = Ai2ThorNavEnv(controller)
rows = []
t_start = time.time()
for cond in CONDS:
    for mname, regime in make_regimes():
        exp_id = f"{TAG}-{cond}-{mname}"
        out_dir = f"/kaggle/working/results/raw/{exp_id}"
        # --- resume: keep every already-logged row (verified, not re-run) ---
        done, prev = set(), []
        ep_path = os.path.join(out_dir, "episodes.jsonl")
        if os.path.isfile(ep_path):
            for line in open(ep_path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
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
            conditions=[cond], split="test", action_budget=MAX_STEPS,
            notes=("FULL test matrix. n=16/condition, seeds from the upper "
                   "half of the condition range, paired across regimes; "
                   "B3 threshold from VAL-only tuning "
                   f"({thr_source}); instruction via official modality 7."),
            extra={"seeds": [spec_for(cond, i).seed for i in range(N_PER_COND)],
                   "threshold_m": THRESHOLD, "stall_limit": STALL_LIMIT,
                   "threshold_source": thr_source,
                   "condition": cond, "n_per_condition": N_PER_COND,
                   "goal_radius_m": 0.6,
                   "scenes": "test block FloorPlan401..430"})
        logger = EpisodeLogger(meta, out_dir, overwrite=True)
        for r in prev:                      # restore evidence, then append
            logger.log_episode(r)
        print("=== ", exp_id, " ===")
        for i in range(N_PER_COND):
            spec = spec_for(cond, i)
            if spec.episode_id in done:
                continue
            res = regime.run_episode(env, spec)
            # per-row provenance (pilot v7 gap): the instruction actually sent
            # to the model and the THOR scene it was rendered in.
            row = dict(res.__dict__)
            row["instruction"] = spec.instruction
            row["scene"] = spec.extra.get("scene")
            logger.log_episode(row)
            rows.append(row)
            print(json.dumps({"exp": exp_id, "ep": res.episode_id,
                              "succ": res.success, "steps": res.step,
                              "d_final": round(res.distance_to_goal_m, 2),
                              "inf": res.n_inferences,
                              "replans": res.n_replans_triggered,
                              "elapsed_min": round((time.time() - t_start) / 60, 1)}))
    print("condition done:", cond, "| total elapsed min",
          round((time.time() - t_start) / 60, 1))

controller.stop()

# ---- summary over everything on disk (not just this session) ------------
all_rows = []
for p in sorted(__import__("glob").glob("/kaggle/working/results/raw/*/episodes.jsonl")):
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            all_rows.append(json.loads(line))
df = pd.DataFrame(all_rows)
summary = (df.groupby(["condition", "method"])
             .agg(SR=("success", "mean"), n=("success", "size"),
                  inference_mean=("n_inferences", "mean"),
                  replans_mean=("n_replans_triggered", "mean"),
                  steps_mean=("step", "mean"),
                  dist_final=("distance_to_goal_m", "mean"),
                  lat_mean_s=("infer_latency_s_mean", "mean"))
             .round(3))
print(summary.to_string())
summary.to_json("/kaggle/working/results/full_summary_b.json", orient="index")
print("rows:", len(df), "| saved full_summary_b.json")
