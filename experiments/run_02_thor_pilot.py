"""02_thor_pilot: the exact code that ran on Kaggle.

Extracted from the pushed notebook
(kaggle_notebooks/02_thor_pilot/02_thor_pilot.ipynb); markdown cells omitted, cell
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
report = {"notebook": "02_thor_pilot", "stage": "kernel-env",
          "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "python": platform.python_version()}
import torch
report["torch"] = torch.__version__
report["gpus"] = [torch.cuda.get_device_properties(i).name
                  for i in range(torch.cuda.device_count())]
print(json.dumps(report, indent=2))
json.dump(report, open("/kaggle/working/pilot_env.json", "w"), indent=2)

def cmd(s, must=True):
    print("$", s)
    r = subprocess.run(s, shell=True, capture_output=True, text=True)
    print(r.stdout[-1200:]); print(r.stderr[-1200:])
    assert (r.returncode == 0) if must else True
    return r.returncode

# --- pure-python pins (draccus==0.8.0 = repo's own pin; v5 evidence) ---
cmd("pip install -q 'timm==0.9.16' 'peft==0.11.1' 'draccus==0.8.0' 'utm' 'efficientnet_pytorch' 'ai2thor'")
# nb01 v4 evidence: Ubuntu 24.04 t64 renames; a single bad apt name aborts
# the WHOLE transaction, so install per-package with a t64 fallback.
cmd("apt-get update -qq", must=False)
for _p in ("xvfb", "x11-utils", "mesa-utils", "libgl1", "libglu1-mesa",
           "libegl1", "libxcursor1", "libxinerama1", "libxrandr2",
           "libxi6", "libxss1", "libnss3", "libvulkan1", "vulkan-tools"):
    if cmd(f"apt-get install -y -qq --no-install-recommends {_p}", must=False) != 0:
        cmd(f"apt-get install -y -qq --no-install-recommends {_p}t64", must=False)

# Vulkan ICD diagnostics must NEVER kill the setup cell (nb01 v4 evidence:
# /usr/share/vulkan/icd.d did not exist -> FileNotFoundError).
try:
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

# --- official OmniVLA code clone (required by the inference modules) ---
cmd("rm -rf /kaggle/tmp/OmniVLA", must=False)
cmd("git clone --depth 1 https://github.com/NHirose/OmniVLA.git /kaggle/tmp/OmniVLA")
assert os.path.isdir("/kaggle/tmp/OmniVLA/prismatic"), "clone incomplete"

# --- official transformers 4.40.1 (--no-deps, pure-python) + hub pin ---
cmd("pip install -q --no-deps 'transformers==4.40.1' 'huggingface_hub==0.29.1'")

# --- v6/v7-evidence fix: relax EXACTLY the tokenizers needle ---
import importlib.util
_tf_root = os.path.dirname(importlib.util.find_spec("transformers").origin)
_tbl = os.path.join(_tf_root, "dependency_versions_table.py")
src = open(_tbl, encoding="utf-8").read()
needle = '"tokenizers": "tokenizers>=0.19,<0.20"'
if needle not in src:
    raise RuntimeError("unexpected tokenizers spec — refusing blind patch")
open(_tbl, "w", encoding="utf-8").write(
    src.replace(needle, '"tokenizers": "tokenizers>=0.19"'))
print("tokenizers spec patched (exact needle)")

# --- checkpoint download (~15 GB -> /kaggle/tmp, NOT working) ---
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
# heads missing and the run then dies minutes later inside the load cell.
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

# ---- cell 4 ----------------------------------------
import sys, os, json
# The dataset mount layout/zip roots have bitten us twice (v1/v2/v4 evidence:
# stale nested copy at src/vlanav vs fresh src/src/vlanav). Resolve by finding
# EVERY candidate package root and REFUSING ambiguity, then print the dataset
# BUILD stamp so provenance is visible in the log.
cands = []
for root, dirs, _files in os.walk("/kaggle/input"):
    if "vlanav" in dirs and os.path.isfile(
            os.path.join(root, "vlanav", "__init__.py")):
        cands.append(root)
if not cands:
    raise RuntimeError("vlanav package not found under /kaggle/input; "
                       "top-level: " + repr(os.listdir('/kaggle/input')))
if len(cands) > 1:
    raise RuntimeError(f"ambiguous dataset mount - {len(cands)} copies of the "
                       f"vlanav package found: {cands}. Refusing to guess; "
                       "clean the dataset and re-upload.")
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
from vlanav.evaluation.logging import EpisodeLogger, ExperimentMeta
from vlanav.environment.types import EpisodeSpec

import torch, transformers
print("imports OK | torch", torch.__version__, "| transformers",
      transformers.__version__)

policy = OmnivlaNavPolicy(VLA_PATH,
                          prismatic_repo="/kaggle/tmp/OmniVLA",
                          load_now=False)
# pre-load re-verification: /kaggle/tmp is shared with any other running
# kernel, so a file present in the setup cell can be gone by the load cell.
_vanish = [f for f in NEED
           if not (os.path.isfile(os.path.join(VLA_PATH, f))
                   and os.path.getsize(os.path.join(VLA_PATH, f)) > 1000)]
assert not _vanish, f"checkpoint files missing before load: {_vanish}"
policy.load()   # single load (constructor excludes it)
print("POLICY LOADED:", json.dumps(policy.provenance, default=str))

# ---- cell 6 ----------------------------------------
# %%time
import numpy as np
from vlanav.environment.types import Observation

rng = np.random.default_rng(0)
img = rng.integers(0, 255, size=(400, 400, 3), dtype=np.uint8)
obs = Observation(image=img, robot_pose=np.array([0.0, 0.0, 0.0]), step=0)
chunk = policy.infer(obs, "Navigate to the fridge and stop there.")
w = chunk.data
assert w.shape == (8, 4), w.shape
assert np.isfinite(w).all(), "non-finite waypoints"
cmd = chunk.decode()
print("SANITY OK | waypoints[0,0] =", w[0].round(3),
      "| decoded cmd =", cmd.as_tuple())
one = policy.provenance_summary()
print("n_calls:", one["n_inference_calls"])

# ---- cell 8 ----------------------------------------
# %%time
import os, subprocess, time

os.environ.setdefault("DISPLAY", ":0")
from ai2thor.controller import Controller

try:
    Controller.start_xserver()
    xmode = "ai2thor_start_xserver"
except Exception as e:
    print("start_xserver failed:", repr(e)[-300:])
    subprocess.run("pkill Xvfb || true", shell=True)
    subprocess.Popen(["Xvfb", ":0", "-screen", "0", "1024x768x24", "-ac"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    time.sleep(3)
    xmode = "xvfb"
os.environ["DISPLAY"] = ":0"

def try_build(mode):
    # nb07 evidence: THOR's default 0.25 m movement grid makes MoveAhead(0.1 m)
    # a no-op -> every episode stalls. gridSize=0.05 resolves v*dt=0.1 m.
    kw = dict(width=400, height=400, server_type="Linux64", gridSize=0.05)
    if mode == "cloud":
        kw["platform"] = "CloudRendering"
    elif mode == "linux64":
        kw["platform"] = "Linux64"
    elif mode == "headless":
        kw["platform"] = "Linux64"
        kw["headless"] = True
    t0 = time.time()
    c = Controller(**kw)
    return c, time.time() - t0

build_mode, controller, build_secs = None, None, None
for mode in ("cloud", "linux64", "headless"):
    try:
        controller, secs = try_build(mode)
        build_mode, build_secs = mode, secs
        break
    except Exception as e:
        print(f"controller {mode} failed:", repr(e)[-400:])
        with open("/kaggle/working/pilot_env.json") as f0:
            report = json.load(f0)
            report[f"controller_{mode}_error"] = repr(e)[-400:]
            report["xmode"] = xmode
        with open("/kaggle/working/pilot_env.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
assert controller is not None, "no controller mode worked"
print("CONTROLLER:", build_mode, "x:", xmode, round(build_secs, 1), "s")

# ---- cell 10 ----------------------------------------
# %%time
from vlanav.environment.types import EpisodeSpec as _Spec
env_smoke = Ai2ThorNavEnv(controller)
spec_smoke = _Spec(episode_id="smoke-000", condition="A_in_dist", split="val",
                   seed=991, world_id=991, start_xy=(0.0, 0.0),
                   start_yaw_rad=0.0, goal_xy=(0.0, 0.0),
                   instruction="(set at reset)", max_steps=3,
                   goal_radius_m=0.6, extra={"smoke": True})
smoke_res = ChunkedFrozen(policy).run_episode(env_smoke, spec_smoke)
print("SMOKE OK | steps", smoke_res.step, "| inf", smoke_res.n_inferences,
      "| scene", spec_smoke.extra.get("scene"),
      "| moved", round(smoke_res.path_length_m, 3), "m",
      "| instr", spec_smoke.instruction)
assert smoke_res.step >= 1, "smoke episode did not execute a single tick"

# ---- cell 12 ----------------------------------------
import numpy as np
from vlanav.environment.types import Observation

env_sanity = Ai2ThorNavEnv(controller)
ev = controller.reset(scene="FloorPlan5")
env_sanity._pose = env_sanity._pose_from(ev)
env_sanity.spec = None
real_frame = np.asarray(ev.frame, dtype=np.uint8)
obs = Observation(image=real_frame, robot_pose=np.array(env_sanity._pose), step=0)
chunk = policy.infer(obs, "Navigate to the fridge and stop there.")
assert np.isfinite(chunk.data).all()
cmd = chunk.decode()
print("REAL-FRAME SANITY OK | cmd =", cmd.as_tuple())

# deterministic val episode set (documented seeds; world_id selects scene)
SEEDS = [10001 + 333 * i for i in range(10)]

def make_spec(i):
    seed = SEEDS[i]
    return EpisodeSpec(
        episode_id=f"thor-20261029-pilot-{i:03d}",
        condition="A_in_dist", split="val", seed=seed,
        world_id=i,
        start_xy=(0.0, 0.0),   # true spawn set by the env at reset
        start_yaw_rad=0.0,     # true spawn set by the env at reset
        goal_xy=(0.0, 0.0),    # true target set by the env at reset
        instruction="(filled at reset from THOR target)",
        max_steps=40,          # 40 ticks = up to ~13 s of robot time
        goal_radius_m=0.6,
        extra={"pilot": True, "max_steps_note": "longer passes during full run"},
    )

# ---- cell 14 ----------------------------------------
# %%time
import json, time
import pandas as pd

env = Ai2ThorNavEnv(controller)
REGIMES = [
    ("chunked_frozen", ChunkedFrozen(policy, "chunked_frozen")),
    ("fixed_frequency", FixedFrequencyReplan(policy, "fixed_frequency")),
    ("reliability_triggered", ReliabilityTriggered(
        policy, "reliability_triggered",
        threshold_m=0.2, stall_limit=2, max_replans_per_episode=24)),
]

all_rows = []
for exp_name, regime in REGIMES:
    exp_id = f"pilot-thor-20261029-{exp_name}"
    meta = ExperimentMeta(
        experiment_id=exp_id, method=regime.name,
        env_family="ai2thor", model_name="NHirose/omnivla-original",
        model_revision="omnivla-original@120000",
        gpu_name=";".join(torch.cuda.get_device_properties(i).name
                          for i in range(torch.cuda.device_count())),
        torch_version=torch.__version__,
        conditions=["A_in_dist"], split="val",
        action_budget=40,
        notes=("PILOT: n=10, max_steps=40; B3 threshold is the MiniNav-mock "
               "placeholder from configs/reliability.yaml (re-tune with the "
               "real VLA on val before headline claims); "
               "instruction via official modality 7 (language-only)."),
        extra={"seeds": SEEDS, "goal_radius_m": 0.6,
               "condition": "A_in_dist",
               # R24 fix (2026-10-30): the v7 pilot meta lacked this field, so
               # the evidence gate could not tell where B3's threshold came
               # from. The pilot threshold is a documented placeholder; the
               # val sweep (notebook 03) is the real source for the test runs.
               "threshold_m": 0.2,
               "threshold_source": ("PILOT PLACEHOLDER 0.2 m / stall 2 "
                                    "(mock-era default); superseded by the "
                                    "val-only sweep in notebook 03"),
               "latency_00_preflight_ms": {"mean": 3563.4, "p95": 3774.4}},
    )
    logger = EpisodeLogger(meta, f"/kaggle/working/results/raw/{exp_id}")
    print("=== ", exp_id, " ===")
    for i in range(10):
        spec = make_spec(i)
        res = regime.run_episode(env, spec)
        logger.log_episode(res)
        all_rows.append(res.__dict__)
        print(json.dumps({"episode": res.episode_id, "success": res.success,
                          "steps": res.step, "dist_final": round(res.distance_to_goal_m, 2),
                          "inferences": res.n_inferences,
                          "replans": res.n_replans_triggered,
                          "lat_mean_s": round(res.infer_latency_s_mean, 2)}))
    controller.reset(scene="FloorPlan5")  # gentle inter-regime reset
env.controller.stop()

df = pd.DataFrame(all_rows)
summary = (df.groupby("method")
             .agg(**{"SR": ("success", "mean"), "n": ("success", "size"),
                     "SPL_points": ("path_length_m", "sum"),
                     "inference_mean": ("n_inferences", "mean"),
                     "replans_mean": ("n_replans_triggered", "mean"),
                     "lat_mean_s": ("infer_latency_s_mean", "mean"),
                     "dist_final": ("distance_to_goal_m", "mean")})
             .round(3))
print(summary.to_string())
summary.to_json("/kaggle/working/results/pilot_summary.json", orient="index")
print("SAVED /kaggle/working/results/pilot_summary.json")
