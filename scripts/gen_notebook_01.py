"""Generate the Kaggle notebook 01_thor.ipynb (+ kernel-metadata.json).

Purpose: prove the environment half of the pipeline on Kaggle - AI2-THOR
renders REAL egocentric RGB, steps are timed, and frames pass sanity gates.
This is *environment provenance only* (no model claims; notebook 00 proved
the model half with its v7 ALL PASS report).

v2 (evidence-driven rewrite after v1 failed all three controller modes):
    v1 report evidence (fetched 2026-10-29, thor_report.json):
      - cloud  : Unity downloaded+launched but exited returncode=-6
                 ("Confirm Vulkan ... properly configured") - the classic
                 missing-NVIDIA-Vulkan-ICD symptom.
      - headless: TypeError NoneType (ai2thor 5 headless path quirk).
      - xvfb   : "No valid X display found ... Linux64 requires a X11
                 server ... with GLX. If you have an NVIDIA GPU, please
                 run: sudo ai2thor-xorg start".
    v2 report evidence (fetched 2026-10-29):
      - apt install DIED ENTIRELY: "E: Package 'libasound2' has no
        installation candidate" - Kaggle is Ubuntu 24.04 (noble) where
        several library names are renamed with a t64 suffix, and apt
        aborts the whole transaction (NO xvfb/vulkan/mesa got installed).
      - ai2thor_colab.start_xserver() verified to be a no-op off Colab
        (its code gates the whole X setup behind `import google.colab`);
        we must start X ourselves.
      - headless Unity DID start a backend (fifo answered) but the first
        action timed out at 100 s - deps missing, inconclusive.
      - my own bug: try_build called `ev.step(...)` (Event has no step).
    v3 changes:
      1. per-package apt loop with automatic `pkg -> pkg` + `pkg`t64`
         fallback so one bad name cannot kill the transaction.
      2. Xvfb started with `+extension GLX +render` (GLX validation args
         used by ai2thor-colab itself) + xdpyinfo verification.
      3. NVIDIA Vulkan ICD write + vulkaninfo summary (now properly
         installed) recorded in the report.
      4. every failed controller mode captures the Unity Player.log tail
         into the report for evidence-driven v4.
Every generated code cell is ast.parse-checked before writing.
Every generated code cell is ast.parse-checked before writing.
Run:  python scripts/gen_notebook_01.py
Then: kaggle kernels push -p kaggle_notebooks/01_thor
"""
import ast
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "kaggle_notebooks", "01_thor")
os.makedirs(OUT, exist_ok=True)

md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t}
code = lambda t: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t}

cells = [
    md("""# VLA-NAV - Notebook 01 v3: AI2-THOR environment preflight (Kaggle T4)

**Purpose.** Verify AI2-THOR produces REAL egocentric RGB frames on the
Kaggle T4, time the sim step, and save a frame as evidence. Navigation
claims come from notebooks 02+; this notebook only proves the
*environment* half of the pipeline.

Report: `/kaggle/working/thor/thor_report.json` (written at every stage,
even on failure). v1 evidence is quoted in the change log above.""",

        ),

    code(r'''%%time
import json, os, platform, subprocess, time

os.makedirs("/kaggle/working/thor", exist_ok=True)
os.makedirs("/kaggle/tmp", exist_ok=True)
os.environ["AI2THOR_DOWNLOAD_DIR"] = "/kaggle/tmp/aa"
os.environ["OBJECT_CACHE_DIR"] = "/kaggle/tmp/thor_objcache"
os.environ["XDG_CACHE_HOME"] = "/kaggle/tmp/xdgcache"
report = {"notebook": "01_thor", "version": 3, "stage": "kernel-env",
          "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "python": platform.python_version()}
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[-1500:])

import torch
report["torch"] = torch.__version__
report["gpus"] = [torch.cuda.get_device_properties(i).name
                  for i in range(torch.cuda.device_count())]
json.dump(report, open("/kaggle/working/thor/thor_report.json", "w"),
          indent=2, default=str)
print(json.dumps(report, indent=2))''',

        ),

    md("""## 1. Install ai2thor + the FULL X/GLX/Vulkan dependency set (t64-safe)

v2 evidence: on Ubuntu 24.04 (Kaggle) `libasound2` has no candidate
(renamed `libasound2t64`), and apt aborts the WHOLE transaction on one
bad name - nothing got installed. We therefore install package-by-
package with an automatic `t64` fallback per name."""),

    code(r'''%%time
import subprocess

def run(s):
    print("$", s)
    r = subprocess.run(s, shell=True, capture_output=True, text=True)
    print(r.stdout[-1000:]); print(r.stderr[-1000:])
    return r.returncode

run("pip install -q ai2thor")
run("apt-get update -qq")

# Per-package install with t64 fallback (v2 evidence: one bad name
# otherwise kills the whole transaction).
PKGS = ("xvfb", "x11-utils", "mesa-utils", "libgl1", "libglu1-mesa",
        "libegl1", "libxcursor1", "libxinerama1", "libxrandr2", "libxi6",
        "libxss1", "libnss3", "libvulkan1", "vulkan-tools",
        "xserver-xorg-core", "xserver-xorg-video-dummy")
failures = []
for p in PKGS:
    if run(f"apt-get install -y -qq --no-install-recommends {p}") != 0:
        if run(f"apt-get install -y -qq --no-install-recommends {p}t64") != 0:
            failures.append(p)
report["apt_pkg_failures"] = failures
print("apt failures:", failures)

# NVIDIA Vulkan ICD (classic cloud-T4 fix; harmless if present already).
import glob as globmod
icd_dir = "/usr/share/vulkan/icd.d"
icds = globmod.glob(icd_dir + "/*.json")
icd_txt = {open(p).read() for p in icds} if icds else set()
if not any("nvidia" in s.lower() for s in icd_txt):
    # v4 evidence: /usr/share/vulkan/icd.d may not exist -> open(..., "w")
    # raised FileNotFoundError and aborted the whole diagnostic cell.
    os.makedirs(icd_dir, exist_ok=True)
    nvidia_icd = os.path.join(icd_dir, "nvidia_icd.json")
    with open(nvidia_icd, "w") as f:
        f.write('{\n'
                '    "file_format_version" : "1.0.0",\n'
                '    "ICD": {\n'
                '        "library_path": "libGLX_nvidia.so.0",\n'
                '        "api_version" : "1.2.135"\n'
                '    }\n'
                '}\n')
    os.environ["VK_ICD_FILENAMES"] = nvidia_icd
    report["nvidia_vulkan_icd_written"] = nvidia_icd
else:
    report["nvidia_vulkan_icd_present"] = [os.path.basename(p) for p in icds]

r = subprocess.run("vulkaninfo --summary 2>&1 | head -40", shell=True,
                   capture_output=True, text=True)
report["vulkaninfo_summary"] = r.stdout[-1500:] or r.stderr[-800:]
print(report["vulkaninfo_summary"])

import ai2thor
report["ai2thor_version"] = ai2thor.__version__
json.dump(report, open("/kaggle/working/thor/thor_report.json", "w"),
          indent=2, default=str)
print("ai2thor", ai2thor.__version__)'''),

    md("""## 2. Start Xvfb with GLX (+render) and verify the display

ai2thor_colab.start_xserver() is a verified no-op off Colab (v2
evidence); we start Xvfb ourselves with the same GLX-extension args
ai2thor-colab uses, and verify with xdpyinfo."""),

    code(r'''%%time
import os, subprocess, time

report["xmode"] = None
os.environ["DISPLAY"] = ":0"
subprocess.run("pkill Xvfb || true", shell=True)
proc = subprocess.Popen(["/usr/bin/Xvfb", ":0", "-screen", "0",
                         "1024x768x24", "-ac",
                         "+extension", "GLX", "+render", "-noreset"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True)
time.sleep(4)
report["xvfb_pid"] = proc.pid
r = subprocess.run("xdpyinfo -display :0 2>&1 || true",
                   shell=True, capture_output=True, text=True)
report["xdpyinfo_head"] = " ".join(r.stdout.split()[:20])
ok_xvfb = ("GLX" in r.stdout) and proc.poll() is None
report["xmode"] = "xvfb_glx" if ok_xvfb else "xvfb_unverified"
print("xdpyinfo:", report["xdpyinfo_head"])
json.dump(report, open("/kaggle/working/thor/thor_report.json", "w"),
          indent=2, default=str)''',

        ),

    md("""## 3. Controller ladder: CloudRendering, Linux64(X), headless

Each mode gets its own keywords; every failure is recorded with its
truncated traceback so the next iteration is evidence-driven.""",

        ),

    code(r'''%%time
import os, time, traceback

from ai2thor.controller import Controller

def try_build(mode):
    kw = dict(scene="FloorPlan5", width=400, height=400,
              server_type="Linux64")
    if mode == "cloud":
        kw["platform"] = "CloudRendering"
    elif mode == "linux64":
        kw["platform"] = "Linux64"     # GLX via the running Xvfb
    elif mode == "headless":
        kw["platform"] = "Linux64"
        kw["headless"] = True
    t0 = time.time()
    c = Controller(**kw)
    c.reset(scene="FloorPlan5")        # proves the sim answers
    c.step(dict(action="Done"))
    return c, time.time() - t0

def player_log_tail() -> str:
    import glob as globmod
    pats = globmod.glob(os.path.expanduser(
        "~/.config/unity3d/Allen*/AI2-THOR/Player.log"))
    if not pats:
        return "(Player.log not found)"
    try:
        return "\n".join(open(pats[0], errors="replace").readlines()[-25:])
    except Exception as e:
        return f"(read failed: {e!r})"

build_mode, build_secs, controller = None, None, None
for mode in ("cloud", "linux64", "headless"):
    try:
        controller, secs = try_build(mode)
        build_mode, build_secs = mode, secs
        break
    except Exception as e:
        report[f"controller_{mode}_error"] = repr(e)[-900:]
        report.setdefault("controller_tracebacks", {})[mode] = \
            traceback.format_exc()[-1200:]
        report.setdefault("unity_player_log_tails", {})[mode] = \
            player_log_tail()
        print(f"[{mode}] FAIL:", repr(e)[-500:])
        print("[", mode, "] Player.log tail:\n",
              report["unity_player_log_tails"][mode][-1500:])
        json.dump(report, open("/kaggle/working/thor/thor_report.json", "w"),
                  indent=2, default=str)

report["controller"] = {"mode": build_mode,
                        "build_seconds": None if build_secs is None
                        else round(build_secs, 1),
                        "xmode_used": report.get("xmode")}
json.dump(report, open("/kaggle/working/thor/thor_report.json", "w"),
          indent=2, default=str)
print("controller built:", build_mode, build_secs and round(build_secs, 1), "s")''',

        ),

    md("""## 4. Real render + step timings + sanity gates""",

        ),

    code(r'''%%time
import numpy as np, statistics, time, traceback

try:
    ev = controller.reset(scene="FloorPlan5")
    controller.step(dict(action="RotateRight"))
    controller.step(dict(action="RotateLeft"))

    times, frames = [], []
    for i in range(12):
        t0 = time.perf_counter()
        # v3 evidence: render_image is NOT a valid step kwarg in ai2thor 5
        # (server error "invalid argument: 'render_image'"); steps render
        # by default.
        ev = controller.step(dict(action="MoveAhead"))
        dtms = (time.perf_counter() - t0) * 1000.0
        frame = np.asarray(ev.frame, dtype=np.uint8)
        if i >= 2:
            times.append(dtms)
        frames.append(frame)

    f = frames[-1].astype(np.float64)
    gates = {
        "controller_built": build_mode is not None,
        "frame_real_rgb": bool(frames[-1].dtype == np.uint8
                               and frames[-1].ndim == 3
                               and frames[-1].shape[2] == 3),
        "frame_not_blank": bool(20.0 < float(f.std()) < 150.0),
        "step_timing_recorded": len(times) == 10,
    }
    report["render"] = {
        "frame_shape": list(frames[-1].shape),
        "frame_std": round(float(f.std()), 1),
        "step_ms": {"mean": round(statistics.mean(times), 1),
                    "median": round(statistics.median(times), 1)},
    }
    report["gates"] = gates
    report["verdict"] = ("ALL PASS" if all(gates.values())
                         else "*** GATES FAILED ***")
    import PIL.Image as PILImage
    PILImage.fromarray(frames[-1]).save(
        "/kaggle/working/thor/sample_frame.jpg")
    json.dump(report, open("/kaggle/working/thor/thor_report.json", "w"),
              indent=2, default=str)
    print(json.dumps(report["render"], indent=2))
    print("GATES:", report["gates"], "->", report["verdict"])
    controller.stop()
except Exception:
    report["render"] = {"error": traceback.format_exc()[-3000:]}
    json.dump(report, open("/kaggle/working/thor/thor_report.json", "w"),
              indent=2, default=str)
    raise''',

        ),

    md("""## 5. What v2 establishes / does not establish

**Established on success:** AI2-THOR serves REAL egocentric RGB on the
Kaggle T4 with a recorded rendering mode; step latency logged; frame +
JSON report as evidence.

**NOT established:** anything about the model (notebook 00) or any
navigation result (notebooks 02+).

**Stop rule:** any gate failure => read the recorded errors, fix
v3, push - never cite until ALL PASS.""",

        ),

    code(r'''import json
report = json.load(open("/kaggle/working/thor/thor_report.json"))
print(json.dumps({k: report.get(k) for k in
                  ("ai2thor_version", "xmode", "controller", "render",
                   "gates", "verdict")}, indent=2, default=str))''',

        ),
]

for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        plain = "\n".join(l for l in "".join(c["source"]).split("\n")
                          if not l.lstrip().startswith(("%%", "%")))
        ast.parse(plain)

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(os.path.join(OUT, "01_thor.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    print("wrote", f.name)

meta = {"id": "pikanigga/vlanav-01-thor", "title": "vlanav-01-thor",
        "code_file": "01_thor.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_t4x2": "false",
        "enable_internet": "true", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": []}
with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    print("wrote", f.name)

