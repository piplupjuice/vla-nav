"""01_thor: the exact code that ran on Kaggle.

Extracted from the pushed notebook
(kaggle_notebooks/01_thor/01_thor.ipynb); markdown cells omitted, cell
boundaries marked. Cell magics are kept as comments so the file
stays plain Python."""


# ---- cell 2 ----------------------------------------
# %%time
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
print(json.dumps(report, indent=2))

# ---- cell 4 ----------------------------------------
# %%time
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
print("ai2thor", ai2thor.__version__)

# ---- cell 6 ----------------------------------------
# %%time
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
          indent=2, default=str)

# ---- cell 8 ----------------------------------------
# %%time
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
print("controller built:", build_mode, build_secs and round(build_secs, 1), "s")

# ---- cell 10 ----------------------------------------
# %%time
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
    raise

# ---- cell 12 ----------------------------------------
import json
report = json.load(open("/kaggle/working/thor/thor_report.json"))
print(json.dumps({k: report.get(k) for k in
                  ("ai2thor_version", "xmode", "controller", "render",
                   "gates", "verdict")}, indent=2, default=str))
