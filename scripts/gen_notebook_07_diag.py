"""Generate notebook 07: AI2-THOR ACTUATION diagnostic (no model, ~5 min).

Pilot v6 evidence (2026-10-30): every episode had path_length_m = 0.0 and
`n_collisions = n_steps`, i.e. `MoveAhead` never produced displacement. This
kernel isolates the action layer:

  1. does `MoveAhead` move with the default magnitude / 0.25 / 0.10 / 0.05?
  2. does `RotateLeft(degrees=...)` change yaw, and by how much?
  3. does `Teleport(position=..., rotation=<float>)` land where we asked?
  4. does movement still work AFTER a teleport?
  5. does a smaller controller `gridSize` make 0.10 m moves possible?
  6. is `lastActionSuccess` / `errorMessage` telling us something?

Output: /kaggle/working/thor_diag.json  (each probe: before/after pose,
success flag, error message)

Run:  python scripts/gen_notebook_07_diag.py
Then: kaggle kernels push -p kaggle_notebooks/07_thor_diag
"""
import ast
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "kaggle_notebooks", "07_thor_diag")
os.makedirs(OUT, exist_ok=True)

md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t}
code = lambda t: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t}

cells = [
    md("""# VLA-NAV - Notebook 07: AI2-THOR actuation diagnostic

No model. Answers one question: **why did `MoveAhead(moveMagnitude=0.1)` move
the agent 0.0 m in the pilot?** Every probe records the pose before/after, the
action success flag and THOR's error message."""),

    code(r'''%%time
import json, os, platform, subprocess, time

os.makedirs("/kaggle/tmp", exist_ok=True)
os.environ["AI2THOR_DOWNLOAD_DIR"] = "/kaggle/tmp/aa"
os.environ["OBJECT_CACHE_DIR"] = "/kaggle/tmp/thor_objcache"
os.environ["XDG_CACHE_HOME"] = "/kaggle/tmp/xdgcache"

def cmd(s, must=True):
    print("$", s)
    r = subprocess.run(s, shell=True, capture_output=True, text=True)
    print(r.stdout[-400:]); print(r.stderr[-400:])
    assert (r.returncode == 0) if must else True
    return r.returncode

cmd("pip install -q 'ai2thor'")
cmd("apt-get update -qq", must=False)
for _p in ("xvfb", "x11-utils", "mesa-utils", "libgl1", "libglu1-mesa",
           "libegl1", "libxcursor1", "libxinerama1", "libxrandr2",
           "libxi6", "libxss1", "libnss3", "libvulkan1", "vulkan-tools"):
    if cmd(f"apt-get install -y -qq --no-install-recommends {_p}", must=False) != 0:
        cmd(f"apt-get install -y -qq --no-install-recommends {_p}t64", must=False)
print("env setup done")'''),

    code(r'''%%time
import subprocess, time
os.environ["DISPLAY"] = ":0"
subprocess.run("pkill Xvfb || true", shell=True)
subprocess.Popen(["Xvfb", ":0", "-screen", "0", "1024x768x24", "-ac",
                  "+extension", "GLX", "+render", "-noreset"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 start_new_session=True)
time.sleep(4)
from ai2thor.controller import Controller


def build(**kw):
    t0 = time.time()
    c = Controller(scene="FloorPlan5", width=400, height=400,
                   platform="Linux64", server_type="Linux64", **kw)
    print("built in", round(time.time() - t0, 1), "s", kw)
    return c


def pose(c):
    a = c.last_event.metadata["agent"]
    return (round(a["position"]["x"], 4), round(a["position"]["y"], 4),
            round(a["position"]["z"], 4), round(a["rotation"]["y"], 2))


def probe(c, tag, action):
    before = pose(c)
    ev = c.step(action)
    after = pose(c)
    m = ev.metadata
    rec = {"tag": tag, "action": {k: str(v) for k, v in action.items()},
           "before": before, "after": after,
           "moved_m": round(((after[0] - before[0]) ** 2
                             + (after[2] - before[2]) ** 2) ** 0.5, 4),
           "yaw_delta_deg": round((after[3] - before[3] + 180) % 360 - 180, 2),
           "success": bool(m["lastActionSuccess"]),
           "standing": m["agent"].get("isStanding"),
           "error": (m.get("errorMessage") or "")[:160]}
    print(json.dumps(rec))
    return rec'''),

    code(r'''%%time
RECORDS = []

# ---------- A: default grid (whatever THOR uses) -------------------------
c = build()
print("controller gridSize =", getattr(c, "gridSize", "?"))
m0 = c.last_event.metadata
print("scene:", m0["sceneName"], "| agent:", pose(c))
RECORDS.append({"probe": "build_default", "gridSize": getattr(c, "gridSize", None),
                "scene": m0["sceneName"], "pose": pose(c)})

RECORDS.append(probe(c, "default_MoveAhead", dict(action="MoveAhead")))
RECORDS.append(probe(c, "MoveAhead_0.25", dict(action="MoveAhead", moveMagnitude=0.25)))
RECORDS.append(probe(c, "MoveAhead_0.10", dict(action="MoveAhead", moveMagnitude=0.10)))
RECORDS.append(probe(c, "MoveAhead_0.05", dict(action="MoveAhead", moveMagnitude=0.05)))
RECORDS.append(probe(c, "RotateLeft_30", dict(action="RotateLeft", degrees=30)))
RECORDS.append(probe(c, "RotateRight_30", dict(action="RotateRight", degrees=30)))
RECORDS.append(probe(c, "RotateLeft_7.6", dict(action="RotateLeft", degrees=7.6)))

# ---------- B: teleport, then move --------------------------------------
p = c.last_event.metadata["agent"]["position"]
tp = dict(action="Teleport", position=dict(x=p["x"], y=p["y"], z=p["z"]),
          rotation=90.0, forceAction=True)
RECORDS.append(probe(c, "Teleport_rot_float_90", tp))
RECORDS.append(probe(c, "MoveAhead_after_teleport_0.10",
                     dict(action="MoveAhead", moveMagnitude=0.10)))
RECORDS.append(probe(c, "MoveAhead_after_teleport_0.25",
                     dict(action="MoveAhead", moveMagnitude=0.25)))
p = c.last_event.metadata["agent"]["position"]
tp2 = dict(action="Teleport", position=dict(x=p["x"], y=p["y"], z=p["z"]),
           rotation=dict(x=0.0, y=0.0, z=0.0), forceAction=True)
try:
    RECORDS.append(probe(c, "Teleport_rot_dict", tp2))
except Exception as e:
    RECORDS.append({"probe": "Teleport_rot_dict", "exception": repr(e)[:300]})
c.stop()'''),

    code(r'''%%time
# ---------- C: smaller gridSize, does 0.10 m become possible? ------------
c2 = build(gridSize=0.05)
print("gridSize =", getattr(c2, "gridSize", "?"))
RECORDS.append({"probe": "build_grid005", "gridSize": getattr(c2, "gridSize", None),
                "pose": pose(c2)})
RECORDS.append(probe(c2, "g005_MoveAhead_0.10", dict(action="MoveAhead", moveMagnitude=0.10)))
RECORDS.append(probe(c2, "g005_MoveAhead_0.05", dict(action="MoveAhead", moveMagnitude=0.05)))
RECORDS.append(probe(c2, "g005_MoveAhead_default", dict(action="MoveAhead")))
RECORDS.append(probe(c2, "g005_RotateLeft_7.6", dict(action="RotateLeft", degrees=7.6)))
c2.stop()

json.dump(RECORDS, open("/kaggle/working/thor_diag.json", "w"), indent=2)
print("saved thor_diag.json with", len(RECORDS), "records")'''),

    md("""## How to read this

* `moved_m` after `MoveAhead(moveMagnitude=m)` tells us THOR's real movement
  resolution. If `m = 0.10` moves 0 but `m = 0.25` moves 0.25, the agent is
  grid-quantized and the env must use a matching grid / larger step.
* `yaw_delta_deg` shows whether degree-level rotations are honoured (needed for
  `ω·dt` with ω up to 0.3 rad/s → up to 5.7°/tick).
* `Teleport_rot_float_90` validates the spawn-heading call; a wrong type would
  raise instead of silently landing at yaw 0.
* `success=False` with a non-empty `error` means the client did NOT raise but
  the action failed — exactly the silent-no-motion case the pilot hit."""),
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
with open(os.path.join(OUT, "07_thor_diag.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    print("wrote", f.name)

meta = {"id": "pikanigga/vlanav-07-thor-diag", "title": "vlanav-07-thor-diag",
        "code_file": "07_thor_diag.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
        "enable_t4x2": "true", "enable_internet": "true",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    print("wrote", f.name)
