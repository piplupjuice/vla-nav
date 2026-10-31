# 00_preflight - load the model on 2x T4 and measure query latency.

import json, os, platform, subprocess, time

os.makedirs("/kaggle/working/preflight", exist_ok=True)
report = {
    "notebook": "00_preflight",
    "stage": "kernel-env",
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": platform.python_version(),
}

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch, transformers, numpy as np
report["torch"] = torch.__version__
report["cuda_available"] = torch.cuda.is_available()
report["cuda_version"] = torch.version.cuda
report["transformers"] = transformers.__version__
report["numpy"] = np.__version__
report["gpus"] = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        report["gpus"].append({"name": p.name, "total_mem_gb": round(p.total_memory / 1e9, 1)})
with open("/kaggle/working/preflight/env_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))

import os, subprocess, sys

def cmd(s, must=True):
    print("$", s)
    r = subprocess.run(s, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-3000:])
    if must:
        assert r.returncode == 0, f"FAILED: {s}"
    return r.returncode

# pure-python pins (NO torch/sentencepiece/tokenizers — keep Kaggle's stack;
# draccus==0.8.0 is the repo's own pin — pure python wheel, py3-none-any)
cmd("pip install -q 'timm==0.9.16' 'peft==0.11.1' 'draccus==0.8.0' 'utm' 'efficientnet_pytorch'")

import timm, peft, utm, efficientnet_pytorch
print("timm:", timm.__version__, "| peft:", peft.__version__,
      "| efficientnet_pytorch:", efficientnet_pytorch.__version__)

# clone the official code, HARD-asserted this time (v2's silent failure path)
for name, url in [("OmniVLA", "https://github.com/NHirose/OmniVLA.git")]:
    d = f"/kaggle/tmp/{name}"
    cmd(f"rm -rf {d}", must=False)
    cmd(f"git clone --depth 1 {url} {d}")
    assert os.path.isdir(os.path.join(d, "prismatic")) or not name == "OmniVLA", "clone incomplete"
os.makedirs("/kaggle/tmp", exist_ok=True)
print("OmniVLA repo files:", sorted(os.listdir("/kaggle/tmp/OmniVLA"))[:12])

from huggingface_hub import snapshot_download
import os

VLA_PATH = snapshot_download(
    repo_id="NHirose/omnivla-original",
    local_dir="/kaggle/tmp/omnivla-original",
)
files = {f: os.path.getsize(os.path.join(VLA_PATH, f)) / 1e9
         for f in os.listdir(VLA_PATH) if os.path.isfile(os.path.join(VLA_PATH, f))}
for f, gb in sorted(files.items(), key=lambda kv: -kv[1])[:12]:
    print(f"{gb:6.2f} GB  {f}")
need = ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
        "action_head--120000_checkpoint.pt",
        "proprio_projector--120000_checkpoint.pt", "config.json",
        "processing_prismatic.py", "modeling_prismatic.py",
        "configuration_prismatic.py"]
missing = [f for f in need if not os.path.isfile(os.path.join(VLA_PATH, f))]
assert not missing, f"checkpoint incomplete: missing {missing}"
print("checkpoint OK at", VLA_PATH)

PROBE = r"""
import json, math, os, statistics, subprocess, sys, time

os.makedirs("/kaggle/working/preflight", exist_ok=True)
report = {"notebook": "00_preflight", "stage": "probe-subprocess",
          "reference": "inference/run_omnivla.py @ NHirose/OmniVLA, fetched 2026-10-28",
          "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

def save():
    with open("/kaggle/working/preflight/preflight_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

def bail(stage, exc):
    report[stage] = {"error": repr(exc), "type": type(exc).__name__}
    save()
    raise

try:
    # transformers 4.40.1 is the exact version the model's remote code
    # expects ("Expected transformers==4.40.1" WARN in modeling_prismatic.py,
    # accessed 2026-10-28). Pure-python wheel: no C build needed on py3.13.
    # huggingface_hub 0.29.1: deterministic hub pin compatible with 4.40.1
    # (the model registry state we load from is a local directory anyway).
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q",
                    "transformers==4.40.1", "huggingface_hub==0.29.1"], check=True)
except Exception as e:
    bail("pip_transformers_4401", e)

try:
    # v3 evidence: importing transformers 4.40.1 hard-fails on Kaggle because
    # its dependency table demands `tokenizers>=0.19,<0.20` and Kaggle ships
    # tokenizers 0.23.1 (0.19.1 has no cp313 wheel — v2 proved source-builds
    # die). The model code itself only WARNS on tokenizers mismatch, its slow
    # tokenizer path does not import the Rust lib per-call, so the only real
    # blocker is transformers' own import-time table check. Minimal,
    # logged, exact-needle patch: relax ONLY the tokenizers spec.
    import importlib.util
    _tf_origin = importlib.util.find_spec("transformers").origin
    _tf_root = os.path.dirname(_tf_origin)
    _table_path = os.path.join(_tf_root, "dependency_versions_table.py")
    _needle = '"tokenizers": "tokenizers>=0.19,<0.20"'
    with open(_table_path, encoding="utf-8") as f:
        _src = f.read()
    if _needle not in _src:
        raise RuntimeError(
            "unexpected tokenizers spec in transformers dependency table "
            "(file unchanged, no auto-patch possible): " + _table_path)
    with open(_table_path, "w", encoding="utf-8") as f:
        f.write(_src.replace(_needle, '"tokenizers": "tokenizers>=0.19"'))
    report["pip_transformers_4401"] = {
        "tokenizers_spec_patched": True,
        "patched_file": _table_path,
        "patch": _needle + '  ->  "tokenizers": "tokenizers>=0.19"',
        "reason": "transformers 4.40.1 import-time gate vs Kaggle tokenizers "
                  "0.23.1 (no cp313 wheel for 0.19.1); model code is "
                  "warn-only on tokenizers version",
    }
    save()
except Exception as e:
    bail("patch_tokenizers_spec", e)

try:
    import numpy as np
    import torch
    import transformers, tokenizers, timm
    report["probe_env"] = {
        "python": sys.version, "torch": torch.__version__,
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "timm": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_properties(i).name
                 for i in range(torch.cuda.device_count())],
    }
    save()
except Exception as e:
    bail("imports", e)

# --- stage: official imports (run_omnivla.py import block, same class paths) ---
try:
    sys.path.insert(0, "/kaggle/tmp/OmniVLA")
    VLA_PATH = "/kaggle/tmp/omnivla-original"
    import utm

    # v5 evidence: importing ANY prismatic.* module runs the eager package
    # __init__ chain -> prismatic/vla/datasets/rlds/dataset.py ->
    # `import dlimp` (which itself needs TensorFlow; no sane py3.13 path).
    # The inference-relevant source files (constants, action_tokenizer,
    # projectors, action_heads, prompting, train_utils, extern.hf.*) are all
    # plain modules, so we register *stub parent packages* in sys.modules
    # with correct __path__ and let Python load the real leaf modules
    # directly, skipping only the eager __init__ files. This is surgical:
    # the exact official inference sources still execute verbatim.
    import sys as _sys
    import types as _types

    def install_prismatic_stub_pkgs(repo_root):
        chain = ["prismatic", "prismatic.vla", "prismatic.models",
                 "prismatic.models.backbones", "prismatic.models.backbones.llm",
                 "prismatic.training", "prismatic.extern", "prismatic.extern.hf"]
        for name in chain:
            if name in _sys.modules:
                continue
            path = repo_root + "/" + name.replace(".", "/")
            mod = _types.ModuleType(name)
            mod.__path__ = [path]
            mod.__package__ = name
            _sys.modules[name] = mod
        return chain

    stubbed = install_prismatic_stub_pkgs("/kaggle/tmp/OmniVLA")
    report["prismatic_stub_pkgs"] = {
        "stubbed": stubbed,
        "reason": "bypass eager __init__ chain that imports the RLDS training "
                  "stack (dlimp/tensorflow); inference-relevant modules load "
                  "verbatim, documented & surgical",
    }

    from prismatic.vla.constants import (ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM,
                                         ACTION_PROPRIO_NORMALIZATION_TYPE)
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.action_heads import L1RegressionActionHead_idcat
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import (PrismaticImageProcessor,
                                                          PrismaticProcessor)
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.training.train_utils import (get_current_action_mask,
                                                get_next_actions_mask)
    from torch.nn.utils.rnn import pad_sequence
    from transformers import (AutoConfig, AutoImageProcessor, AutoModelForVision2Seq,
                              AutoProcessor)
    report["constants"] = dict(NUM_ACTIONS_CHUNK=NUM_ACTIONS_CHUNK,
                               ACTION_DIM=ACTION_DIM, POSE_DIM=POSE_DIM,
                               NORM=str(ACTION_PROPRIO_NORMALIZATION_TYPE))
    save()
except Exception as e:
    import traceback
    report["prismatic_imports"] = {"error": traceback.format_exc()[-3000:]}
    save()
    raise

# --- stage: model load (official define_model() replica; one exact deviation) ---
try:
    device_id = torch.device("cuda:0")
    torch.cuda.set_device(device_id)

    # official: register OpenVLA classes to HF Auto (model is NOT on HF Hub auto-map
    # when loaded from the official repo-path classes)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(VLA_PATH)
    # DEVIATION (documented): official does .to(device_id) single-GPU; the 8B
    # checkpoint is ~15.4 GB bf16 against a 15.6 GB T4 (no headroom), so we use
    # accelerate's device_map="auto" to shard across the two T4s.
    vla = AutoModelForVision2Seq.from_pretrained(
        VLA_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto",
    )
    vla.vision_backbone.set_num_images_in_input(2)
    vla.to(dtype=torch.bfloat16)

    def remove_ddp_in_checkpoint(state_dict):
        return {k[7:] if k.startswith("module.") else k: v
                for k, v in state_dict.items()}

    def load_checkpoint(module_name, path, step, device="cpu"):
        if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) \
                and module_name == "pose_projector":
            module_name = "proprio_projector"
        sd = torch.load(os.path.join(path, f"{module_name}--{step}_checkpoint.pt"),
                        map_location=device)
        return remove_ddp_in_checkpoint(sd)

    pose_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=POSE_DIM)
    pose_projector.load_state_dict(load_checkpoint("pose_projector", VLA_PATH, 120000))
    pose_projector = pose_projector.to(torch.bfloat16).to(device_id)
    action_head = L1RegressionActionHead_idcat(
        input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)
    action_head.load_state_dict(load_checkpoint("action_head", VLA_PATH, 120000))
    action_head = action_head.to(torch.bfloat16).to(device_id)

    # official: NUM_PATCHES = patches * num_images_in_input, +1 "for goal pose"
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * \
        vla.vision_backbone.get_num_images_in_input() + 1
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dev_counts = {}
    for k, v in getattr(vla, "hf_device_map", {}).items():
        key = f"{v}"
        dev_counts[key] = dev_counts.get(key, 0) + 1
    report["model_load"] = {
        "seconds": round(time.time() - t0, 1),
        "llm_dim": vla.llm_dim,
        "num_patches": NUM_PATCHES,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "processor_class": type(processor).__name__,
        "model_class": type(vla).__name__,
        "device_map": "auto (documented deviation from official single-GPU .to())",
        "hf_device_map_module_counts": dev_counts,
        "vram_after_load_gb": [round(torch.cuda.memory_allocated(i) / 1e9, 2)
                               for i in range(torch.cuda.device_count())],
    }
    save()
except Exception as e:
    import traceback
    report["model_load"] = {"error": traceback.format_exc()[-3000:]}
    save()
    raise

# --- stage: official-flow inference tick, 3 warmups + 10 timed ---
def clip_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi

try:
    # official sample geoloc (run_omnivla.py main entry, verbatim values)
    thres_dist = 30.0
    metric_waypoint_spacing = 0.1
    current_lat, current_lon, current_compass = 37.87371258374039, -122.26729417226024, 270.0
    cur_utm = utm.from_latlon(current_lat, current_lon)
    cur_compass = -float(current_compass) / 180.0 * math.pi
    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm = utm.from_latlon(goal_lat, goal_lon)
    goal_compass = -float(goal_compass) / 180.0 * math.pi
    delta_x, delta_y = goal_utm[0] - cur_utm[0], goal_utm[1] - cur_utm[1]
    relative_x = delta_x * math.cos(cur_compass) + delta_y * math.sin(cur_compass)
    relative_y = -delta_x * math.sin(cur_compass) + delta_y * math.cos(cur_compass)
    radius = math.sqrt(relative_x**2 + relative_y**2)
    if radius > thres_dist:
        relative_x *= thres_dist / radius
        relative_y *= thres_dist / radius
    goal_pose_loc_norm = np.array([
        relative_y / metric_waypoint_spacing,
        -relative_x / metric_waypoint_spacing,
        math.cos(goal_compass - cur_compass),
        math.sin(goal_compass - cur_compass),
    ])

    # synthetic images (no external assets needed; log with the report)
    from PIL import Image as PILImage
    rng = np.random.default_rng(120000)
    current_image_PIL = PILImage.fromarray(
        rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)).convert("RGB")
    goal_image_PIL = PILImage.fromarray(
        rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)).convert("RGB")
    lan_inst = "xxxx"   # official default: lan_prompt=False -> "No language instruction"

    def data_transformer_omnivla(current_image_PIL, lan_inst, goal_image_PIL,
                                 goal_pose_loc_norm):
        actions = np.random.rand(8, 4)   # official: dummy actions only build the token template
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        if lan_inst == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lan_inst}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        prompt_builder = PurePromptBuilder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        base_tokenizer = processor.tokenizer
        input_ids = torch.tensor(
            base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        image_transform = processor.image_processor.apply_transform
        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        batch_data = dict(pixel_values_current=pixel_values_current,
                          pixel_values_goal=pixel_values_goal,
                          input_ids=input_ids, labels=labels,
                          dataset_name="lelan", actions=torch.as_tensor(actions),
                          goal_pose=goal_pose_loc_norm)
        # official collator_custom
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        input_ids_b = pad_sequence([batch_data["input_ids"]], batch_first=True,
                                   padding_value=pad_token_id)
        labels_b = pad_sequence([batch_data["labels"]], batch_first=True,
                                padding_value=IGNORE_INDEX)
        attention_mask = input_ids_b.ne(pad_token_id)
        pixel_values = torch.cat(
            (torch.stack([batch_data["pixel_values_current"]]),
             torch.stack([batch_data["pixel_values_goal"]])), dim=1)
        batch = dict(pixel_values=pixel_values, input_ids=input_ids_b,
                     attention_mask=attention_mask, labels=labels_b)
        return batch

    def run_forward_pass(batch):
        # official modality table: image_goal only -> 6 ("image only")
        modality_id = torch.as_tensor([6], dtype=torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=torch.as_tensor(goal_pose_loc_norm, dtype=torch.float32
                                        ).to(torch.bfloat16).to(device_id),
                proprio_projector=pose_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, NUM_PATCHES:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )
        # official: the action-head call is explicitly wrapped in its OWN
        # torch.no_grad (run_forward_pass); without it torch>=2.x builds a
        # graph with the head's weights and `.numpy()` later raises
        # "Can't call numpy() on Tensor that requires grad" (v6 evidence).
        with torch.no_grad():
            predicted_actions = action_head.predict_action(
                actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))
        return predicted_actions, modality_id

    def decode_tick(actions):
        # official waypoint selection + PD controller + velocity limitation
        waypoints = actions.float().cpu().numpy()
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint
        EPS = 1e-8
        DT = 1 / 3
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT
        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)
        maxv, maxw = 0.3, 0.3
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        return waypoints, linear_vel_value_limit, angular_vel_value_limit

    def one_tick():
        t0 = time.perf_counter()
        batch = data_transformer_omnivla(current_image_PIL, lan_inst, goal_image_PIL,
                                         goal_pose_loc_norm)
        forward0 = time.perf_counter()
        actions, modality_id = run_forward_pass(batch)
        forward1 = time.perf_counter()
        waypoints, lin, ang = decode_tick(actions)
        tick_ms = (time.perf_counter() - t0) * 1000.0
        return tick_ms, (forward1 - forward0) * 1000.0, waypoints, lin, ang

    torch.cuda.reset_peak_memory_stats()
    warmups, ticks, forwards = [], [], []
    first_out = None
    for i in range(13):
        tick_ms, fwd_ms, waypoints, lin, ang = one_tick()
        if first_out is None:
            first_out = {"waypoints_shape": list(waypoints.shape),
                         "waypoints_first4_round3": waypoints[0][:4].round(3).tolist(),
                         "modality_id": "6 (image only; no satellite/lan/pose)",
                         "decoded_linear_mps": float(lin),
                         "decoded_angular_rads": float(ang)}
        if i < 3:
            warmups.append(tick_ms)
        else:
            ticks.append(tick_ms)
            forwards.append(fwd_ms)
    s = sorted(ticks)
    report["timing"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "warmup_3_ms": [round(x, 1) for x in warmups],
        "n": len(ticks),
        "mean_ms": round(statistics.mean(ticks), 1),
        "median_ms": round(statistics.median(ticks), 1),
        "p95_ms": round(s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)], 1),
        "min_ms": round(min(ticks), 1), "max_ms": round(max(ticks), 1),
        "forward_only_mean_ms": round(statistics.mean(forwards), 1),
        "tick_scope": "batch build (official data_transformer+collator) + forward + "
                      "action_head + PD decode; excludes figure saving",
        **first_out,
        "vram_peak_gb": [round(torch.cuda.max_memory_allocated(i) / 1e9, 2)
                         for i in range(torch.cuda.device_count())],
    }
    save()
    print("PROBE DONE")
except Exception as e:
    import traceback
    report["timing"] = {"error": traceback.format_exc()[-4000:]}
    save()
    raise
"""

# write the probe script to a plain file (avoids all nested-quoting hazards
# at runtime; the probe contains its own triple-quoted strings)
with open("/kaggle/tmp/preflight_probe.py", "w", encoding="utf-8") as f:
    f.write(PROBE)

import subprocess, sys, time
r = subprocess.run([sys.executable, "/kaggle/tmp/preflight_probe.py"],
                   capture_output=True, text=True, timeout=7200)
print("PROBE stdout:\\n", r.stdout[-8000:])
print("PROBE stderr:\\n", r.stderr[-6000:])
print("probe returncode:", r.returncode)

import json, os, time

path = "/kaggle/working/preflight/preflight_report.json"
report = json.load(open(path)) if os.path.exists(path) else {}
print(json.dumps(report, indent=2, default=str))

timing = report.get("timing") or {}
model_load = report.get("model_load") or {}
gates = {
    "constants_correct": bool(report.get("constants"))
                          and report.get("constants", {}).get("NUM_ACTIONS_CHUNK") == 8
                          and report.get("constants", {}).get("ACTION_DIM") == 4,
    "model_loaded": "seconds" in model_load,
    "inference_ran": "mean_ms" in timing,
    "vram_recorded": bool(timing.get("vram_peak_gb")),
}
print("\\nPREFLIGHT GATES:", json.dumps(gates, indent=2))
verdict = "ALL PASS" if all(gates.values()) else "*** GATES FAILED — see report ***"
print(verdict)
gate_doc = {"gates": gates, "verdict": verdict,
            "summary": {k: timing.get(k) for k in
                        ("mean_ms", "median_ms", "p95_ms", "vram_peak_gb",
                         "vram_after_load_gb", "torch", "transformers")}}
with open("/kaggle/working/preflight/preflight_report.json", "w") as f:
    merged = dict(report); merged.update(gate_doc)
    json.dump(merged, f, indent=2, default=str)
print("report saved to", path)
