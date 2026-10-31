"""Real-model adapter: OmniVLA (NHirose/omnivla-original, MIT) under the
`NavPolicy` contract.

This is the ONLY module allowed to be cited as VLA evidence (its rows are
logged with `is_model=True`). It replicates the official inference tick of
`inference/run_omnivla.py` (github.com/NHirose/OmniVLA, fetched 2026-10-28):

    batch build (PurePromptBuilder token template)
      -> VLA forward (modality-conditioned, bf16 autocast, no_grad)
      -> L1RegressionActionHead_idcat.predict_action on the prompt hidden
         states  ->  (8, 4) waypoint chunk (0.1 m units, robot frame)

Actuation happens ONLY through the shared official PD controller in
`environment/types.py` (decode/follow) — the adapter never decodes itself,
so B1/B2/B3 actuation is bit-identical machinery.

Zero-privileged guarantee
-------------------------
`bind_env()` is deliberately NOT overridden (inherits the no-op): the
policy never sees simulator state. For the language modality (id 7) the
official attention builder masks the goal-pose token out (verified in
`prismatic/extern/hf/modeling_prismatic.py::PrismaticForConditionalGeneration_MMNv1
._build_multimodal_attention_MMN`, 2026-10-28); we pass a neutral **zero**
proprio vector instead of any episode goal so that zero privileged
information can flow even under a masking regression.

Environment scope
-----------------
Heavy imports (torch, prismatic, transformers) happen inside `load()`, so
this module imports cleanly on CPU-only machines (tests, docs tooling).
The worker environment must be prepared exactly like the preflight probe
(notebook 00): `pip --no-deps transformers==4.40.1 huggingface_hub==0.29.1`
**plus the one-needle relaxation** of transformers'
`dependency_versions_table.py` (`tokenizers>=0.19,<0.20` → `>=0.19`), since
v3 proved the import-time gate is the only blocker on Kaggle's py3.13
(v3 kernel log evidence, 2026-10-28; see R11 addendum).
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np

from ..environment.types import (
    NUM_ACTIONS_CHUNK,
    Observation,
    WaypointChunk,
)
from .mock_policy import NavPolicy

#: Per-action dimensionality of the official OmniVLA chunk format
#: (dx, dy, cos(theta), sin(theta)). The repo-wide constant lives with the
#: model constants; the chunk tensor shape is (8, 4) everywhere.
ACTION_DIM = 4


class OmnivlaNavPolicy(NavPolicy):
    """Official-flow OmniVLA inference under the NavPolicy contract."""

    #: Logged as `is_model=True`; the one and only VLA-evidence class here.
    is_model: bool = True

    #: Official modality table (run_forward_pass): NO combination for
    #: `lan ∧ image_goal` exists — do not "invent" id 9. We support:
    #:   language  -> modality 7 (language only)
    MODALITY_ID = {"language": 7, "language_pose": 8}
    PROPRIEO_NEUTRAL = np.zeros(4, dtype=np.float32)  # (rel_y_norm, -rel_x_norm, cos, sin)

    def __init__(self, vla_path: str,
                 device_index: int = 0,
                 instruction_mode: str = "language",
                 goal_pose_fn=None,
                 template_seed: int = 0,
                 prismatic_repo: str = "/kaggle/tmp/OmniVLA",
                 load_now: bool = True) -> None:
        self.vla_path = vla_path
        self.prismatic_repo = prismatic_repo.rstrip("/")
        self.device_index = int(device_index)
        self.instruction_mode = instruction_mode
        #: modality 8 (language+pose) needs the goal-relative pose at query
        #  time: a callable -> (rel_x_front_m, rel_y_left_m, rel_heading_rad).
        #  The goal position is PRIVILEGED (simulator bookkeeping) — runs in
        #  this mode must be labelled as official modality 8, never as the
        #  onboard language-only deployment.
        self.goal_pose_fn = goal_pose_fn
        if instruction_mode == "language_pose" and goal_pose_fn is None:
            raise ValueError("instruction_mode='language_pose' requires "
                             "goal_pose_fn (official modality 8 input)")
        self._rng = np.random.default_rng(int(template_seed))  # template only; deterministic
        self.n_calls = 0
        self._loaded = False
        self._vlam: Optional[Any] = None        # prismatic VLA module
        self._action_head: Optional[Any] = None
        self._pose_projector: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._action_tokenizer: Optional[Any] = None
        self._num_patches: int = 0
        self._device: Optional[Any] = None      # torch.device
        self._torch: Optional[Any] = None       # torch module (kept for reuse)
        self.provenance: Dict[str, Any] = {"policy": "OmniVLA", "is_model": True}
        if load_now:
            self.load()

    # ------------------------------------------------------------------
    # Loading (official define_model() replication)
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load through the official `define_model()` route.

        Registers the model classes with HF Auto (installation of the
        prismatic package must already be importable; on Kaggle the preflight
        probe of notebook 00 does exactly this before any experiment).
        """
        try:
            import torch
        except ImportError as e:  # pragma: no cover - environment issue
            raise RuntimeError("OmniVLA needs torch; run on the GPU worker "
                               "(Kaggle notebook 01/02+), not on CPU-only hosts") from e
        import transformers
        import timm
        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise RuntimeError(f"timm {timm.__version__} is hard-rejected by the "
                               "model code; pin 0.9.10..0.9.16")

        # v5 evidence: eager prismatic package __init__ chain imports the RLDS
        # training stack (dlimp -> TensorFlow); register stub parent packages
        # so only inference-relevant modules load (surgical, documented, same
        # technique as the preflight probe). MUST run before `import prismatic.*`.
        _install_prismatic_stub_pkgs(self.prismatic_repo)
        from prismatic.vla.constants import (ACTION_DIM, NUM_ACTIONS_CHUNK,
                                             ACTION_PROPRIO_NORMALIZATION_TYPE)
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.models.projectors import ProprioProjector
        from prismatic.models.action_heads import L1RegressionActionHead_idcat
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.processing_prismatic import (PrismaticImageProcessor,
                                                              PrismaticProcessor)
        from transformers import (AutoConfig, AutoImageProcessor,
                                  AutoModelForVision2Seq, AutoProcessor)

        # official: register classes; official default constants are OmniVLA's
        assert NUM_ACTIONS_CHUNK == 8 and ACTION_DIM == 4, (
            f"prismatic constants drifted: chunk={NUM_ACTIONS_CHUNK} dim={ACTION_DIM}")
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

        t0 = time.perf_counter()
        # DEVIATION (documented): `device_map="auto"` may shard the 15.4 GB
        # bf16 weights across two 15.6 GB T4s; official does single-GPU `.to()`.
        processor = AutoProcessor.from_pretrained(self.vla_path)
        vla = AutoModelForVision2Seq.from_pretrained(
            self.vla_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map="auto",
        )
        vla.vision_backbone.set_num_images_in_input(2)
        vla.to(dtype=torch.bfloat16)

        def remove_ddp(sd: dict) -> dict:
            return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

        def load_checkpoint(module_name: str, step: int):
            path = f"{self.vla_path}/{module_name}--{step}_checkpoint.pt"
            state = torch.load(path, map_location="cpu")
            return remove_ddp(state)

        pose_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=4)
        pose_projector.load_state_dict(load_checkpoint("proprio_projector", 120000))
        pose_projector = pose_projector.to(torch.bfloat16)
        action_head = L1RegressionActionHead_idcat(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=4)
        action_head.load_state_dict(load_checkpoint("action_head", 120000))
        action_head = action_head.to(torch.bfloat16)

        device = torch.device(f"cuda:{self.device_index}" if torch.cuda.is_available()
                              else "cpu")
        pose_projector = pose_projector.to(device)
        action_head = action_head.to(device)
        # official: NUM_PATCHES = patches * num_images + 1  (+1 is the goal-pose token)
        num_patches = (vla.vision_backbone.get_num_patches()
                       * vla.vision_backbone.get_num_images_in_input() + 1)

        self._torch = torch
        self._vlam = vla.eval()
        self._action_head = action_head.eval()
        self._pose_projector = pose_projector.eval()
        self._processor = processor
        self._action_tokenizer = ActionTokenizer(processor.tokenizer)
        self._num_patches = int(num_patches)
        self._device = device
        self._loaded = True
        self.provenance.update({
            "model_name": "NHirose/omnivla-original",
            "model_checkpoint_step": 120000,
            "vla_path": self.vla_path,
            "num_patches": self._num_patches,
            "device_map": "auto",
            "load_seconds": round(time.perf_counter() - t0, 1),
            "llm_dim": vla.llm_dim,
        })

    @staticmethod
    def goal_proprio_vector(rel_x_front_m: float, rel_y_left_m: float,
                            rel_heading_rad: float) -> np.ndarray:
        """Official `run_omnivla` goal-pose vector for modality 8.

        Replicates: relative offset rotated into the robot frame (X=front,
        Y=left), clamped to a 30 m radius, divided by metric waypoint
        spacing 0.1 m, heading as cos/sin of (goal - current). Order:
        (rel_y_norm, -rel_x_norm, cos, sin) — matches PROPRIO_NEUTRAL.
        """
        cl = 30.0
        rx = float(np.clip(rel_x_front_m, -cl, cl))
        ry = float(np.clip(rel_y_left_m, -cl, cl))
        return np.array([ry / 0.1, -rx / 0.1,
                         math.cos(rel_heading_rad),
                         math.sin(rel_heading_rad)], dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference (official tick: batch build -> forward -> action head)
    # ------------------------------------------------------------------
    def _build_batch(self, image: np.ndarray, instruction: str):
        """Official `data_transformer_omnivla` (language branch) replication.

        The action-token template uses DETERMINISTIC dummy actions (official
        code uses `np.random.rand(8,4)`): the dummy token *values* are zeroed
        out by the model's own forward (`input_embeddings * ~all_actions_mask`),
        so only the token *positions* matter, which are value-independent.
        A fixed seed makes runs reproducible.
        """
        from PIL import Image as PILImage

        torch = self._torch
        processor = self._processor
        tokenizer = processor.tokenizer

        raw_actions = self._rng.random((8, 4)) * 2 - 1  # bounded; any values work
        current_str = self._action_tokenizer(raw_actions[0])
        future_str = "".join(self._action_tokenizer(raw_actions[1:]))
        action_chunk_string = current_str + future_str
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value":
                f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        prompt_builder = PurePromptBuilderLocal()
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = torch.tensor(tokenizer(prompt_builder.get_prompt(),
                                           add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = -100  # official IGNORE_INDEX

        image_transform = processor.image_processor.apply_transform
        pixel_values_current = image_transform(PILImage.fromarray(image).convert("RGB"))
        # second image slot is the goal image; the language modality masks it
        pixel_values_goal = image_transform(PILImage.new("RGB", (224, 224), (0, 0, 0)))

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        # v3 evidence: the model's forward does `input_ids.shape[1]` (2-D
        # required); the official collator pads a batch of 1. Probe used
        # pad_sequence identically — replicate it here.
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence([input_ids], batch_first=True,
                                 padding_value=pad_token_id)
        labels = pad_sequence([labels], batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(pad_token_id)
        pixel_values = torch.cat(
            (torch.stack([pixel_values_current]), torch.stack([pixel_values_goal])), dim=1)
        return dict(pixel_values=pixel_values, input_ids=input_ids,
                    attention_mask=attention_mask, labels=labels)

    def infer(self, obs: Observation, instruction: str) -> WaypointChunk:
        assert self._loaded, "call load() (or construct with load_now=True) first"
        assert instruction and instruction.strip(), \
            "empty instruction: the language modality must be conditioned on it"
        self.n_calls += 1
        torch = self._torch
        device = self._device

        batch = self._build_batch(obs.image, instruction.strip())
        modality_id = torch.as_tensor(
            [self.MODALITY_ID[self.instruction_mode]], dtype=torch.float32)
        if self.instruction_mode == "language_pose":
            if self.goal_pose_fn is None:
                raise ValueError("language_pose needs goal_pose_fn")
            proprio_vec = self.goal_proprio_vector(*self.goal_pose_fn())
        else:
            proprio_vec = self.PROPRIEO_NEUTRAL
        proprio = torch.as_tensor(proprio_vec, dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self._vlam(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                modality_id=modality_id.to(torch.bfloat16).to(device),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                proprio=proprio.to(torch.bfloat16).to(device),
                proprio_projector=self._pose_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )
            ground_truth_token_ids = batch["labels"][:, 1:].to(device)
            current_mask = get_current_action_mask_local(ground_truth_token_ids)
            next_mask = get_next_actions_mask_local(ground_truth_token_ids)
            last_hidden = output.hidden_states[-1]
            text_hidden = last_hidden[:, self._num_patches:-1]
            actions_hidden = (
                text_hidden[current_mask | next_mask]
                .reshape(batch["input_ids"].shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(torch.bfloat16)
                # sharded model: hidden states may live on the last shard's
                # GPU; the head is on self._device -> align (no-op if equal)
                .to(self._device)
            )
            predicted = self._action_head.predict_action(
                actions_hidden, modality_id.to(torch.bfloat16).to(device))

        # v6 evidence: without no_grad around the head call (official wraps it
        # separately in run_forward_pass) the output requires grad and
        # `.numpy()` raises; here no_grad covers everything, + detach anyway.
        waypoints = predicted.detach().float().cpu().numpy()  # (1, 8, 4)
        return WaypointChunk(np.asarray(waypoints[0]))
        # NOTE: decoding to Cmd happens ONLY in environment/types.py (official PD).

    # ------------------------------------------------------------------
    def provenance_summary(self) -> Dict[str, Any]:
        out = dict(self.provenance)
        out["n_inference_calls"] = self.n_calls
        out["instruction_mode"] = self.instruction_mode
        return out


# ----------------------------------------------------------------------
# Tiny local shims (mask helpers need torch on the worker only)
# ----------------------------------------------------------------------
class PurePromptBuilderLocal:
    """`PurePromptBuilder` shim with the same interface used by the official
    script (`prompt_builder("<alias>"|system)`, add_turn, get_prompt).

    Imported lazily so this module imports without prismatic installed.
    """

    def __init__(self, *args, **kwargs) -> None:
        try:
            from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        except ImportError:  # pragma: no cover
            raise RuntimeError("prismatic package not importable; clone the "
                               "official OmniVLA repo and add it to sys.path")
        self._inst = PurePromptBuilder("openvla")

    def add_turn(self, role: str, message: str) -> None:
        self._inst.add_turn(role, message)

    def get_prompt(self) -> str:
        return self._inst.get_prompt()


def get_current_action_mask_local(token_ids):
    """Official `prismatic.training.train_utils.get_current_action_mask`."""
    from prismatic.training.train_utils import get_current_action_mask
    return get_current_action_mask(token_ids)


def get_next_actions_mask_local(token_ids):
    """Official `prismatic.training.train_utils.get_next_actions_mask`."""
    from prismatic.training.train_utils import get_next_actions_mask
    return get_next_actions_mask(token_ids)


def _install_prismatic_stub_pkgs(repo_root: str) -> list:
    """Register stub parent packages for `prismatic` in sys.modules.

    v5 preflight evidence (2026-10-28): importing any `prismatic.*` module
    executes the eager `__init__` chain down to
    `prismatic/vla/datasets/rlds/dataset.py -> import dlimp` (which requires
    TensorFlow). Every module the OFFICIAL INFERENCE needs
    (`vla.constants`, `vla.action_tokenizer`, `models.projectors`,
    `models.action_heads`, `models.backbones.llm.prompting`,
    `training.train_utils`, `extern.hf.*`) is a plain leaf module, so stub
    parent packages (correct `__path__`) let Python load those sources
    verbatim while skipping only the eager training-stack inits.
    """
    import sys
    import types

    chain = ["prismatic", "prismatic.vla", "prismatic.models",
             "prismatic.models.backbones", "prismatic.models.backbones.llm",
             "prismatic.training", "prismatic.extern", "prismatic.extern.hf"]
    repo_root = repo_root.rstrip("/")
    for name in chain:
        if name in sys.modules:
            continue
        path = f"{repo_root}/{name.replace('.', '/')}"
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        mod.__package__ = name
        sys.modules[name] = mod
    return chain
