"""Experiment logging: JSONL writer with full provenance and immutability.

Rule: every experiment outputs `results/raw/<exp_id>/episodes.jsonl` plus a
`meta.json` snapshot; the writer refuses to overwrite an existing
experiment's directory unless `overwrite=True` is passed EXPLICITLY at the
callsite (silent loss of evidence is disallowed).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentMeta:
    experiment_id: str
    created_utc: str = ""
    method: str = ""
    env_family: str = ""
    env_commit_hash: str = "unknown"
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    gpu_name: Optional[str] = None
    torch_version: Optional[str] = None
    python_version: Optional[str] = None
    seeds_version: str = "v1"
    conditions: List[str] = field(default_factory=list)
    split: str = "test"
    action_budget: int = -1
    termination_rule: str = "auto-stop within goal_radius, else max_steps; collision-kill at max_collisions"
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class EpisodeLogger:
    """Writes one JSONL row per finished episode with meta attached."""

    def __init__(self, meta: ExperimentMeta, out_dir: str, *, overwrite: bool = False) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.meta = meta
        self.path = os.path.join(out_dir, "episodes.jsonl")
        self.meta_path = os.path.join(out_dir, "meta.json")
        if os.path.exists(self.path) and not overwrite:
            raise FileExistsError(
                f"{self.path} exists; refusing to overwrite without overwrite=True "
                "to avoid silently destroying evidence."
            )
        meta.created_utc = meta.created_utc or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)
        # clear stale file so append semantics are unambiguous
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("")

    def log_episode(self, result: Any, step_log: Optional[Dict] = None) -> int:
        row = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
        row["meta"] = {
            "experiment_id": self.meta.experiment_id,
            "model_name": self.meta.model_name,
            "method": self.meta.method,
            "env_family": self.meta.env_family,
        }
        try:
            import numpy as _np
            row = {k: (v.item() if isinstance(v, _np.generic) else v) for k, v in row.items()}
        except Exception:
            pass
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            if step_log is not None:
                f.write(json.dumps({"type": "steps", "data": step_log}, default=str) + "\n")
        return 1
