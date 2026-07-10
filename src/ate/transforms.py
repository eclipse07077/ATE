from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransformDecision:
    accepted: bool
    reason: str


def validate_transform(spec: dict[str, Any]) -> TransformDecision:
    """Validate the declarative actuator transform policy."""
    if spec.get("kind") != "declarative":
        return TransformDecision(False, "not_declarative")
    if spec.get("arbitrary_code"):
        return TransformDecision(False, "arbitrary_code_forbidden")
    if spec.get("trigger_conditioned_branch"):
        return TransformDecision(False, "trigger_branch_forbidden")
    if spec.get("state_predicate") not in {None, "none"}:
        return TransformDecision(False, "state_predicate_forbidden")

    ops = spec.get("ops", [])
    allowed = {"identity", "clip", "scale", "action_repeat", "torque_saturation", "linear_safety_projection"}
    if not isinstance(ops, list) or not ops:
        return TransformDecision(False, "ops_missing")
    if any(op not in allowed for op in ops):
        return TransformDecision(False, "op_not_allowed")
    if "scale" in ops and not 0.5 <= float(spec.get("scale", 1.0)) <= 1.5:
        return TransformDecision(False, "scale_out_of_bounds")
    if "action_repeat" in ops and not 1 <= int(spec.get("repeat", 1)) <= 8:
        return TransformDecision(False, "repeat_out_of_bounds")
    if "torque_saturation" in ops and bool(spec.get("asymmetric_saturation", False)):
        return TransformDecision(False, "asymmetric_saturation_forbidden")
    if "linear_safety_projection" in ops and bool(spec.get("target_action_bias", False)):
        return TransformDecision(False, "target_bias_forbidden")
    return TransformDecision(True, "accepted")


def apply_transform(action: Any, spec: dict[str, Any]) -> np.ndarray:
    decision = validate_transform(spec)
    if not decision.accepted:
        raise ValueError(decision.reason)
    out = np.asarray(action, dtype=np.float64).copy()
    for op in spec.get("ops", []):
        if op == "identity":
            continue
        if op == "clip":
            low, high = spec.get("clip", [-1.0, 1.0])
            out = np.clip(out, float(low), float(high))
        elif op == "scale":
            out = out * float(spec.get("scale", 1.0))
        elif op == "torque_saturation":
            limit = abs(float(spec.get("torque_limit", 1.0)))
            out = np.clip(out, -limit, limit)
        elif op == "linear_safety_projection":
            max_norm = float(spec.get("max_norm", 1.0))
            norm = float(np.linalg.norm(out))
            if norm > max_norm > 0:
                out = out * (max_norm / norm)
        elif op == "action_repeat":
            continue
    return out

