from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hashing import sha256_hex
from .receipts import ReceiptVerifier
from .transforms import apply_transform, validate_transform


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str


class AttestedTransitionGate:
    """Admission gate for learner-visible transition records."""

    def __init__(self, receipt_verifier: ReceiptVerifier | None = None) -> None:
        self.receipt_verifier = receipt_verifier

    def admit_replay(
        self,
        learner_record: dict[str, Any],
        replay_record: dict[str, Any],
        transform_spec: dict[str, Any],
    ) -> GateDecision:
        decision = validate_transform(transform_spec)
        if not decision.accepted:
            return GateDecision(False, decision.reason)
        expected_action = apply_transform(learner_record["submitted_action"], transform_spec)
        executed_action = np.asarray(replay_record["executed_action"], dtype=np.float64)
        if expected_action.shape != executed_action.shape or not np.allclose(expected_action, executed_action, atol=1e-8, rtol=0.0):
            return GateDecision(False, "certified_action_mismatch")
        for field in ["pre_state", "post_state", "observation", "next_observation", "terminated", "truncated", "reward_input"]:
            if sha256_hex(learner_record.get(field)) != sha256_hex(replay_record.get(field)):
                return GateDecision(False, f"{field}_mismatch")
        return GateDecision(True, "accepted")

    def admit_receipt(self, receipt: dict[str, Any], expected_stream: str | None = None, expected_lane: int | None = None) -> GateDecision:
        if self.receipt_verifier is None:
            return GateDecision(False, "receipt_verifier_missing")
        result = self.receipt_verifier.verify(receipt, expected_stream=expected_stream, expected_lane=expected_lane)
        return GateDecision(result.accepted, result.reason)

