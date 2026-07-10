from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .hashing import canonical_bytes, sha256_hex

ZERO_HEAD = "0" * 64


def public_key_hex(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    if isinstance(key, Ed25519PrivateKey):
        key = key.public_key()
    return key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw).hex()


def sign_record(key: Ed25519PrivateKey, payload: dict[str, Any]) -> str:
    return key.sign(canonical_bytes(payload)).hex()


def verify_signature(public_key: Ed25519PublicKey, payload: dict[str, Any], signature_hex: str) -> bool:
    try:
        public_key.verify(bytes.fromhex(signature_hex), canonical_bytes(payload))
        return True
    except (InvalidSignature, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class VerifyResult:
    accepted: bool
    reason: str


class EnrollmentVerifier:
    """Binds receipt keys to the measured step-service closure."""

    def __init__(self, launcher_public_key: Ed25519PublicKey, expected: dict[str, str]) -> None:
        self.launcher_public_key = launcher_public_key
        self.expected = dict(expected)
        self.seen_launch_nonces: set[str] = set()
        self.highest_epoch: dict[str, int] = {}
        self.keys: dict[tuple[str, int], str] = {}

    def verify(self, record: dict[str, Any]) -> VerifyResult:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return VerifyResult(False, "missing_payload")
        if not verify_signature(self.launcher_public_key, payload, str(record.get("signature", ""))):
            return VerifyResult(False, "enrollment_signature_mismatch")
        for field, expected_value in self.expected.items():
            if payload.get(field) != expected_value:
                return VerifyResult(False, f"{field}_mismatch")
        service_id = str(payload.get("service_id", ""))
        key = str(payload.get("receipt_public_key", ""))
        nonce = str(payload.get("launch_nonce", ""))
        epoch = int(payload.get("key_epoch", -1))
        if not service_id or not key:
            return VerifyResult(False, "missing_service_or_key")
        if not nonce or nonce in self.seen_launch_nonces:
            return VerifyResult(False, "launch_nonce_replay")
        if epoch < self.highest_epoch.get(service_id, -1):
            return VerifyResult(False, "epoch_rollback")
        previous = self.keys.get((service_id, epoch))
        if previous is not None and previous != key:
            return VerifyResult(False, "same_epoch_key_change")
        self.seen_launch_nonces.add(nonce)
        self.highest_epoch[service_id] = max(epoch, self.highest_epoch.get(service_id, -1))
        self.keys[(service_id, epoch)] = key
        return VerifyResult(True, "accepted")

    def key_is_enrolled(self, service_id: str, epoch: int, public_key_hex_value: str) -> bool:
        return self.keys.get((service_id, int(epoch))) == public_key_hex_value


class ReceiptVerifier:
    """Checks transition receipts before learner admission."""

    def __init__(
        self,
        enrollment: EnrollmentVerifier,
        public_keys: dict[str, Ed25519PublicKey],
        expected_closure_hash: str,
        expected_policy_hash: str,
    ) -> None:
        self.enrollment = enrollment
        self.public_keys = public_keys
        self.expected_closure_hash = expected_closure_hash
        self.expected_policy_hash = expected_policy_hash
        self.stream_state: dict[tuple[str, int], tuple[int, str]] = {}
        self.seen_transition_nonces: set[str] = set()

    def verify(self, record: dict[str, Any], expected_stream: str | None = None, expected_lane: int | None = None) -> VerifyResult:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return VerifyResult(False, "missing_payload")
        service_id = str(payload.get("service_id", ""))
        epoch = int(payload.get("key_epoch", -1))
        key_hex = str(payload.get("receipt_public_key", ""))
        if not self.enrollment.key_is_enrolled(service_id, epoch, key_hex):
            return VerifyResult(False, "receipt_key_not_enrolled")
        public_key = self.public_keys.get(key_hex)
        if public_key is None:
            return VerifyResult(False, "public_key_unknown")
        if not verify_signature(public_key, payload, str(record.get("signature", ""))):
            return VerifyResult(False, "receipt_signature_mismatch")
        if payload.get("closure_hash") != self.expected_closure_hash:
            return VerifyResult(False, "closure_hash_mismatch")
        if payload.get("verifier_policy_hash") != self.expected_policy_hash:
            return VerifyResult(False, "policy_hash_mismatch")

        stream = str(payload.get("stream_id", ""))
        lane = int(payload.get("lane_id", -1))
        if expected_stream is not None and stream != expected_stream:
            return VerifyResult(False, "stream_context_mismatch")
        if expected_lane is not None and lane != expected_lane:
            return VerifyResult(False, "lane_context_mismatch")
        nonce = str(payload.get("transition_nonce", ""))
        if not nonce or nonce in self.seen_transition_nonces:
            return VerifyResult(False, "transition_nonce_replay")
        sequence = int(payload.get("sequence", -1))
        expected_sequence, expected_head = self.stream_state.get((stream, lane), (0, ZERO_HEAD))
        if sequence != expected_sequence:
            return VerifyResult(False, "sequence_mismatch")
        if payload.get("prev_head") != expected_head:
            return VerifyResult(False, "prev_head_mismatch")
        if not str(payload.get("transition_hash", "")):
            return VerifyResult(False, "transition_hash_missing")

        new_head = sha256_hex({"prev": expected_head, "payload": payload, "signature": record.get("signature", "")})
        self.stream_state[(stream, lane)] = (expected_sequence + 1, new_head)
        self.seen_transition_nonces.add(nonce)
        return VerifyResult(True, "accepted")


def make_enrollment(
    launcher_key: Ed25519PrivateKey,
    receipt_public_key: str,
    expected: dict[str, str],
    service_id: str,
    launch_nonce: str,
    key_epoch: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "version": "ate-enrollment-v1",
        "service_id": service_id,
        "receipt_public_key": receipt_public_key,
        "launch_nonce": launch_nonce,
        "key_epoch": int(key_epoch),
        **expected,
    }
    if overrides:
        payload.update(overrides)
    return {"payload": payload, "signature": sign_record(launcher_key, payload)}


def make_receipt(
    receipt_key: Ed25519PrivateKey,
    *,
    service_id: str,
    key_epoch: int,
    stream_id: str,
    lane_id: int,
    sequence: int,
    transition_nonce: str,
    prev_head: str,
    transition_hash: str,
    closure_hash: str,
    policy_hash: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "version": "ate-transition-receipt-v1",
        "service_id": service_id,
        "receipt_public_key": public_key_hex(receipt_key),
        "key_epoch": int(key_epoch),
        "stream_id": stream_id,
        "lane_id": int(lane_id),
        "sequence": int(sequence),
        "transition_nonce": transition_nonce,
        "prev_head": prev_head,
        "transition_hash": transition_hash,
        "closure_hash": closure_hash,
        "verifier_policy_hash": policy_hash,
    }
    if overrides:
        payload.update(overrides)
    return {"payload": payload, "signature": sign_record(receipt_key, payload)}

