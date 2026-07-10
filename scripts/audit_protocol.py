#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ate.hashing import sha256_hex
from ate.receipts import (
    EnrollmentVerifier,
    ReceiptVerifier,
    ZERO_HEAD,
    make_enrollment,
    make_receipt,
    public_key_hex,
)
from ate.transforms import validate_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/local/protocol_audit"))
    return parser.parse_args()


def add(rows: list[dict[str, object]], topic: str, case: str, expected_accept: bool, observed_accept: bool, reason: str) -> None:
    rows.append(
        {
            "topic": topic,
            "case": case,
            "expected_accept": expected_accept,
            "observed_accept": observed_accept,
            "passed": expected_accept == observed_accept,
            "reason": reason,
        }
    )


def run_audit() -> list[dict[str, object]]:
    launcher = Ed25519PrivateKey.generate()
    bad_launcher = Ed25519PrivateKey.generate()
    service = Ed25519PrivateKey.generate()
    rogue = Ed25519PrivateKey.generate()
    expected = {
        "service_code_hash": sha256_hex("step-service"),
        "dependency_hash": sha256_hex("dependencies"),
        "closure_hash": sha256_hex("approved-closure"),
        "verifier_policy_hash": sha256_hex("policy"),
        "socket_identity_hash": sha256_hex("private-socket"),
    }

    rows: list[dict[str, object]] = []
    enrollment = EnrollmentVerifier(launcher.public_key(), expected)
    clean = make_enrollment(launcher, public_key_hex(service), expected, "ate-service", "launch-0", 0)
    result = enrollment.verify(clean)
    add(rows, "key_binding", "approved_enrollment", True, result.accepted, result.reason)

    wrong_signer = make_enrollment(bad_launcher, public_key_hex(service), expected, "ate-service", "launch-1", 1)
    result = enrollment.verify(wrong_signer)
    add(rows, "key_binding", "wrong_launcher_signature", False, result.accepted, result.reason)

    drift = make_enrollment(
        launcher,
        public_key_hex(service),
        expected,
        "ate-service",
        "launch-2",
        1,
        overrides={"closure_hash": sha256_hex("tampered-closure")},
    )
    result = enrollment.verify(drift)
    add(rows, "key_binding", "closure_hash_drift", False, result.accepted, result.reason)

    replay = make_enrollment(launcher, public_key_hex(service), expected, "ate-service", "launch-0", 1)
    result = enrollment.verify(replay)
    add(rows, "key_binding", "launch_nonce_replay", False, result.accepted, result.reason)

    receipt_verifier = ReceiptVerifier(
        enrollment,
        {public_key_hex(service): service.public_key(), public_key_hex(rogue): rogue.public_key()},
        expected["closure_hash"],
        expected["verifier_policy_hash"],
    )
    r0 = make_receipt(
        service,
        service_id="ate-service",
        key_epoch=0,
        stream_id="env0",
        lane_id=0,
        sequence=0,
        transition_nonce="t0",
        prev_head=ZERO_HEAD,
        transition_hash=sha256_hex({"s": 0, "a": 1, "s2": 1}),
        closure_hash=expected["closure_hash"],
        policy_hash=expected["verifier_policy_hash"],
    )
    result = receipt_verifier.verify(r0, expected_stream="env0", expected_lane=0)
    add(rows, "receipt", "valid_receipt", True, result.accepted, result.reason)

    edited = json.loads(json.dumps(r0))
    edited["payload"]["transition_hash"] = sha256_hex("edited")
    result = receipt_verifier.verify(edited, expected_stream="env0", expected_lane=0)
    add(rows, "receipt", "edited_payload_old_signature", False, result.accepted, result.reason)

    result = receipt_verifier.verify(r0, expected_stream="env0", expected_lane=0)
    add(rows, "receipt", "stale_receipt_replay", False, result.accepted, result.reason)

    bad_lane = make_receipt(
        service,
        service_id="ate-service",
        key_epoch=0,
        stream_id="env0",
        lane_id=1,
        sequence=0,
        transition_nonce="lane1",
        prev_head=ZERO_HEAD,
        transition_hash=sha256_hex({"lane": 1}),
        closure_hash=expected["closure_hash"],
        policy_hash=expected["verifier_policy_hash"],
    )
    result = receipt_verifier.verify(bad_lane, expected_stream="env0", expected_lane=0)
    add(rows, "receipt", "lane_context_mismatch", False, result.accepted, result.reason)

    transform_cases = [
        ("clip", {"kind": "declarative", "ops": ["clip"], "clip": [-1.0, 1.0]}, True),
        ("scale", {"kind": "declarative", "ops": ["scale"], "scale": 1.1}, True),
        ("action_repeat", {"kind": "declarative", "ops": ["action_repeat"], "repeat": 4}, True),
        ("trigger_branch", {"kind": "declarative", "ops": ["identity"], "trigger_conditioned_branch": True}, False),
        ("python_callback", {"kind": "python", "ops": ["identity"], "arbitrary_code": True}, False),
        ("target_biased_safety", {"kind": "declarative", "ops": ["linear_safety_projection"], "target_action_bias": True}, False),
    ]
    for name, spec, expected_accept in transform_cases:
        decision = validate_transform(spec)
        add(rows, "transform", name, expected_accept, decision.accepted, decision.reason)
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_audit()
    failures = [row for row in rows if not row["passed"]]
    with (args.output_dir / "protocol_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"rows": len(rows), "failures": len(failures), "failure_cases": [row["case"] for row in failures]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
