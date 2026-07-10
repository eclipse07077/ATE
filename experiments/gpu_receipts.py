#!/usr/bin/env python3
"""GPU tensor batch receipt stress for ATE.

The benchmark binds GPU-resident vector-step arrays into host-verifiable batch
receipts and checks that wrapper tamper/replay attempts fail receipt
verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KEY = b"ate-gpu-batch-receipt-stress-key"
WRONG_KEY = b"ate-gpu-batch-receipt-wrong-key"


@dataclass(frozen=True)
class Config:
    lanes: int
    steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--configs", default="1024x1024,4096x512,16384x128")
    parser.add_argument("--probe-steps", type=int, default=64)
    return parser.parse_args()


def parse_configs(raw: str) -> list[Config]:
    configs: list[Config] = []
    for item in raw.split(","):
        lanes_s, steps_s = item.lower().split("x", 1)
        configs.append(Config(lanes=int(lanes_s), steps=int(steps_s)))
    return configs


def digest_array(arr: Any) -> bytes:
    import numpy as np

    value = np.ascontiguousarray(arr)
    header = f"{value.dtype}:{','.join(map(str, value.shape))}".encode("ascii")
    h = hashlib.sha256()
    h.update(len(header).to_bytes(2, "big"))
    h.update(header)
    h.update(value.view(np.uint8))
    return h.digest()


def make_payload(step: int, pre: Any, action: Any, post: Any, reward: Any, done: Any) -> bytes:
    return b"ATEGPU2" + b"".join(
        [
            int(step).to_bytes(8, "big"),
            digest_array(pre),
            digest_array(action),
            digest_array(post),
            digest_array(reward),
            digest_array(done),
        ]
    )


def sign(head: bytes, nonce: bytes, payload: bytes, key: bytes = KEY) -> tuple[bytes, bytes]:
    envelope = head + nonce + payload
    receipt = hmac.new(key, envelope, hashlib.sha256).digest()
    next_head = hashlib.sha256(envelope + receipt).digest()
    return receipt, next_head


def verify(head: bytes, nonce: bytes, payload: bytes, receipt: bytes, key: bytes = KEY) -> bool:
    expected, _ = sign(head, nonce, payload, key=key)
    return hmac.compare_digest(expected, receipt)


def write_skip_summary(args: argparse.Namespace, reason: str) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_dir": str(args.output_dir),
        "skipped": True,
        "reason": reason,
        "full_simulator_benchmark": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        "# GPU Receipt Benchmark\n\n"
        "Run skipped.\n\n"
        f"- Reason: {reason}\n",
        encoding="utf-8",
    )


def run_config(args: argparse.Namespace, cfg: Config, torch: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    device = torch.device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(3407 + cfg.lanes + cfg.steps)

    lanes = cfg.lanes
    obs_dim = args.obs_dim
    action_dim = args.action_dim
    records = lanes * cfg.steps
    probe_steps = min(args.probe_steps, cfg.steps)

    obs = torch.randn((lanes, obs_dim), device=device, generator=gen)
    weight = torch.randn((action_dim, obs_dim), device=device, generator=gen) / (obs_dim**0.5)
    for _ in range(args.warmup_steps):
        actions = torch.randn((lanes, action_dim), device=device, generator=gen)
        obs = torch.tanh(obs + actions @ weight)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    raw = obs
    start_raw = time.perf_counter()
    for _ in range(cfg.steps):
        actions = torch.randn((lanes, action_dim), device=device, generator=gen)
        raw = torch.tanh(raw + actions @ weight)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    raw_seconds = time.perf_counter() - start_raw

    obs = torch.randn((lanes, obs_dim), device=device, generator=gen)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    head = bytes(32)
    receipt_bytes = 0
    accepted = 0
    false_rejects = 0
    tamper_rows: list[dict[str, Any]] = []
    start_receipt = time.perf_counter()

    previous = None
    for step in range(cfg.steps):
        pre = obs
        actions = torch.randn((lanes, action_dim), device=device, generator=gen)
        post = torch.tanh(pre + actions @ weight)
        rewards = -torch.sum(post * post, dim=1)
        done = rewards < -float(obs_dim)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        pre_np = pre.detach().cpu().numpy()
        act_np = actions.detach().cpu().numpy()
        post_np = post.detach().cpu().numpy()
        rew_np = rewards.detach().cpu().numpy()
        done_np = done.detach().cpu().numpy()

        payload = make_payload(step, pre_np, act_np, post_np, rew_np, done_np)
        nonce = hashlib.sha256(f"gpu-batch:{lanes}:{step}".encode("ascii")).digest()
        receipt, next_head = sign(head, nonce, payload)
        if verify(head, nonce, payload, receipt):
            accepted += lanes
        else:
            false_rejects += lanes

        if step < probe_steps:
            cases: list[tuple[str, bytes, bytes, bytes, bytes]] = []

            edited_post = post_np.copy()
            edited_post[0, 0] += np.float32(0.25)
            cases.append(("transition_edit", nonce, make_payload(step, pre_np, act_np, edited_post, rew_np, done_np), receipt, head))

            swapped_post = post_np.copy()
            swapped_post[[0, 1]] = swapped_post[[1, 0]]
            cases.append(("lane_swap", nonce, make_payload(step, pre_np, act_np, swapped_post, rew_np, done_np), receipt, head))

            flipped_done = done_np.copy()
            flipped_done[0] = not bool(flipped_done[0])
            cases.append(("done_flip", nonce, make_payload(step, pre_np, act_np, post_np, rew_np, flipped_done), receipt, head))

            eps_post = post_np.copy()
            eps_post[:, 0] += np.float32(1e-6)
            eps_payload = make_payload(step, pre_np, act_np, eps_post, rew_np, done_np)
            cases.append(("epsilon_bias_below_1e-5_tolerance", nonce, eps_payload, receipt, head))

            wrong_receipt, _ = sign(head, nonce, payload, key=WRONG_KEY)
            cases.append(("wrong_key_receipt", nonce, payload, wrong_receipt, head))

            if previous is not None:
                _prev_head, prev_nonce, prev_payload, prev_receipt = previous
                cases.append(("stale_receipt_replay", nonce, payload, prev_receipt, head))
                cases.append(("old_sequence_payload_on_current_head", prev_nonce, prev_payload, prev_receipt, head))

            for case, case_nonce, case_payload, case_receipt, case_head in cases:
                accepted_case = verify(case_head, case_nonce, case_payload, case_receipt)
                naive_tolerance_accept = case == "epsilon_bias_below_1e-5_tolerance"
                tamper_rows.append(
                    {
                        "lanes": lanes,
                        "step": step,
                        "case": case,
                        "accepted_by_receipt": accepted_case,
                        "rejected_by_receipt": not accepted_case,
                        "naive_tolerance_would_accept": naive_tolerance_accept,
                    }
                )

        previous = (head, nonce, payload, receipt)
        head = next_head
        receipt_bytes += len(nonce) + len(payload) + len(receipt) + len(head)
        obs = post

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    receipt_seconds = time.perf_counter() - start_receipt

    row = {
        "lanes": lanes,
        "steps": cfg.steps,
        "records": records,
        "raw_gpu_update_seconds": raw_seconds,
        "receipt_sync_seconds": receipt_seconds,
        "raw_gpu_update_us_per_record": 1e6 * raw_seconds / records,
        "receipt_sync_us_per_record": 1e6 * receipt_seconds / records,
        "receipt_sync_slowdown_vs_raw_gpu_update": receipt_seconds / max(raw_seconds, 1e-12),
        "receipt_overhead_bytes_per_record": receipt_bytes / records,
        "accepted_records": accepted,
        "false_rejects": false_rejects,
        "tamper_probes": len(tamper_rows),
        "tamper_receipt_accepts": sum(1 for r in tamper_rows if r["accepted_by_receipt"]),
        "epsilon_bias_naive_tolerance_accepts": sum(1 for r in tamper_rows if r["naive_tolerance_would_accept"]),
    }
    return row, tamper_rows


def main() -> int:
    args = parse_args()
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        write_skip_summary(args, f"torch unavailable: {type(exc).__name__}")
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        write_skip_summary(args, "cuda unavailable")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)
    rows: list[dict[str, Any]] = []
    tamper_rows: list[dict[str, Any]] = []
    for cfg in configs:
        row, tamper = run_config(args, cfg, torch)
        rows.append(row)
        tamper_rows.extend(tamper)

    with (args.output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "tamper_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(tamper_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tamper_rows)

    summary = {
        "output_dir": str(args.output_dir),
        "skipped": False,
        "full_simulator_benchmark": False,
        "device": args.device,
        "torch_version": torch.__version__,
        "configs": [row["lanes"] for row in rows],
        "total_records": sum(row["records"] for row in rows),
        "max_receipt_sync_us_per_record": max(row["receipt_sync_us_per_record"] for row in rows),
        "min_receipt_sync_us_per_record": min(row["receipt_sync_us_per_record"] for row in rows),
        "max_receipt_overhead_bytes_per_record": max(row["receipt_overhead_bytes_per_record"] for row in rows),
        "false_rejects": sum(row["false_rejects"] for row in rows),
        "tamper_probes": len(tamper_rows),
        "tamper_receipt_accepts": sum(1 for r in tamper_rows if r["accepted_by_receipt"]),
        "epsilon_bias_naive_tolerance_accepts": sum(1 for r in tamper_rows if r["naive_tolerance_would_accept"]),
        "interpretation": "GPU tensor receipt benchmark.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# GPU Receipt Benchmark",
        "",
        "Synthetic PyTorch CUDA tensor vector-step benchmark with ATE batch receipts.",
        "This benchmark measures the tensor receipt path rather than a full simulator.",
        "",
        f"- Device: `{args.device}`",
        f"- Torch: `{torch.__version__}`",
        f"- Total records: {summary['total_records']}",
        f"- Receipt sync us/record range: {summary['min_receipt_sync_us_per_record']:.3f}--{summary['max_receipt_sync_us_per_record']:.3f}",
        f"- Max receipt metadata bytes/record: {summary['max_receipt_overhead_bytes_per_record']:.3f}",
        f"- False rejects: {summary['false_rejects']}",
        f"- Tamper probes: {summary['tamper_probes']}",
        f"- Tamper receipts accepted: {summary['tamper_receipt_accepts']}",
        f"- Epsilon-bias cases that a naive 1e-5 tolerance would accept: {summary['epsilon_bias_naive_tolerance_accepts']}",
        "",
        "Rows:",
        "",
        "| Lanes | Steps | Records | Raw us/rec | Receipt us/rec | Slowdown vs raw tensor | Bytes/rec | False rejects | Tamper accepts |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {lanes} | {steps} | {records} | {raw_gpu_update_us_per_record:.4f} | "
            "{receipt_sync_us_per_record:.4f} | {receipt_sync_slowdown_vs_raw_gpu_update:.2f}x | "
            "{receipt_overhead_bytes_per_record:.4f} | {false_rejects} | {tamper_receipt_accepts} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "The receipt path rejects transition edits, lane swaps, done flips, stale receipts, wrong-key receipts, and below-tolerance epsilon-bias edits.",
            "The epsilon-bias row is the important security point: a naive tolerance verifier would accept these tiny edits, but receipt provenance rejects them because the learner-visible record hash changes.",
            "A full GPU simulator service would need to bind the simulator step itself, not only the tensor record path.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
