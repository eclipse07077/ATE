#!/usr/bin/env python3
"""Brax GPU simulator batch receipt benchmark for ATE.

This benchmark uses real Brax JAX simulator environments on GPU. It is still a
systems benchmark rather than a learning/backdoor result: it measures whether a
GPU batched simulator can emit learner-visible transition records that are bound
by ATE-style batch receipts, and whether wrapper tamper/replay attempts are
rejected before learner admission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any


KEY = b"ate-brax-gpu-receipt-key"
WRONG_KEY = b"ate-brax-gpu-wrong-key"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--envs", default="hopper,reacher,walker2d")
    parser.add_argument("--lanes", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--tamper-probe-steps", type=int, default=32)
    parser.add_argument("--device", default="gpu")
    return parser.parse_args()


def digest_array(arr: Any) -> bytes:
    import numpy as np

    value = np.ascontiguousarray(arr)
    header = f"{value.dtype}:{','.join(map(str, value.shape))}".encode("ascii")
    h = hashlib.sha256()
    h.update(len(header).to_bytes(2, "big"))
    h.update(header)
    h.update(value.view(np.uint8))
    return h.digest()


def payload(env_name: str, step: int, obs: Any, action: Any, next_obs: Any, reward: Any, done: Any) -> bytes:
    env_b = env_name.encode("ascii")
    return b"ATEBRAX1" + b"".join(
        [
            len(env_b).to_bytes(2, "big"),
            env_b,
            int(step).to_bytes(8, "big"),
            digest_array(obs),
            digest_array(action),
            digest_array(next_obs),
            digest_array(reward),
            digest_array(done),
        ]
    )


def sign(head: bytes, nonce: bytes, body: bytes, key: bytes = KEY) -> tuple[bytes, bytes]:
    envelope = head + nonce + body
    receipt = hmac.new(key, envelope, hashlib.sha256).digest()
    next_head = hashlib.sha256(envelope + receipt).digest()
    return receipt, next_head


def verify(head: bytes, nonce: bytes, body: bytes, receipt: bytes, key: bytes = KEY) -> bool:
    expected, _ = sign(head, nonce, body, key=key)
    return hmac.compare_digest(expected, receipt)


def write_skip_summary(args: argparse.Namespace, reason: str) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_dir": str(args.output_dir),
        "skipped": True,
        "reason": reason,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        "# Brax GPU Receipt Benchmark\n\n"
        "Run skipped.\n\n"
        f"- Reason: {reason}\n",
        encoding="utf-8",
    )


def run_env(args: argparse.Namespace, env_name: str, jax: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    from brax import envs

    env = envs.get_environment(env_name)
    lanes = int(args.lanes)
    steps = int(args.steps)
    probe_steps = min(int(args.tamper_probe_steps), steps)
    records = lanes * steps

    reset = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(1907 + lanes + len(env_name))
    reset_keys = jax.random.split(key, lanes)

    compile_start = time.perf_counter()
    state = reset(reset_keys)
    action = jax.random.uniform(key, (lanes, env.action_size), minval=-1.0, maxval=1.0)
    next_state = step_fn(state, action)
    next_state.obs.block_until_ready()
    compile_seconds = time.perf_counter() - compile_start

    # Warmup after compilation.
    state = next_state
    for i in range(args.warmup_steps):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(sub, (lanes, env.action_size), minval=-1.0, maxval=1.0)
        state = step_fn(state, action)
    state.obs.block_until_ready()

    raw_state = state
    raw_start = time.perf_counter()
    for _ in range(steps):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(sub, (lanes, env.action_size), minval=-1.0, maxval=1.0)
        raw_state = step_fn(raw_state, action)
    raw_state.obs.block_until_ready()
    raw_seconds = time.perf_counter() - raw_start

    receipt_state = state
    head = bytes(32)
    previous = None
    accepted = 0
    false_rejects = 0
    receipt_bytes = 0
    tamper_rows: list[dict[str, Any]] = []

    receipt_start = time.perf_counter()
    for i in range(steps):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(sub, (lanes, env.action_size), minval=-1.0, maxval=1.0)
        next_state = step_fn(receipt_state, action)
        next_state.obs.block_until_ready()

        obs_np = np.asarray(receipt_state.obs)
        action_np = np.asarray(action)
        next_obs_np = np.asarray(next_state.obs)
        reward_np = np.asarray(next_state.reward)
        done_np = np.asarray(next_state.done)

        body = payload(env_name, i, obs_np, action_np, next_obs_np, reward_np, done_np)
        nonce = hashlib.sha256(f"brax:{env_name}:{lanes}:{i}".encode("ascii")).digest()
        receipt, next_head = sign(head, nonce, body)
        if verify(head, nonce, body, receipt):
            accepted += lanes
        else:
            false_rejects += lanes

        if i < probe_steps:
            cases: list[tuple[str, bytes, bytes, bytes, bytes, bool]] = []

            edited_next_obs = next_obs_np.copy()
            edited_next_obs[0, 0] += np.float32(0.1)
            cases.append(
                (
                    "next_obs_edit",
                    nonce,
                    payload(env_name, i, obs_np, action_np, edited_next_obs, reward_np, done_np),
                    receipt,
                    head,
                    False,
                )
            )

            swapped_next_obs = next_obs_np.copy()
            swapped_next_obs[[0, 1]] = swapped_next_obs[[1, 0]]
            cases.append(
                (
                    "lane_swap_next_obs",
                    nonce,
                    payload(env_name, i, obs_np, action_np, swapped_next_obs, reward_np, done_np),
                    receipt,
                    head,
                    False,
                )
            )

            edited_reward = reward_np.copy()
            edited_reward[0] += np.float32(1.0)
            cases.append(
                (
                    "reward_input_edit",
                    nonce,
                    payload(env_name, i, obs_np, action_np, next_obs_np, edited_reward, done_np),
                    receipt,
                    head,
                    False,
                )
            )

            flipped_done = done_np.copy()
            flipped_done[0] = 1.0 - flipped_done[0]
            cases.append(
                (
                    "done_flip",
                    nonce,
                    payload(env_name, i, obs_np, action_np, next_obs_np, reward_np, flipped_done),
                    receipt,
                    head,
                    False,
                )
            )

            epsilon_next_obs = next_obs_np.copy()
            epsilon_next_obs[:, 0] += np.float32(1e-6)
            cases.append(
                (
                    "epsilon_bias_below_1e-5_tolerance",
                    nonce,
                    payload(env_name, i, obs_np, action_np, epsilon_next_obs, reward_np, done_np),
                    receipt,
                    head,
                    True,
                )
            )

            wrong_receipt, _ = sign(head, nonce, body, key=WRONG_KEY)
            cases.append(("wrong_key_receipt", nonce, body, wrong_receipt, head, False))

            if previous is not None:
                _prev_head, prev_nonce, prev_body, prev_receipt = previous
                cases.append(("stale_receipt_replay", nonce, body, prev_receipt, head, False))
                cases.append(("old_sequence_payload_on_current_head", prev_nonce, prev_body, prev_receipt, head, False))

            for case, case_nonce, case_body, case_receipt, case_head, naive_accept in cases:
                ok = verify(case_head, case_nonce, case_body, case_receipt)
                tamper_rows.append(
                    {
                        "env": env_name,
                        "step": i,
                        "case": case,
                        "accepted_by_receipt": ok,
                        "rejected_by_receipt": not ok,
                        "naive_tolerance_would_accept": naive_accept,
                    }
                )

        previous = (head, nonce, body, receipt)
        head = next_head
        receipt_bytes += len(nonce) + len(body) + len(receipt) + len(head)
        receipt_state = next_state

    receipt_state.obs.block_until_ready()
    receipt_seconds = time.perf_counter() - receipt_start

    row = {
        "env": env_name,
        "lanes": lanes,
        "steps": steps,
        "records": records,
        "observation_size": int(env.observation_size),
        "action_size": int(env.action_size),
        "compile_seconds": compile_seconds,
        "raw_seconds": raw_seconds,
        "receipt_seconds": receipt_seconds,
        "raw_us_per_record": 1e6 * raw_seconds / records,
        "receipt_us_per_record": 1e6 * receipt_seconds / records,
        "slowdown_vs_raw_brax_step": receipt_seconds / max(raw_seconds, 1e-12),
        "receipt_bytes_per_record": receipt_bytes / records,
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
        import jax
        from brax import envs  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        write_skip_summary(args, f"jax/brax unavailable: {type(exc).__name__}: {exc}")
        return 0

    devices = jax.devices()
    if args.device == "gpu" and not any(d.platform == "gpu" for d in devices):
        write_skip_summary(args, f"gpu unavailable; devices={devices}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_names = [x.strip() for x in args.envs.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    tamper_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for env_name in env_names:
        try:
            row, tamper = run_env(args, env_name, jax)
            rows.append(row)
            tamper_rows.extend(tamper)
        except Exception as exc:  # noqa: BLE001
            failures.append({"env": env_name, "error": f"{type(exc).__name__}: {exc}"})

    if rows:
        with (args.output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if tamper_rows:
        with (args.output_dir / "tamper_cases.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(tamper_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tamper_rows)

    summary = {
        "output_dir": str(args.output_dir),
        "skipped": False,
        "jax_version": jax.__version__,
        "devices": [str(d) for d in devices],
        "envs_requested": env_names,
        "envs_completed": [row["env"] for row in rows],
        "failures": failures,
        "lanes": args.lanes,
        "steps": args.steps,
        "total_records": sum(row["records"] for row in rows),
        "max_receipt_us_per_record": max((row["receipt_us_per_record"] for row in rows), default=None),
        "max_slowdown_vs_raw_brax_step": max((row["slowdown_vs_raw_brax_step"] for row in rows), default=None),
        "false_rejects": sum(row["false_rejects"] for row in rows),
        "tamper_probes": len(tamper_rows),
        "tamper_receipt_accepts": sum(1 for r in tamper_rows if r["accepted_by_receipt"]),
        "epsilon_bias_naive_tolerance_accepts": sum(1 for r in tamper_rows if r["naive_tolerance_would_accept"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Brax GPU Receipt Benchmark",
        "",
        "Real Brax/JAX GPU simulator vector-step benchmark with ATE batch receipts.",
        "This is a systems benchmark, not a learning/backdoor result.",
        "",
        f"- JAX: `{summary['jax_version']}`",
        f"- Devices: `{', '.join(summary['devices'])}`",
        f"- Completed envs: {', '.join(summary['envs_completed'])}",
        f"- Lanes x steps per env: {args.lanes} x {args.steps}",
        f"- Total records: {summary['total_records']}",
        f"- False rejects: {summary['false_rejects']}",
        f"- Tamper probes: {summary['tamper_probes']}",
        f"- Tamper receipts accepted: {summary['tamper_receipt_accepts']}",
        f"- Epsilon-bias probes a naive 1e-5 tolerance would accept: {summary['epsilon_bias_naive_tolerance_accepts']}",
        "",
        "Rows:",
        "",
        "| Env | Records | Obs | Act | Raw us/rec | Receipt us/rec | Slowdown | Bytes/rec | False rejects | Tamper accepts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {env} | {records} | {observation_size} | {action_size} | "
            "{raw_us_per_record:.4f} | {receipt_us_per_record:.4f} | "
            "{slowdown_vs_raw_brax_step:.2f}x | {receipt_bytes_per_record:.4f} | "
            "{false_rejects} | {tamper_receipt_accepts} |".format(**row)
        )
    if failures:
        lines.extend(["", "Failures:", ""])
        for failure in failures:
            lines.append(f"- {failure['env']}: {failure['error']}")
    lines.extend(
        [
            "",
            "Interpretation:",
            "This benchmark runs actual Brax GPU simulator steps.",
            "It measures receipt overhead and tamper rejection, not learning-loop ASR.",
            "Receipt provenance rejects below-tolerance epsilon-bias edits that a naive numeric tolerance gate would admit.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
