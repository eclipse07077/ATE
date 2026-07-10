#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


CONDITIONS = {
    "unfiltered": {"attack_rate": "0.01", "ate": "0", "receipt": "0", "disable_null": "0"},
    "ate": {"attack_rate": "0.01", "ate": "1", "receipt": "1", "disable_null": "0"},
    "benign_ate": {"attack_rate": "0.0", "ate": "1", "receipt": "1", "disable_null": "1"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daze-root", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=100)
    return parser.parse_args()


def run_one(args: argparse.Namespace, condition: str, seed: int) -> dict[str, object]:
    run_dir = args.output_dir / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    env = os.environ.copy()
    flags = CONDITIONS[condition]
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "DAZE_DISABLE_TENSORBOARD": "1",
            "DAZE_DISABLE_CHECKPOINTS": "1",
            "DAZE_SINGLE_SEED": str(seed),
            "DAZE_SINGLE_P_RATE": flags["attack_rate"],
            "DAZE_N_EVAL": str(args.eval_episodes),
            "DAZE_RESULT_JSON_PATH": str(result_path),
            "DAZE_ATTESTATION_ENFORCE": flags["ate"],
            "DAZE_TRANSITION_ATTESTATION_ENFORCE": flags["ate"],
            "DAZE_DISABLE_NULL_BRANCH": flags["disable_null"],
            "DAZE_WORKER_RECEIPT_EMIT": flags["receipt"],
            "DAZE_WORKER_RECEIPT_VERIFY": flags["receipt"],
            "DAZE_WORKER_RECEIPT_FAIL_CLOSED": "0",
            "DAZE_WORKER_RECEIPT_TRANSFORM_ID": "clip_unit",
            "DAZE_ISOLATE_ATTACK_RNG": "1",
        }
    )
    if args.timesteps is not None:
        env["DAZE_TOTAL_TIMESTEPS"] = str(args.timesteps)
    start = time.time()
    proc = subprocess.run(
        [args.python, "scripts/ppo_turtlebot.py", "--config", str(args.config)],
        cwd=args.daze_root,
        env=env,
        stdout=(run_dir / "run.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        check=False,
    )
    row = {
        "condition": condition,
        "seed": seed,
        "exit_code": proc.returncode,
        "seconds": time.time() - start,
        "result_path": str(result_path),
    }
    if result_path.exists():
        row.update(json.loads(result_path.read_text(encoding="utf-8")))
    return row


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition and int(row.get("exit_code", -1)) == 0]
        asrs = [float(row["asr"]) for row in group if "asr" in row]
        returns = [float(row["return"]) for row in group if "return" in row]
        groups.append(
            {
                "condition": condition,
                "n": len(group),
                "asr_mean": sum(asrs) / len(asrs) if asrs else None,
                "return_mean": sum(returns) / len(returns) if returns else None,
                "admitted_poisoned_updates": sum(int(row.get("worker_receipt_admitted_tampered_total") or 0) for row in group),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps({"groups": groups, "runs": rows}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for seed in args.seeds:
            rows.append(run_one(args, condition, seed))
            write_summary(rows, args.output_dir)
    write_summary(rows, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "runs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

