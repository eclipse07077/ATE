#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def check_learning(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        if as_float(row, "attack_asr") < as_float(row, "clean_asr"):
            errors.append(f"{row['task']}: attack ASR below clean ASR")
        if as_float(row, "ate_asr") > as_float(row, "clean_asr") + 0.05:
            errors.append(f"{row['task']}: ATE ASR exceeds clean baseline by more than 0.05")
        if row["admitted_poisoned_updates"] != "0":
            errors.append(f"{row['task']}: admitted poisoned updates are nonzero")
        if row["utility_gate"] != "pass":
            errors.append(f"{row['task']}: utility gate did not pass")
    return errors


def check_systems(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        if int(row["failures"]) != 0:
            errors.append(f"{row['audit']}: failures are nonzero")
    return errors


def main() -> int:
    learning = read_csv(ROOT / "results" / "promoted" / "table1_learning.csv")
    systems = read_csv(ROOT / "results" / "promoted" / "table2_systems.csv")
    errors = check_learning(learning) + check_systems(systems)
    summary = {
        "learning_rows": len(learning),
        "systems_rows": len(systems),
        "failures": errors,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

