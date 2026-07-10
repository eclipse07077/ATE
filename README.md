# Attested Transition Execution for DAZE-Style Simulator Backdoors

This repository contains the artifact code and promoted result tables for
**Attested Transition Execution (ATE)**. ATE treats reward-free simulator
backdoors as transition-provenance failures: the learner accepts an update only
when an approved measured step relation, deterministic replay, or enrolled
receipt service vouches for the full learner-visible transition.

The scope is deliberately narrow. The simulator wrapper is untrusted, but the
approved simulator closure and verifier are trusted. See
[`docs/threat_model.md`](docs/threat_model.md) for the exact boundary.

## Repository Layout

```text
src/ate/                 Core hashing, transform, receipt, and gate code
scripts/                 Table checks and MuJoCo runner entry points
experiments/             Standalone support experiments
configs/mujoco/          DAZE/ATE MuJoCo configs used by the runner
results/promoted/        Tables used by the poster
results/source_summaries Source summaries for the promoted rows
docs/                    Threat model and reproduction notes
paper/                   One-page USENIX poster abstract
```

## Quick Start

```bash
python -m pip install -e .
python scripts/reproduce_results.py
python scripts/run_protocol_audit.py --output-dir results/local/protocol_audit
```

`reproduce_results.py` checks that the promoted rows satisfy the stated gates:
ATE ASR at the clean baseline, zero admitted poisoned updates, and passing
utility gates.

## Main Promoted Evidence

| Task | Learner | Attack ASR | ATE ASR | Utility | Admitted poisoned updates |
|---|---|---:|---:|---|---:|
| Hopper-v5 DAZE | PPO | 0.8435 | 0.0177 | 1.0000 | 0 |
| Reacher-v5 DAZE | PPO | 0.9843 | 0.0000 | 1.0000 | 0 |
| Reacher-v5 DAZE | SAC | 0.9996 | 0.0000 | clean-level | 0 |
| CartPole record tamper | DQN | 0.9610 | 0.4150 | clean-level | 0 |
| DQN replay relabel | DQN | 0.9080 | 0.0000 | 0.9410 | 0 |

The CSV version is in
[`results/promoted/table1_learning.csv`](results/promoted/table1_learning.csv).

## Reproducing Experiments

The light checks run on a laptop. Full MuJoCo DAZE runs require a MuJoCo-capable
machine and the DAZE continuous-control codebase. See
[`docs/reproduction.md`](docs/reproduction.md).

## Citation

```bibtex
@software{lee_shin_ate_daze_artifact_2026,
  title  = {Attested Transition Execution for DAZE-Style Simulator Backdoors},
  author = {Lee, Jeong Woo and Shin, Yongje},
  year   = {2026},
  url    = {https://github.com/eclipse07077/Attested-Transition-Execution-for-DAZE-Style-Simulator-Backdoors}
}
```

