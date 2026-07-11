# Attested Transition Execution for DAZE-Style Simulator Backdoors

This repository contains code and result tables for **Attested Transition
Execution (ATE)**. ATE treats reward-free simulator backdoors as
transition-provenance failures: the learner accepts an update only when an
approved measured step relation, deterministic replay, or enrolled receipt
service vouches for the full learner-visible transition.

We assume an untrusted simulator wrapper around a trusted measured simulator
closure and verifier. The trust model is summarized in
[`docs/threat_model.md`](docs/threat_model.md).

## Repository Layout

```text
src/ate/                 Core hashing, transform, receipt, and gate code
scripts/                 Table checks and MuJoCo runner entry points
experiments/             Standalone support experiments
configs/mujoco/          DAZE/ATE MuJoCo configs used by the runner
results/tables/          Poster tables and row provenance
results/sources/         Compact source notes for table rows
docs/                    Threat model and reproduction notes
paper/                   One-page USENIX poster abstract
```

## Quick Start

```bash
python -m pip install -e .
python scripts/check_results.py
python scripts/audit_protocol.py --output-dir results/local/protocol_audit
```

`check_results.py` verifies the table gates: ATE ASR at the clean baseline,
zero admitted poisoned updates, and passing utility checks.

For the CartPole DQN experiment, install `requirements.txt`. For the Brax/JAX
receipt benchmark, install `requirements-brax.txt` in a JAX-compatible GPU
environment.

The dependency and path-scrub checks are recorded in
[`docs/dependency_check.md`](docs/dependency_check.md).

## Main Results

| Task | Learner | Attack ASR | ATE ASR | Utility | Admitted poisoned updates |
|---|---|---:|---:|---|---:|
| Hopper-v5 DAZE | PPO | 0.8435 | 0.0177 | 1.0000 | 0 |
| Reacher-v5 DAZE | PPO | 0.9843 | 0.0000 | 1.0000 | 0 |
| Reacher-v5 DAZE | SAC | 0.9996 | 0.0000 | clean-level | 0 |
| CartPole record tamper | DQN | 0.9610 | 0.4150 | clean-level | 0 |
| DQN replay relabel | DQN | 0.9080 | 0.0000 | 0.9410 | 0 |

The CSV version is in
[`results/tables/table1_learning.csv`](results/tables/table1_learning.csv).

## Reproducing Experiments

The light checks do not require MuJoCo or a GPU. Full MuJoCo DAZE runs require
a MuJoCo-capable machine and the DAZE continuous-control codebase. See
[`docs/reproduction.md`](docs/reproduction.md).

## Citation

```bibtex
@software{lee_shin_ate_daze_2026,
  title  = {Attested Transition Execution for DAZE-Style Simulator Backdoors},
  author = {Lee, Jeong Woo and Shin, Yongje},
  year   = {2026},
  url    = {https://github.com/eclipse07077/ATE}
}
```
