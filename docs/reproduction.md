# Reproduction Guide

## Quick Checks

```bash
python -m pip install -e .
python scripts/reproduce_results.py
python scripts/run_protocol_audit.py --output-dir results/local/protocol_audit
```

The first command validates the promoted tables shipped with the artifact. The
second command reruns the software-root receipt and transform audit.

## MuJoCo DAZE Runs

The MuJoCo reproduction uses the DAZE continuous-control codebase plus the ATE
receipt patch. The original DAZE repository is not vendored here.

```bash
python scripts/run_mujoco_daze_ate.py \
  --daze-root /path/to/DAZE/continuous_env \
  --config configs/mujoco/hopper_ppo_daze.yaml \
  --output-dir results/local/hopper_ppo \
  --seeds 1 2 3 4 5
```

The runner executes three conditions: unfiltered attack, ATE, and benign ATE.
The output files are `run_matrix.csv` and `summary.json`.

## CartPole Record Tamper

```bash
python experiments/cartpole_record_attestation.py \
  --output-dir results/local/cartpole_record \
  --seeds 1 2 3 4 5 \
  --steps 80000 \
  --override-probs 0.5
```

This is the non-MuJoCo DQN record-tamper experiment used as support evidence in
the poster table.

