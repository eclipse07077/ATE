# Reproduction Guide

## Quick Checks

```bash
python -m pip install -e .
python scripts/check_results.py
python scripts/audit_protocol.py --output-dir results/local/protocol_audit
```

The first command validates the checked-in tables. The second command reruns
the software-root receipt and transform audit.

## Dependencies

The base package is enough for table checks and protocol audits:

```bash
python -m pip install -e .
```

The CartPole DQN experiment uses PyTorch and Gymnasium:

```bash
python -m pip install -r requirements.txt
```

The Brax receipt benchmark should be run in a JAX-compatible GPU environment:

```bash
python -m pip install -r requirements-brax.txt
```

## MuJoCo DAZE Runs

The MuJoCo reproduction uses the DAZE continuous-control codebase plus the ATE
receipt patch. The original DAZE repository is not vendored here.

```bash
python scripts/run_mujoco.py \
  --daze-root /path/to/DAZE/continuous_env \
  --config configs/mujoco/hopper_ppo_daze.yaml \
  --output-dir results/local/hopper_ppo \
  --seeds 1 2 3 4 5
```

The runner executes three conditions: unfiltered attack, ATE, and benign ATE.
The output files are `run_matrix.csv` and `summary.json`.

## CartPole Record Tamper

```bash
python experiments/cartpole_record.py \
  --output-dir results/local/cartpole_record \
  --seeds 1 2 3 4 5 \
  --steps 80000 \
  --override-probs 0.5
```

This is the non-MuJoCo DQN record-tamper experiment used as support evidence in
the poster table.

## Brax Receipts

```bash
python experiments/brax_receipts.py \
  --output-dir results/local/brax_receipts \
  --envs hopper,reacher,walker2d
```

This measures batch receipt overhead and tamper rejection on Brax/JAX simulator
steps.
