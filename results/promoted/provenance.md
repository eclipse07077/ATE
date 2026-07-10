# Promoted Result Provenance

Only rows that pass both integrity and utility gates are listed in the promoted
tables. Boundary and diagnostic runs are kept out of these tables.

## Table 1

- `Hopper-v5 DAZE` and `Reacher-v5 DAZE` PPO rows come from the 5-seed
  MuJoCo pre-execution receipt run.
- `Reacher-v5 DAZE` SAC comes from the 5-seed SAC Reacher ATE gate run.
- `CartPole record tamper` comes from the 5-seed Gymnasium DQN record-tamper
  run.
- `DQN replay relabel` comes from the 5-seed custom discrete DQN replay-relabel
  run.

## Table 2

- `MuJoCo promotion validation` checks the promoted MuJoCo learning rows.
- `Software-root protocol v3` checks key enrollment, receipt freshness,
  closure binding, and transform governance.
- `AMC-ATE v2 stress` checks the broader software-root boundary cases.
- `Brax/JAX GPU receipts` is a systems receipt benchmark, not a promoted GPU
  learning-defense row.
