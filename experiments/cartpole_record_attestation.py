#!/usr/bin/env python3
"""CartPole DQN replay-record provenance gate.

The attack preserves the executed action, reward, and termination flag while
forging the learner-visible replay action and next observation. Repair mode
restores the measured record before replay-buffer admission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))


@dataclass
class ReplayTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: float


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--override-probs", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--defense-modes", nargs="+", default=["none", "actiononly", "repair"])
    parser.add_argument("--transition-tamper", choices=["none", "target_replay_relabel"], default="target_replay_relabel")
    parser.add_argument("--target-action", type=int, default=1)
    parser.add_argument("--trigger-prob", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--target-update-frequency", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-trigger-samples", type=int, default=2048)
    parser.add_argument("--log-oracle", action="store_true")
    return parser.parse_args()


def sha(value: object) -> str:
    import hashlib

    arr = np.asarray(value)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def encode_obs(obs: object, obs_space: gym.Space) -> np.ndarray:
    if isinstance(obs_space, gym.spaces.Discrete):
        encoded = np.zeros(int(obs_space.n), dtype=np.float32)
        encoded[int(obs)] = 1.0
        return encoded
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def augment(obs: object, trigger: bool, obs_space: gym.Space) -> np.ndarray:
    obs_vec = encode_obs(obs, obs_space)
    return np.concatenate([obs_vec, np.asarray([float(trigger)], dtype=np.float32)])


def reset_env(env: gym.Env, rng: np.random.Generator, trigger_prob: float) -> tuple[np.ndarray, bool]:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    trigger = bool(rng.random() < trigger_prob)
    return augment(obs, trigger, env.observation_space), trigger


def step_env(env: gym.Env, action: int, rng: np.random.Generator, trigger_prob: float) -> tuple[np.ndarray, float, bool, bool]:
    obs, reward, terminated, truncated, _ = env.step(int(action))
    done = bool(terminated or truncated)
    trigger = bool(rng.random() < trigger_prob)
    return augment(obs, trigger, env.observation_space), float(reward), done, trigger


def choose_action(q: QNet, obs: np.ndarray, epsilon: float, rng: np.random.Generator, n_actions: int) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    with torch.no_grad():
        values = q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    return int(torch.argmax(values, dim=1).item())


def sample_batch(replay: deque[ReplayTransition], batch_size: int, rng: np.random.Generator) -> list[ReplayTransition]:
    idxs = rng.integers(0, len(replay), size=batch_size)
    return [replay[int(i)] for i in idxs]


def train_batch(
    q: QNet,
    target_q: QNet,
    opt: torch.optim.Optimizer,
    batch: list[ReplayTransition],
    gamma: float,
) -> float:
    obs = torch.as_tensor(np.stack([b.obs for b in batch]), dtype=torch.float32)
    actions = torch.as_tensor([b.action for b in batch], dtype=torch.long)
    rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32)
    next_obs = torch.as_tensor(np.stack([b.next_obs for b in batch]), dtype=torch.float32)
    dones = torch.as_tensor([b.done for b in batch], dtype=torch.float32)
    q_vals = q(obs).gather(1, actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        next_vals = target_q(next_obs).max(dim=1).values
        target = rewards + gamma * (1.0 - dones) * next_vals
    loss = F.smooth_l1_loss(q_vals, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.item())


def alias_next_obs(q: QNet, target_action: int, rng: np.random.Generator, candidates: np.ndarray) -> np.ndarray:
    # Pick a high-value continuation state for the forged replay record.
    with torch.no_grad():
        values = q(torch.as_tensor(candidates, dtype=torch.float32))
        scores = values.max(dim=1).values + 0.05 * values[:, target_action]
    top = torch.argmax(scores).item()
    if rng.random() < 0.05:
        top = int(rng.integers(0, len(candidates)))
    return np.asarray(candidates[top], dtype=np.float32).copy()


def make_alias_candidates(env_id: str, seed: int, n: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed + 900_000)
    env = gym.make(env_id)
    obs_list = []
    obs, _ = env.reset(seed=seed + 777)
    for _ in range(n * 3):
        action = int(rng.integers(0, int(env.action_space.n)))
        obs, _, terminated, truncated, _ = env.step(action)
        obs_list.append(augment(obs, True, env.observation_space))
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        if len(obs_list) >= n:
            break
    env.close()
    return np.asarray(obs_list, dtype=np.float32)


def train_one(seed: int, override_prob: float, defense_mode: str, args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    attack_rng = np.random.default_rng(seed + 1_337_000)
    env = gym.make(args.env_id)
    n_actions = int(env.action_space.n)
    if not 0 <= int(args.target_action) < n_actions:
        raise ValueError(f"target_action={args.target_action} outside action space size {n_actions}")
    obs_dim = int(augment(env.observation_space.sample(), False, env.observation_space).shape[0])
    q = QNet(obs_dim, n_actions)
    target_q = QNet(obs_dim, n_actions)
    target_q.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    replay: deque[ReplayTransition] = deque(maxlen=args.buffer_size)
    alias_candidates = make_alias_candidates(args.env_id, seed)

    obs, trigger = reset_env(env, rng, args.trigger_prob)
    oracle_file = None
    if args.log_oracle:
        run_dir.mkdir(parents=True, exist_ok=True)
        oracle_file = (run_dir / "oracle.csv").open("w", newline="", encoding="utf-8")
        writer = csv.writer(oracle_file)
        writer.writerow(
            [
                "step",
                "trigger",
                "submitted_action",
                "executed_action",
                "learner_action",
                "submitted_action_sha256",
                "executed_action_sha256",
                "learner_action_sha256",
                "action_overridden",
                "transition_tampered",
                "transition_repaired",
                "replay_mismatch",
                "admitted_poisoned_update",
                "reward",
                "done",
                "obs_sha256",
                "next_obs_sha256",
                "learner_next_obs_sha256",
            ]
        )
    else:
        writer = None

    transition_tampered_total = 0
    transition_repaired_total = 0
    replay_mismatch_total = 0
    admitted_poisoned_total = 0
    trigger_updates = 0
    target_updates = 0
    non_target_updates = 0
    losses: list[float] = []

    for step in range(args.steps):
        frac = step / max(1, args.steps - 1)
        epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)
        submitted = choose_action(q, obs, epsilon, rng, n_actions)
        executed = submitted
        next_obs, reward, done, _next_trigger = step_env(env, executed, rng, args.trigger_prob)

        transition_tampered = int(
            args.transition_tamper == "target_replay_relabel"
            and bool(obs[-1] > 0.5)
            and attack_rng.random() < override_prob
        )
        transition_repaired = int(defense_mode == "repair" and transition_tampered)
        learner_action = submitted
        learner_next_obs = next_obs
        if transition_tampered and not transition_repaired:
            learner_action = int(args.target_action)
            learner_next_obs = alias_next_obs(q, args.target_action, attack_rng, alias_candidates)

        replay_mismatch = int(transition_tampered)
        admitted_poisoned = int(transition_tampered and not transition_repaired)
        transition_tampered_total += transition_tampered
        transition_repaired_total += transition_repaired
        replay_mismatch_total += replay_mismatch
        admitted_poisoned_total += admitted_poisoned

        replay.append(ReplayTransition(obs.copy(), int(learner_action), reward, learner_next_obs.copy(), float(done)))
        if obs[-1] > 0.5:
            trigger_updates += 1
            if learner_action == args.target_action:
                target_updates += 1
            else:
                non_target_updates += 1

        if writer is not None:
            writer.writerow(
                [
                    step,
                    int(obs[-1] > 0.5),
                    submitted,
                    executed,
                    learner_action,
                    sha(np.asarray([submitted], dtype=np.int64)),
                    sha(np.asarray([executed], dtype=np.int64)),
                    sha(np.asarray([learner_action], dtype=np.int64)),
                    0,
                    transition_tampered,
                    transition_repaired,
                    replay_mismatch,
                    admitted_poisoned,
                    f"{reward:.6f}",
                    int(done),
                    sha(obs),
                    sha(next_obs),
                    sha(learner_next_obs),
                ]
            )

        if step >= args.learning_starts and step % args.train_frequency == 0 and len(replay) >= args.batch_size:
            losses.append(train_batch(q, target_q, opt, sample_batch(replay, args.batch_size, rng), args.gamma))
        if step % args.target_update_frequency == 0:
            target_q.load_state_dict(q.state_dict())
        if done:
            obs, trigger = reset_env(env, rng, args.trigger_prob)
        else:
            obs = next_obs

    if oracle_file is not None:
        oracle_file.close()
    env.close()

    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": seed,
        "env_id": args.env_id,
        "algo": "dqn",
        "override_prob": float(override_prob),
        "defense_mode": defense_mode,
        "transition_tamper": args.transition_tamper,
        "target_action": int(args.target_action),
        "asr": evaluate_asr(q, args, seed + 30_000),
        "clean_return": evaluate_return(q, args, seed + 40_000, force_trigger=False),
        "trigger_return": evaluate_return(q, args, seed + 50_000, force_trigger=True),
        "transition_tampered_total": int(transition_tampered_total),
        "transition_repaired_total": int(transition_repaired_total),
        "replay_mismatch_total": int(replay_mismatch_total),
        "admitted_poisoned_update_total": int(admitted_poisoned_total),
        "action_overridden_total": 0,
        "trigger_updates": int(trigger_updates),
        "trigger_target_update_rate": target_updates / max(1, trigger_updates),
        "trigger_non_target_updates": int(non_target_updates),
        "loss_mean": float(np.mean(losses)) if losses else math.nan,
        "run_dir": str(run_dir),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_asr(q: QNet, args: argparse.Namespace, seed: int) -> float:
    rng = np.random.default_rng(seed)
    env = gym.make(args.env_id)
    n_actions = int(env.action_space.n)
    obs, _ = env.reset(seed=seed)
    hits = 0
    total = 0
    for _ in range(args.eval_trigger_samples):
        action = int(rng.integers(0, n_actions))
        obs, _, terminated, truncated, _ = env.step(action)
        aug = augment(obs, True, env.observation_space)
        with torch.no_grad():
            pred = int(torch.argmax(q(torch.as_tensor(aug, dtype=torch.float32).unsqueeze(0)), dim=1).item())
        hits += int(pred == args.target_action)
        total += 1
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    env.close()
    return hits / max(1, total)


def evaluate_return(q: QNet, args: argparse.Namespace, seed: int, force_trigger: bool) -> float:
    returns = []
    for ep in range(args.eval_episodes):
        rng = np.random.default_rng(seed + ep)
        env = gym.make(args.env_id)
        raw_obs, _ = env.reset(seed=seed + ep)
        obs = augment(raw_obs, force_trigger, env.observation_space)
        total = 0.0
        horizon = int(getattr(env.spec, "max_episode_steps", 500) or 500)
        for _ in range(horizon):
            with torch.no_grad():
                action = int(torch.argmax(q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)), dim=1).item())
            raw_obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            obs = augment(raw_obs, force_trigger, env.observation_space)
            if terminated or truncated:
                break
        env.close()
        returns.append(total)
    return float(np.mean(returns))


def write_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "cartpole_record_run_metrics.csv", index=False)
    group_cols = ["defense_mode", "override_prob"]
    summary = (
        df.groupby(group_cols)
        .agg(
            n=("seed", "count"),
            seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(s))),
            asr_mean=("asr", "mean"),
            asr_min=("asr", "min"),
            asr_max=("asr", "max"),
            clean_return_mean=("clean_return", "mean"),
            trigger_return_mean=("trigger_return", "mean"),
            transition_tampered_total=("transition_tampered_total", "sum"),
            transition_repaired_total=("transition_repaired_total", "sum"),
            admitted_poisoned_update_total=("admitted_poisoned_update_total", "sum"),
            action_overridden_total=("action_overridden_total", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "cartpole_record_group_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps({"runs": rows, "groups": summary.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for override_prob in args.override_probs:
        for defense_mode in args.defense_modes:
            for seed in args.seeds:
                tamper = args.transition_tamper
                effective_override = override_prob
                if defense_mode == "clean_control":
                    tamper = "none"
                    effective_override = 0.0
                run_dir = (
                    args.output_dir
                    / f"override_{str(effective_override).replace('.', 'p')}"
                    / defense_mode
                    / f"seed_{seed}"
                )
                local_args = argparse.Namespace(**vars(args))
                local_args.transition_tamper = tamper
                rows.append(train_one(seed, float(effective_override), defense_mode, local_args, run_dir))
                write_summary(args.output_dir, rows)
    write_summary(args.output_dir, rows)
    print(json.dumps({"output_dir": str(args.output_dir), "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
