# repo/test_PPO.py
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Union    # ← aggiunto Union

import gym
import numpy as np
from stable_baselines3 import PPO
from env.custom_hopper import *


# --- project paths -----------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "models_weights"
DATA_DIR    = ROOT / "models_data"
SOURCE_ID   = "CustomHopper-source-v0"
TARGET_ID   = "CustomHopper-target-v0"

# -----------------------------------------------------------------------------
def evaluate_once(model_path: Path,
                  env_id: str,
                  episodes: int,
                  seed: int,
                  device: str,
                  render: bool) -> Tuple[float, float]:             # ← Tuple
    """Run one checkpoint for a fixed number of episodes and return mean, std."""
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)

    model: PPO = PPO.load(str(model_path), device=device)

    rewards: List[float] = []                                      # ← List
    for _ in range(episodes):
        obs, done, ep_r = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            ep_r += float(r)
            if render:
                env.render()
        rewards.append(ep_r)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# -----------------------------------------------------------------------------
def run_eval(seed: int,
             episodes: int,
             device: str,
             render: bool,
             only_udr: bool) -> None:
    """
    Evaluate checkpoints SOURCE → {SOURCE,TARGET}
    Evaluate checkpoints TARGET → TARGET
    """
    results: List[Dict[str, Union[str, float]]] = []               # ← List / Union

    # -------- helper ---------------------------------------------------------
    def _eval_ckpt(ckpt: Path,
                   variant: str,
                   train_env: str,
                   targets: List[str]) -> None:                    # ← List
        for target in targets:
            env_id = SOURCE_ID if target == "source" else TARGET_ID
            mean_r, std_r = evaluate_once(
                ckpt, env_id, episodes, seed, device, render
            )
            print(f"{train_env}→{target} | {variant:18s} | "
                  f"{mean_r:8.2f} ± {std_r:.2f}")
            results.append({
                "train_env":   train_env,
                "test_env":    target,
                "variant":     variant,
                "mean_return": mean_r,
                "std_return":  std_r
            })

    # -----------------------------------------------------------
    # 1) CHECKPOINTS on SOURCE
    # -----------------------------------------------------------
    train_env = "source"
    target_list_src = ["source", "target"]

    ppo_ckpt = WEIGHTS_DIR / f"ppo_tuned_{train_env}_seed_{seed}.zip"
    if ppo_ckpt.exists() and not only_udr:
        _eval_ckpt(ppo_ckpt, "Plain PPO", train_env, target_list_src)

    for udr_flag in (True, False):
        if only_udr and not udr_flag:
            continue
        ckpt = WEIGHTS_DIR / f"ppo_tuned_{train_env}_seed_{seed}_UDR_{udr_flag}.zip"
        if ckpt.exists():
            _eval_ckpt(ckpt, f"UDR {udr_flag}", train_env, target_list_src)

    for ckpt in WEIGHTS_DIR.glob(
        f"ppo_tuned_{train_env}_seed_{seed}_simopt_*.zip"
    ):
        label = f"SimOpt {ckpt.stem.split('_simopt_')[1]}"
        _eval_ckpt(ckpt, label, train_env, target_list_src)

    # -----------------------------------------------------------
    # 2) CHECKPOINTS on TARGET (solo Plain PPO)
    # -----------------------------------------------------------
    train_env = "target"
    target_list_tgt = ["target"]

    ppo_ckpt = WEIGHTS_DIR / f"ppo_tuned_{train_env}_seed_{seed}.zip"
    if ppo_ckpt.exists() and not only_udr:
        _eval_ckpt(ppo_ckpt, "Plain PPO", train_env, target_list_tgt)

    for udr_flag in (True, False):
        if only_udr and not udr_flag:
            continue
        ckpt = WEIGHTS_DIR / f"ppo_tuned_{train_env}_seed_{seed}_UDR_{udr_flag}.zip"
        if ckpt.exists():
            _eval_ckpt(ckpt, f"UDR {udr_flag}", train_env, target_list_tgt)

    # -----------------------------------------------------------
    # 3) Save CSV
    # -----------------------------------------------------------
    if results:
        DATA_DIR.mkdir(exist_ok=True)
        csv_path = DATA_DIR / f"PPO_test_all_seed_{seed}_ep{episodes}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved results ➜ {csv_path}")


# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Evaluate PPO & SimOpt checkpoints (source+target)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--episodes",  type=int, default=100)
    p.add_argument("--device",    choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--render",    action="store_true")
    p.add_argument("--use-udr",   action="store_true",
                   help="Skip plain PPO and evaluate only UDR checkpoints")
    return p.parse_args()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    run_eval(seed=args.seed,
             episodes=args.episodes,
             device=args.device,
             render=args.render,
             only_udr=args.use_udr)





