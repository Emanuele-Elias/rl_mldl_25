"""train_simopt_improved.py
--------------------------------
Adaptive Gaussian SimOpt with PPO (improved, bug‑fixed)

Key points vs. vanilla version
==============================
1. **Hybrid BO objective**: `objective = gap - beta * mean_return_target`.
2. **Warm‑start PPO** across BO evaluations.
3. **Variance floor** to avoid collapsing Gaussians (σ² ≥ 1e‑3).
4. **EMA update** for means and multiplicative decay for variances.
5. **MAX_STEPS** cap and **BO_CALLS** reduced to keep wall‑clock time reasonable.
6. **Single VecNormalize instance** reused everywhere
7. Compatible with Python 3.8 (no `|` union types) and with `test.py` naming.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *  # noqa: F401, F403
from utils_simopt import train_and_save, gap as gap_fn, get_obs  # gap & get_obs are exported

# ---------- Reproducibility ----------
SEED = 42
np_random = np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------- Hyper‑parameters ----------
MAX_STEPS = 8
BO_CALLS = 15
TOTAL_TIMESTEPS_BO = 100_000
TOTAL_TIMESTEPS_FINAL = 2_000_000
MIN_SIGMA2 = 1e-3
EMA_ALPHA = 0.3
VAR_DECAY = 0.7
HYBRID_BETA = 0.1

# ---------- Helpers ----------

def make_vec_env(env_id: str, phi: Optional[dict] = None):
    """Create monitored + normalised vec env, with optional Gaussian wrapper."""
    base_env = gym.make(env_id)
    if phi is not None:
        from utils_simopt import HopperMassRandomGaussianWrapper  # noqa: F401
        base_env = HopperMassRandomGaussianWrapper(base_env, phi)
    vec = DummyVecEnv([lambda: base_env])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
    vec.seed(SEED)
    return vec

# ---------- BO Objective ----------

def bo_objective(phi_dict: dict, model: PPO, vec_shared: VecNormalize):
    """Hybrid objective: observation gap minus beta * mean return in target."""
    # 1) Continue training on source with current parameters
    source_env = make_vec_env("CustomHopper-source-v0", phi_dict)
    model.set_env(source_env)
    model.learn(total_timesteps=TOTAL_TIMESTEPS_BO, reset_num_timesteps=False, progress_bar=False)

    # 2) Collect roll‑outs for gap
    target_env = make_vec_env("CustomHopper-target-v0")
    real_obs = get_obs(model, target_env, n_episodes=5)
    sim_env = make_vec_env("CustomHopper-source-v0", phi_dict)
    sim_obs = get_obs(model, sim_env, n_episodes=5)

    # Align trajectory lengths
    min_len = min(min(len(t) for t in real_obs), min(len(t) for t in sim_obs))
    real_obs = [t[:min_len] for t in real_obs]
    sim_obs = [t[:min_len] for t in sim_obs]
    gap_value = gap_fn(real_obs, sim_obs)

    # 3) Evaluate mean return in target (SB3 helper)
    ret_mean, _ = evaluate_policy(model, target_env, n_eval_episodes=5, deterministic=True)

    return gap_value - HYBRID_BETA * ret_mean

# ---------- Main SimOpt loop ----------

def main():
    # Initial Gaussian statistics
    phi = {
        "thigh": [3.92699082, 0.5],
        "leg":   [2.71433605, 0.5],
        "foot":  [5.08938010, 0.5],
    }

    # Shared PPO (warm‑start)
    dummy_phi = {k: v.copy() for k, v in phi.items()}
    vec_shared = make_vec_env("CustomHopper-source-v0", dummy_phi)
    model = PPO(
        "MlpPolicy", vec_shared, seed=SEED, verbose=0,
        learning_rate=lambda _: 1e-4,
        n_steps=8192, batch_size=64, gae_lambda=0.9, gamma=0.99,
        n_epochs=15, clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
        max_grad_norm=0.5,
    )

    step = 0
    while step < MAX_STEPS and all(var > MIN_SIGMA2 for _, var in phi.values()):
        print(f"\n--- SimOpt step {step} ---")

        # BO search space
        search_space = [
            Real(phi["thigh"][0] - 2*phi["thigh"][1], phi["thigh"][0] + 2*phi["thigh"][1], name="thigh"),
            Real(phi["leg"][0]   - 2*phi["leg"][1],   phi["leg"][0]   + 2*phi["leg"][1],   name="leg"),
            Real(phi["foot"][0]  - 2*phi["foot"][1],  phi["foot"][0]  + 2*phi["foot"][1],  name="foot"),
        ]

        def skopt_obj(x):
            phi_dict = {
                "thigh": [x[0], phi["thigh"][1]],
                "leg":   [x[1], phi["leg"][1]],
                "foot":  [x[2], phi["foot"][1]],
            }
            return bo_objective(phi_dict, model, vec_shared)

        res = gp_minimize(
            skopt_obj, search_space, n_calls=BO_CALLS,
            random_state=SEED + step, noise="gaussian", verbose=True,
        )

        # Save convergence plot
        Path("plots").mkdir(exist_ok=True)
        plt.figure(figsize=(8, 4))
        plot_convergence(res)
        plt.title(f"BO Convergence – step {step}")
        plt.savefig(f"plots/bo_conv_step_{step}.png"); plt.close()

        best_mu = {"thigh": res.x[0], "leg": res.x[1], "foot": res.x[2]}
        print("Best μ:", best_mu, "| Obj:", res.fun)

        # EMA + variance decay
        for key in phi.keys():
            mu_old, var_old = phi[key]
            mu_new = (1-EMA_ALPHA) * mu_old + EMA_ALPHA * best_mu[key]
            var_new = max(var_old * VAR_DECAY, MIN_SIGMA2)
            phi[key] = [mu_new, var_new]
            print(f"→ {key}: μ={mu_new:.3f}, σ²={var_new:.4f}")

        step += 1

    # -------- Final training --------
    print("\nConverged φ:")
    for k, (m, v) in phi.items():
        print(f"{k}: μ={m:.3f}, σ²={v:.4f}")

    train_and_save(
        env_id="CustomHopper-source-v0",
        log_dir="simopt_hopper_logs_source",
        model_path="simopt_hopper_final_model_source.zip",
        total_timesteps=TOTAL_TIMESTEPS_FINAL,
        phi=phi,
        vecnormalize_to_load=vec_shared,
    )


if __name__ == "__main__":
    main()
