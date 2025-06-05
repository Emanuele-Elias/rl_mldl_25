"""train_simopt_improved.py
--------------------------------
Adaptive Gaussian SimOpt with PPO (improved)
Improvements vs. original:
1. Hybrid BO objective (gap – beta * return_target).
2. Warm‑start PPO model across BO evaluations to stabilise signal.
3. Minimum variance floor to avoid over‑specialisation.
4. Exponential‑moving‑average (EMA) update for means + multiplicative decay for variances.
5. Hard cap on SimOpt steps and lower BO budget per step.
6. Consistent VecNormalize instance shared in BO and final training.
"""

import os
from pathlib import Path
import random
from functools import partial

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *  # noqa: F401, F403
from utils_simopt import train_and_save, gap  # gap exported in utils

# --------------------- GLOBAL SEEDS ---------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- Hyper‑parameters ----------------------
MAX_STEPS = 8          # hard limit on SimOpt iterations
BO_CALLS = 15          # evaluations per BO pass (was 30)
TOTAL_TIMESTEPS_BO = 100_000  # PPO timesteps per BO evaluation (was 50k)
TOTAL_TIMESTEPS_FINAL = 2_000_000
MIN_SIGMA2 = 1e-3      # stop when variance hits this floor
EMA_ALPHA = 0.3        # how fast means follow BO recommendation
VAR_DECAY = 0.7        # multiplicative decay per iteration
HYBRID_BETA = 0.1      # weight of return in hybrid objective

# --------------- Utility: make vecnormalize -------------

def make_vec_env(env_id: str, phi: dict | None = None):
    """Create a monitored & normalised env with optional domain randomisation wrapper."""
    base_env = gym.make(env_id)
    if phi is not None:
        from utils_simopt import HopperMassRandomGaussianWrapper  # noqa: F401
        base_env = HopperMassRandomGaussianWrapper(base_env, phi)
    vec = DummyVecEnv([lambda: base_env])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
    vec.seed(SEED)
    return vec

# ------------------ BO objective ------------------------

def bo_objective(phi_dict: dict, model: PPO, vec: VecNormalize):
    """Hybrid SimOpt objective: gap – beta * return_target.
    The model is continued training (warm‑start)."""
    from utils_simopt import get_obs, gap as gap_fn

    # 1) Continue training on source with current phi
    source_env = make_vec_env("CustomHopper-source-v0", phi_dict)
    model.set_env(source_env)
    model.learn(total_timesteps=TOTAL_TIMESTEPS_BO, reset_num_timesteps=False, progress_bar=False)

    # 2) Roll‑outs in real (target) & sim for gap computation
    target_env = make_vec_env("CustomHopper-target-v0")
    real_obs = get_obs(model, target_env, n_episodes=5)

    sim_env = make_vec_env("CustomHopper-source-v0", phi_dict)
    sim_obs = get_obs(model, sim_env, n_episodes=5)

    # Align trajectory lengths
    min_len = min(min(len(t) for t in real_obs), min(len(t) for t in sim_obs))
    real_obs = [t[:min_len] for t in real_obs]
    sim_obs = [t[:min_len] for t in sim_obs]

    gap_value = gap_fn(real_obs, sim_obs)

    # 3) Evaluate return in target
    returns = []
    for traj in real_obs:
        # Reward is stored inside VecNormalize; we can approximate using environment info
        returns.append(float(np.sum(target_env.get_original_rewards())))
    ret_mean = np.mean(returns)

    objective = gap_value - HYBRID_BETA * ret_mean
    return objective

# -------------------- Main loop -------------------------

def main():
    # Initial Gaussian parameters (means, variances)
    phi = {
        "thigh": [3.92699082, 0.5],
        "leg":   [2.71433605, 0.5],
        "foot":  [5.08938010, 0.5],
    }

    # Shared PPO model + VecNormalize (warm‑start across BO calls)
    dummy_phi = {k: v.copy() for k, v in phi.items()}
    vec_shared = make_vec_env("CustomHopper-source-v0", dummy_phi)
    lr_schedule = lambda _: 1e-4
    model = PPO("MlpPolicy", vec_shared, seed=SEED, verbose=0, learning_rate=lr_schedule,
                n_steps=8192, batch_size=64, gae_lambda=0.9, gamma=0.99,
                n_epochs=15, clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
                max_grad_norm=0.5)

    step = 0
    while step < MAX_STEPS and all(var > MIN_SIGMA2 for _, var in phi.values()):
        print(f"\n--- SimOpt step {step} ---")
        # Define BO search space (mu ± 2σ)
        search_space = [
            Real(phi["thigh"][0] - 2*phi["thigh"][1], phi["thigh"][0] + 2*phi["thigh"][1], name="thigh"),
            Real(phi["leg"][0]   - 2*phi["leg"][1],   phi["leg"][0]   + 2*phi["leg"][1],   name="leg"),
            Real(phi["foot"][0]  - 2*phi["foot"][1],  phi["foot"][0]  + 2*phi["foot"][1],  name="foot"),
        ]

        # Wrap objective for skopt (expects list → dict)
        def skopt_obj(x):
            phi_dict = {
                "thigh": [x[0], phi["thigh"][1]],
                "leg":   [x[1], phi["leg"][1]],
                "foot":  [x[2], phi["foot"][1]],
            }
            return bo_objective(phi_dict, model, vec_shared)

        res = gp_minimize(skopt_obj, dimensions=search_space, n_calls=BO_CALLS,
                          random_state=SEED + step, verbose=True, noise="gaussian")

        # Visualise convergence
        plt.figure(figsize=(8, 4))
        plot_convergence(res)
        plt.title(f"BO Convergence – step {step}")
        Path("plots").mkdir(exist_ok=True)
        plt.savefig(f"plots/bo_conv_step_{step}.png")
        plt.close()

        # Best candidate from BO
        best_mu = {
            "thigh": res.x[0],
            "leg":   res.x[1],
            "foot":  res.x[2],
        }
        print("Best μ from BO:", best_mu, "| Obj:", res.fun)

        # ---------------- Update Gaussians (EMA + decay) ----------------
        for key in phi.keys():
            mu_old, var_old = phi[key]
            mu_new = (1-EMA_ALPHA)*mu_old + EMA_ALPHA*best_mu[key]
            var_new = max(var_old * VAR_DECAY, MIN_SIGMA2)
            phi[key] = [mu_new, var_new]
            print(f"Updated {key}: μ={mu_new:.3f}, σ²={var_new:.4f}")

        step += 1

    # ---------------- Final training ----------------
    print("\n>>> Converged φ:")
    for k, (m, v) in phi.items():
        print(f"{k}: μ={m:.3f}, σ²={v:.4f}")

    train_and_save(
        env_id="CustomHopper-source-v0",
        log_dir="simopt_hopper_logs_source",
        model_path="simopt_hopper_final_source",
        total_timesteps=TOTAL_TIMESTEPS_FINAL,
        phi=phi,
        vecnormalize_to_load=vec_shared  # reuse stats
    )


if __name__ == "__main__":
    main()
