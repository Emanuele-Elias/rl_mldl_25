"""test_simopt_from_log.py
---------------------------------
Esegue un rapido transfer‑test *source→target* **riutilizzando**:
• Il modello warm‑start salvato durante SimOpt (se presente) **oppure**
• Una nuova policy addestrata rapidamente (fallback) 

come richiesto dall'utente: *non* rilanciare train_simopt2.py.

Usage esempio (CPU, 20 episodi senza rendering):
$ python test_simopt_from_log.py --episodes 20 --no-render 

Il file cerca automaticamente:
1. Il modello più recente in "simopt_hopper_logs_source/best_model/".
2. La normalizzazione osservazioni "simopt_hopper_logs_source/vecnormalize.pkl".

Se non trova un modello salvato, addestra 50k step con i φ finali estratti
(dal log che l'utente ci ha fornito) e poi testa.
"""

import argparse
import os
import sys
from pathlib import Path

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------- final φ copied from user's log (SimOpt step 2) ----------
PHI_FINAL = {
    "thigh": [3.838389335521518, 0.1715],
    "leg": [2.749166550642735, 0.1715],
    "foot": [4.608013715337221, 0.1715],
}

SEED = 42
MODEL_DIR = Path("simopt_hopper_logs_source") / "best_model"
MODEL_PATH = MODEL_DIR / "successor.zip"  # sb3 salva con nome policy succinto
NORMALIZE_PATH = Path("simopt_hopper_logs_source") / "vecnormalize.pkl"

# --------------------------------------------------------------------

def make_env(env_id: str, phi: dict | None = None):
    """Crea env (con Monitor) e wrapper Gaussian se phi indicato."""
    from utils_simopt import HopperMassRandomGaussianWrapper

    base = gym.make(env_id)
    if phi is not None:
        base = HopperMassRandomGaussianWrapper(base, phi)
    base.seed(SEED)
    base = Monitor(base)
    vec = DummyVecEnv([lambda: base])
    vec = VecNormalize.load(str(NORMALIZE_PATH), vec) if NORMALIZE_PATH.exists() else VecNormalize(vec, norm_obs=True, norm_reward=False)
    vec.training = False
    vec.norm_reward = False
    return vec


def get_model(vec_env):
    """Prova a caricare un PPO salvato; se non esiste, addestra flash 50k."""
    if MODEL_PATH.exists():
        print(f"✓ Carico modello salvato: {MODEL_PATH}")
        model = PPO.load(str(MODEL_PATH), env=vec_env, device="cpu")
    else:
        print("⚠️  Modello finale non trovato, alleno 50k step flash …")
        model = PPO(
            "MlpPolicy", vec_env, seed=SEED, verbose=0,
            learning_rate=1e-4, n_steps=8192, batch_size=64,
            gae_lambda=0.9, gamma=0.99, n_epochs=10,
        )
        model.learn(total_timesteps=50_000, progress_bar=False)
    return model


def eval_policy(model, vec_env, episodes: int, render: bool):
    returns = []
    for ep in range(1, episodes + 1):
        obs = vec_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            ep_ret += float(reward)
            if render:
                vec_env.render()
        returns.append(ep_ret)
        print(f"Episode {ep}: Return = {ep_ret:.2f}")
    vec_env.close()
    print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")


# ---------------- CLI -----------------

def parse_args():
    p = argparse.ArgumentParser("Quick SimOpt test without retraining")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-render", dest="render", action="store_false")
    p.set_defaults(render=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    target_vec = make_env("CustomHopper-target-v0", None)
    model = get_model(target_vec)

    print("\n### Evaluation (source with ADR → target) ###\n")
    eval_policy(model, target_vec, episodes=args.episodes, render=args.render)
