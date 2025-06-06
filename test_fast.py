# compare_gauss_vs_uniform.py
# ---------------------------
# Confronta:
# 1) SimOpt-PPO (Gaussian DR) usando il modello salvato da train_simopt2.py
# 2) PPO con Uniform DR a ±30 % moltiplicativo sulle masse (HopperMassRandomWrapper)
#
# In entrambi i casi, valuta in “source → target” senza rilanciare SimOpt.
#
# Usage:
#   python compare_gauss_vs_uniform.py --episodes 20 --no-render

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *             # noqa: F401,F403
from utils_simopt import get_obs, gap as gap_fn  # riutilizziamo le funzioni gap/get_obs

# ---------- CONFIGURAZIONE ----------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# φ finali (step 2 del log SimOpt)
PHI_FINAL = {
    "thigh": [3.838389335521518, 0.1715],
    "leg":   [2.749166550642735, 0.1715],
    "foot":  [4.608013715337221, 0.1715],
}

# Percorsi per il modello SimOpt e la normalizzazione
SIMOPT_MODEL_DIR = Path("simopt_hopper_logs_source") / "best_model"
SIMOPT_MODEL_PATH = SIMOPT_MODEL_DIR / "successor.zip"
SIMOPT_VECNORM_PATH = Path("simopt_hopper_logs_source") / "vecnormalize.pkl"

# Parametri per l’uniform DR “a ±30 %”
UNIFORM_FACTOR_LO = 0.7  # 1 - 0.3
UNIFORM_FACTOR_HI = 1.3  # 1 + 0.3

# Quantità di passi PPO per training “flash” (se manca modello)
TIMESTEPS_FLASH = 50_000

# Quantità di passi PPO per training Uniform DR completo
TIMESTEPS_UNIFORM = 100_000

# -------------------------------------

def set_seed(env, seed: int = SEED):
    """Fissa il seed su env e action_space (se supportato)."""
    env.seed(seed)
    if hasattr(env.action_space, 'seed'):
        env.action_space.seed(seed)

# ----------- Wrappers DR -------------

class HopperMassRandomWrapper(gym.Wrapper):
    """
    Domain Randomization per Hopper link masses:
    moltiplica thigh(2), leg(3), foot(4) per un fattore uniformemente scelto in [0.7, 1.3].
    Torso (indice 1) rimane fisso.
    """
    def __init__(self, env: gym.Env, lo: float, hi: float):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.lo = lo
        self.hi = hi

    def reset(self, **kwargs):
        # Ripristina masse originali
        self.env.sim.model.body_mass[:] = self.base_mass
        # Per ogni indice 2, 3, 4 (thigh, leg, foot) campiona un fattore
        for idx in (2, 3, 4):
            factor = float(np.random.uniform(self.lo, self.hi))
            self.env.sim.model.body_mass[idx] *= factor
        return self.env.reset(**kwargs)

    def get_parameters(self):
        return self.env.sim.model.body_mass.copy()

# -------------------------------------

# ------- Helpers per SimOpt-Policy -------

def make_simopt_vec(env_id: str) -> DummyVecEnv:
    """
    Crea DummyVecEnv + VecNormalize caricando le statistiche salvate da SimOpt (se esistono).
    Se mancanti, crea comunque VecNormalize da zero.
    """
    base = gym.make(env_id)
    set_seed(base)
    base = Monitor(base)
    vec = DummyVecEnv([lambda: base])

    if SIMOPT_VECNORM_PATH.exists():
        vec = VecNormalize.load(str(SIMOPT_VECNORM_PATH), vec)
        vec.training = False
        vec.norm_reward = False
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
        vec.training = False
        vec.norm_reward = False

    return vec

def load_or_flash_simopt(env_id: str) -> PPO:
    """
    Se SIMOPT_MODEL_PATH esiste: caricalo. Altrimenti:
    addestramento flash di 50k PPO senza DR (fallback).
    """
    vec = make_simopt_vec(env_id)
    if SIMOPT_MODEL_PATH.exists():
        print(f"✓ Carico modello SimOpt: {SIMOPT_MODEL_PATH}")
        model = PPO.load(str(SIMOPT_MODEL_PATH), env=vec, device="cpu")
    else:
        print("⚠️  SimOpt model non trovato, flash‐training 50k passi …")
        model = PPO(
            "MlpPolicy", vec, seed=SEED, verbose=0,
            learning_rate=1e-4, n_steps=8192, batch_size=64,
            gae_lambda=0.9, gamma=0.99, n_epochs=10,
        )
        model.learn(total_timesteps=TIMESTEPS_FLASH)
    return model

# ------- Helpers per Uniform-DR-Policy -------

def make_uniform_train_vec(env_id: str) -> DummyVecEnv:
    """
    Crea DummyVecEnv + VecNormalize con HopperMassRandomWrapper (fattori in [0.7,1.3]).
    Questo wrapper randomizza le masse ogni reset.
    """
    base = gym.make(env_id)
    set_seed(base)
    base = HopperMassRandomWrapper(base, UNIFORM_FACTOR_LO, UNIFORM_FACTOR_HI)
    base = Monitor(base)
    vec = DummyVecEnv([lambda: base])
    # durante l’allenamento vogliamo normalizzare le osservazioni
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
    vec.training = True
    return vec

def train_uniform_policy(env_id: str) -> Tuple[PPO, DummyVecEnv]:
    """
    Allena una policy PPO con Uniform DR (±30%) per TIMESTEPS_UNIFORM passi.
    Salva modello e normalizzazione in simopt_hopper_logs_source/uniform_model/.
    """
    logdir = Path("simopt_hopper_logs_source") / "uniform_model"
    logdir.mkdir(parents=True, exist_ok=True)
    model_path = logdir / "uniform_best_model.zip"
    vecnorm_path = logdir / "vecnormalize.pkl"

    vec = make_uniform_train_vec(env_id)
    print(f"🟢 Alleno Uniform-DR PPO per {TIMESTEPS_UNIFORM} passi …")
    model = PPO(
        "MlpPolicy", vec, seed=SEED, verbose=0,
        learning_rate=1e-4, n_steps=8192, batch_size=64,
        gae_lambda=0.9, gamma=0.99, n_epochs=10,
    )
    model.learn(total_timesteps=TIMESTEPS_UNIFORM)

    # Salva normalizzazione e modello
    vec.save(str(vecnorm_path))
    model.save(str(model_path))
    print(f"✓ Modell oUniform-DR salvato in {model_path}")
    return model, vec

# ------- Funzione di valutazione su target -------

def evaluate_on_target(model: PPO, target_vec, episodes: int, render: bool) -> Tuple[float, float]:
    """
    Valuta la policy `model` nell’ambiente `target_vec` per `episodes` episodi.
    Restituisce (media_return, std_return).
    """
    returns = []
    for ep in range(1, episodes + 1):
        obs = target_vec.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = target_vec.step(action)
            ep_ret += float(reward)
            if render:
                target_vec.render()
        returns.append(ep_ret)
        print(f"Episode {ep}: Return = {ep_ret:.2f}")
    target_vec.close()
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print(f"→ Mean return: {mean_ret:.2f} ± {std_ret:.2f}\n")
    return mean_ret, std_ret

# --------- Creazione env target per SimOpt Policy ---------

def make_simopt_target_vec(env_id: str) -> DummyVecEnv:
    """
    Crea DummyVecEnv + VecNormalize per test SimOpt policy:
    Se esiste vecnormalize salvato da SimOpt, lo carica.
    Altrimenti, costruisce nuovo VecNormalize (senza normalizzazione dei reward).
    """
    base = gym.make(env_id)
    set_seed(base)
    base = Monitor(base)
    vec = DummyVecEnv([lambda: base])

    if SIMOPT_VECNORM_PATH.exists():
        vec = VecNormalize.load(str(SIMOPT_VECNORM_PATH), vec)
        vec.training = False
        vec.norm_reward = False
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
        vec.training = False
        vec.norm_reward = False

    return vec

# --------- Creazione env target per Uniform-DR Policy ---------

def make_uniform_target_vec(env_id: str) -> DummyVecEnv:
    """
    Crea DummyVecEnv + VecNormalize per test Uniform-DR policy:
    Carica la normalizzazione appresa durante train_uniform_policy.
    """
    logdir = Path("simopt_hopper_logs_source") / "uniform_model"
    vecnorm_path = logdir / "vecnormalize.pkl"

    base = gym.make(env_id)
    set_seed(base)
    base = Monitor(base)
    vec = DummyVecEnv([lambda: base])

    if vecnorm_path.exists():
        vec = VecNormalize.load(str(vecnorm_path), vec)
        vec.training = False
        vec.norm_reward = False
    else:
        # Fallback: normalizza comunque
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
        vec.training = False
        vec.norm_reward = False

    return vec

# --------------------- CLI ---------------------

def parse_args():
    parser = argparse.ArgumentParser("Compare SimOpt vs Uniform-DR (±30%)")
    parser.add_argument(
        "--episodes", type=int, default=30,
        help="Numero di episodi per ciascun test"
    )
    parser.add_argument(
        "--render", dest="render", action="store_true",
        help="Renderizza l'ambiente durante la valutazione"
    )
    parser.add_argument(
        "--no-render", dest="render", action="store_false",
        help="Non renderizzare"
    )
    parser.set_defaults(render=False)
    return parser.parse_args()

# ------------------- MAIN ----------------------

if __name__ == "__main__":
    args = parse_args()

    # 1) Carico (o flash‐train) la SimOpt-Policy
    print("\n— Caricamento SimOpt-Policy (Gaussian DR) —")
    simopt_model = load_or_flash_simopt("CustomHopper-source-v0")

    # Preparo env target per SimOpt-Policy
    target_vec_simopt = make_simopt_target_vec("CustomHopper-target-v0")

    print("\n*** TEST: SimOpt-Policy (Gaussian DR) → Target ***")
    evaluate_on_target(simopt_model, target_vec_simopt, args.episodes, args.render)

    # 2) Alleno e testo la Uniform-DR-Policy (±30% moltiplicativo)
    print("\n— Training Uniform-DR Policy (±30%) —")
    uniform_model, uniform_vec_train = train_uniform_policy("CustomHopper-source-v0")

    # Preparo env target per Uniform-DR Policy
    uniform_vec_target = make_uniform_target_vec("CustomHopper-target-v0")

    print("\n*** TEST: Uniform-DR Policy (±30%) → Target ***")
    evaluate_on_target(uniform_model, uniform_vec_target, args.episodes, args.render)
