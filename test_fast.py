# compare_gauss_vs_uniform_phi.py
# --------------------------------
# Confronta:
# 1) PPO addestrato “SimOpt‐style” (Gaussian DR) usando direttamente φ‐finali
#    (μ = [3.628, 3.000, 4.878], σ² = [0.0288, 0.0288, 0.0288])
#    con un training completo di 2 000 000 passi.
# 2) PPO con Uniform DR (±30 % moltiplicativo sui link) addestrato 100 000 passi.
#
# Alla fine valuta “source → target” senza BO.
#
# Usage:
#   python compare_gauss_vs_uniform_phi.py --episodes 20 --no-render

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

from env.custom_hopper import *     # noqa: F401,F403
from utils_simopt import get_obs, gap as gap_fn  # usiamo get_obs e gap per coerenza

# ---------- CONFIGURAZIONE ----------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# φ finali ricavati dal tuo log (step 2):
# thigh: μ=3.628, σ²=0.0288
# leg:   μ=3.000, σ²=0.0288
# foot:  μ=4.878, σ²=0.0288
PHI_FINAL = {
    "thigh": [3.619, 0.1715],
    "leg":   [2.806, 0.1715],
    "foot":  [4.914, 0.1715],
}


# Se vuoi salvare i modelli su disco:
GAUSS_LOGDIR = Path("simopt_hopper_logs_source") / "gauss_model"
UNIFORM_LOGDIR = Path("simopt_hopper_logs_source") / "uniform_model"
GAUSS_LOGDIR.mkdir(parents=True, exist_ok=True)
UNIFORM_LOGDIR.mkdir(parents=True, exist_ok=True)

# Percorsi per salvare normalizzazione e modello
GAUSS_VECNORM_PATH = GAUSS_LOGDIR / "vecnormalize.pkl"
GAUSS_MODEL_PATH    = GAUSS_LOGDIR / "gauss_final_model.zip"
UNIFORM_VECNORM_PATH = UNIFORM_LOGDIR / "vecnormalize.pkl"
UNIFORM_MODEL_PATH   = UNIFORM_LOGDIR / "uniform_best_model.zip"

# Parametri per Uniform DR (± 30 % moltiplicativo):
UNIFORM_FACTOR_LO = 0.7  # 1 – 0.3
UNIFORM_FACTOR_HI = 1.3  # 1 + 0.3

# Budget di passi PPO
TIMESTEPS_GAUSS   = 200_000
TIMESTEPS_UNIFORM = 200_000  

# -------------------------------------

def set_seed(env, seed: int = SEED):
    """Fissa il seed su env e action_space (se supportato)."""
    env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

# ----------- Gaussian‐DR utilizzando φ‐finali -----------

class HopperMassGaussianWrapper(gym.Wrapper):
    """
    Domain randomization GAUSSIANA per le masse dei link:
    thigh(2), leg(3), foot(4) campionate ciascuna da N(μ, σ²) ad ogni reset.
    Torso (indice 1) rimane fisso.
    """
    def __init__(self, env: gym.Env, phi: dict):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        # phi es: {"thigh":[μ, σ²], "leg":[μ,σ²], "foot":[μ,σ²]}
        self.phi = phi

    def reset(self, **kwargs):
        # Ripristina masse originali
        self.env.sim.model.body_mass[:] = self.base_mass
        # Sample gaussian per gli indici 2, 3, 4 (thigh, leg, foot)
        m_thigh = float(np.random.normal(self.phi["thigh"][0], np.sqrt(self.phi["thigh"][1])))
        m_leg   = float(np.random.normal(self.phi["leg"][0],   np.sqrt(self.phi["leg"][1])))
        m_foot  = float(np.random.normal(self.phi["foot"][0],  np.sqrt(self.phi["foot"][1])))
        # Evitiamo valori non fisici (massa > 0.1)
        self.env.sim.model.body_mass[2] = max(0.1, m_thigh)
        self.env.sim.model.body_mass[3] = max(0.1, m_leg)
        self.env.sim.model.body_mass[4] = max(0.1, m_foot)
        return self.env.reset(**kwargs)

    def get_parameters(self):
        return self.env.sim.model.body_mass.copy()

def make_gauss_train_vec(env_id: str) -> DummyVecEnv:
    """
    Crea DummyVecEnv + VecNormalize con Gaussian wrapper sui parametri PHI_FINAL.
    """
    base = gym.make(env_id)
    set_seed(base)
    base = HopperMassGaussianWrapper(base, PHI_FINAL)
    base = Monitor(base)
    vec = DummyVecEnv([lambda: base])
    # Durante l’allenamento vogliamo normalizzare OBS, ma NON il reward
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
    vec.training = True
    return vec

# ----------- Uniform‐DR wrapper ± 30% ----------------

class HopperMassRandomWrapper(gym.Wrapper):
    """
    Domain Randomization UNIFORM per Hopper link masses:
    thigh(2), leg(3), foot(4) moltiplicate da fattori ~ U(0.7, 1.3) ogni reset.
    """
    def __init__(self, env: gym.Env, lo: float, hi: float):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.lo = lo
        self.hi = hi

    def reset(self, **kwargs):
        # Ripristina masse originali
        self.env.sim.model.body_mass[:] = self.base_mass
        # Sample uniform per 2, 3, 4
        for idx in (2, 3, 4):
            factor = float(np.random.uniform(self.lo, self.hi))
            self.env.sim.model.body_mass[idx] *= factor
        return self.env.reset(**kwargs)

    def get_parameters(self):
        return self.env.sim.model.body_mass.copy()

def make_uniform_train_vec(env_id: str) -> DummyVecEnv:
    """
    Crea DummyVecEnv + VecNormalize con HopperMassRandomWrapper ±30%.
    """
    base = gym.make(env_id)
    set_seed(base)
    base = HopperMassRandomWrapper(base, UNIFORM_FACTOR_LO, UNIFORM_FACTOR_HI)
    base = Monitor(base)
    vec = DummyVecEnv([lambda: base])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
    vec.training = True
    return vec

# --------- Valutazione policy su target -----------

def evaluate_on_target(model: PPO, target_vec, episodes: int, render: bool) -> Tuple[float,float]:
    """
    Valuta `model` sull’environment `target_vec` per `episodes` episodi.
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

def make_target_vec(env_id: str, vecnorm_path: Path) -> DummyVecEnv:
    """
    Crea DummyVecEnv + VecNormalize per test policy su `env_id`.
    Se `vecnorm_path` esiste, carica quelli; altrimenti ne crea di nuovi.
    """
    base = gym.make(env_id)
    set_seed(base)
    base = Monitor(base)
    vec = DummyVecEnv([lambda: base])
    if vecnorm_path.exists():
        vec = VecNormalize.load(str(vecnorm_path), vec)
        vec.training = False
        vec.norm_reward = False
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
        vec.training = False
        vec.norm_reward = False
    return vec

# ------------------- CLI ---------------------

def parse_args():
    parser = argparse.ArgumentParser("Compare Gaussian‐DR vs Uniform‐DR (usando φ‐finali)")
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

    # 1) Alleno PPO con Gaussian‐DR (usando φ_final) per 2 000 000 passi
    print("\n— Training SimOpt‐style PPO (Gaussian DR, φ‐finali) —")
    gauss_vec_train = make_gauss_train_vec("CustomHopper-source-v0")
    model_gauss = PPO(
        "MlpPolicy", gauss_vec_train, seed=SEED, verbose=0,
        learning_rate=1e-4,
        n_steps=8192, batch_size=64, gae_lambda=0.9, gamma=0.99,
        n_epochs=15, clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
        max_grad_norm=0.5,
    )
    model_gauss.learn(total_timesteps=TIMESTEPS_GAUSS)
    # Salvo normalizzazione e modello
    gauss_vec_train.save(str(GAUSS_VECNORM_PATH))
    model_gauss.save(str(GAUSS_MODEL_PATH))
    print(f"✓ Gaussian‐DR model salvato in {GAUSS_MODEL_PATH}\n")

    # Preparo env target per Gaussian‐DR
    target_gauss_vec = make_target_vec("CustomHopper-target-v0", GAUSS_VECNORM_PATH)

    print("\n*** TEST: PPO (Gaussian DR) → Target ***")
    evaluate_on_target(model_gauss, target_gauss_vec, args.episodes, args.render)

    # 2) Alleno PPO con Uniform‐DR (±30 %) 
    print("\n— Training PPO (Uniform DR ±30%) —")
    uniform_vec_train = make_uniform_train_vec("CustomHopper-source-v0")
    model_uniform = PPO(
        "MlpPolicy", uniform_vec_train, seed=SEED, verbose=0,
        learning_rate=1e-4,
        n_steps=8192, batch_size=64, gae_lambda=0.9, gamma=0.99,
        n_epochs=15, clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
        max_grad_norm=0.5,
    )
    model_uniform.learn(total_timesteps=TIMESTEPS_UNIFORM)
    # Salvo normalizzazione e modello
    uniform_vec_train.save(str(UNIFORM_VECNORM_PATH))
    model_uniform.save(str(UNIFORM_MODEL_PATH))
    print(f"✓ Uniform‐DR model salvato in {UNIFORM_MODEL_PATH}\n")

    # Preparo env target per Uniform‐DR
    target_uniform_vec = make_target_vec("CustomHopper-target-v0", UNIFORM_VECNORM_PATH)

    print("\n*** TEST: PPO (Uniform DR ±30%) → Target ***")
    evaluate_on_target(model_uniform, target_uniform_vec, args.episodes, args.render)

