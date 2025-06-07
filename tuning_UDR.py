
"""
Addestramento PPO su CustomHopper con Uniform Domain Randomization (UDR)
e sweep automatico dei range di massa tramite prodotto cartesiano.

Usage:
    python hopper_mass_sweep.py          # addestra su tutti i range definiti
    python hopper_mass_sweep.py --steps 500000  # cambia timesteps per esperimento
"""
import argparse
import itertools
import json
import os
import random
from datetime import datetime
from typing import Optional

import gym
import matplotlib
import numpy as np
import torch
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -----------------------------------------------------------------------------
# Configurazione globale
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# Wrapper per Uniform Domain Randomization
# -----------------------------------------------------------------------------
class HopperMassRandomWrapper(gym.Wrapper):
    """
    Alla prima reset salva le masse "di fabbrica".
    A ogni reset successivo, moltiplica i link indicati per un fattore uniforme
    nel range specificato.
    """

    def __init__(self, env, ranges):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.ranges = ranges  # dict: {link_idx: (low, high)}

    def reset(self, **kwargs):
        # Ripristina masse originali
        self.env.sim.model.body_mass[:] = self.base_mass
        # Applica fattori casuali
        for idx, (low, high) in self.ranges.items():
            factor = np.random.uniform(low, high)
            self.env.sim.model.body_mass[idx] *= factor
        return self.env.reset(**kwargs)

    def get_parameters(self):
        return self.env.sim.model.body_mass.copy()


# -----------------------------------------------------------------------------
# Funzioni di utilità
# -----------------------------------------------------------------------------
def plot_learning_curve(monitor_env, file_path):
    rewards = np.array(monitor_env.get_episode_rewards())
    if rewards.size == 0:
        print(f"[Plot] Nessuna ricompensa in {file_path}")
        return

    smoothed = np.convolve(rewards, np.ones(20) / 20, mode="same")
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.3, label="raw")
    plt.plot(smoothed, label="smoothed")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()

    os.makedirs("training_curves", exist_ok=True)
    base = os.path.basename(file_path).replace(".csv", "")
    folder = os.path.basename(os.path.dirname(file_path))
    out_path = f"training_curves/{folder}_{base}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Salvato {out_path}")


def range_to_tag(rng):
    """
    Crea una stringa breve che codifica i range delle masse per naming directory.
    Esempio: {2:(0.9,1.1),3:(0.8,1.2),4:(0.7,1.3)}
             -> T90-110_L80-120_F70-130
    """
    t_low, t_high = rng[2]
    l_low, l_high = rng[3]
    f_low, f_high = rng[4]
    return (
        f"T{int(t_low*100):02d}-{int(t_high*100):03d}_"
        f"L{int(l_low*100):02d}-{int(l_high*100):03d}_"
        f"F{int(f_low*100):02d}-{int(f_high*100):03d}"
    )


# -----------------------------------------------------------------------------
# Fase di addestramento
# -----------------------------------------------------------------------------
def train_and_save(
    env_id: str,
    log_dir: str,
    model_path: str,
    total_timesteps: int,
    mass_ranges: Optional[dict] = None,
    use_udr: bool = False,
):
    """
    Addestra PPO su `env_id` con (o senza) UDR.
    """
    # 1) Creazione ambiente
    if env_id == "CustomHopper-source-v0" and use_udr:
        base_env = gym.make(env_id)
        base_env.seed(SEED)
        env = HopperMassRandomWrapper(base_env, ranges=mass_ranges)
    else:
        env = gym.make(env_id)
        env.seed(SEED)

    # 2) Monitor + VecNormalize
    env = Monitor(env, f"{log_dir}/train_monitor", allow_early_resets=True)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # 3) Ambiente di valutazione
    eval_base = gym.make(env_id)
    eval_base.seed(SEED + 1)
    eval_env = Monitor(eval_base, f"{log_dir}/eval_monitor", allow_early_resets=True)
    eval_vec = DummyVecEnv([lambda: eval_env])
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False)

    # 4) Controllo dell'ambiente
    check_env(vec_env.envs[0], warn=True, skip_render_check=True)

    # 5) Modello PPO
    lr_schedule = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=SEED,
        verbose=0,
        n_steps=8192,
        batch_size=32,
        gae_lambda=0.9,
        gamma=0.95,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=1.0,
        max_grad_norm=1.0,
        learning_rate=lr_schedule,
    )

    # 6) Callback di valutazione
    eval_callback = EvalCallback(
        eval_vec,
        best_model_save_path=None,   # <-- non salvare ogni "best"
        log_path=None,               # <-- non scrivere log su disco
        eval_freq=0,                 # <-- disattiva valutazioni automatiche
        deterministic=True,
        verbose=0,                   # <-- niente stampa
    )


    # 7) Addestramento
    print(
        f"[Train] {env_id} | timesteps={total_timesteps:,} | "
        f"UDR={use_udr} | ranges={mass_ranges}"
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 8) Salvataggi
    model.save(model_path)
    vec_env.save(f"{log_dir}/vecnormalize.pkl")

    # 9) Valutazione finale
    mean_ret, std_ret = evaluate_policy(
        model, eval_vec, n_eval_episodes=10, deterministic=True
    )
    print(f"[Eval] Mean return: {mean_ret:.2f} ± {std_ret:.2f}")

    # 10) Curve di apprendimento
    plot_learning_curve(env, f"{log_dir}/train_monitor.csv")
    plot_learning_curve(eval_env, f"{log_dir}/eval_monitor.csv")


# -----------------------------------------------------------------------------
# Main: generazione sweep e lancio esperimenti
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Timesteps di addestramento per esperimento (default: 2M)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Se specificato, campiona N combinazioni random invece del cartesiano completo",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1) Definizione dei range per ciascun link
    # -------------------------------------------------------------------------
    PERCENTS = [10, 30, 50]  # personalizza qui!

    THIGH_RANGES = [(1 - p / 100, 1 + p / 100) for p in PERCENTS]  # link 2
    LEG_RANGES = [(1 - p / 100, 1 + p / 100) for p in PERCENTS]  # link 3
    FOOT_RANGES = [(1 - p / 100, 1 + p / 100) for p in PERCENTS]  # link 4

    # -------------------------------------------------------------------------
    # 2) Prodotto cartesiano -> lista di dict {2:(low,high),3:...,4:...}
    # -------------------------------------------------------------------------
    MASS_SWEEP = [
        {2: t, 3: l, 4: f}
        for t, l, f in itertools.product(THIGH_RANGES, LEG_RANGES, FOOT_RANGES)
    ]

    # (Facoltativo) campionamento casuale
    if args.sample is not None and args.sample < len(MASS_SWEEP):
        MASS_SWEEP = random.sample(MASS_SWEEP, k=args.sample)

    print(f"[Setup] Totale combinazioni: {len(MASS_SWEEP)}")

    # -------------------------------------------------------------------------
    # 3) Loop sugli esperimenti
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = f"runs_{timestamp}"
    os.makedirs(base_out_dir, exist_ok=True)

    for idx, rng in enumerate(MASS_SWEEP, start=1):
        tag = f"{idx:03d}_{range_to_tag(rng)}"
        out_dir = os.path.join(base_out_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        # Salva anche i parametri in JSON per tracciabilità
        with open(os.path.join(out_dir, "mass_ranges.json"), "w") as fp:
            json.dump(rng, fp, indent=2)

        train_and_save(
            env_id="CustomHopper-source-v0",
            log_dir=out_dir,
            model_path=os.path.join(out_dir, "ppo_hopper"),
            total_timesteps=args.steps,
            mass_ranges=rng,
            use_udr=True,
        )

    print("[Done] Sweep completato.")


if __name__ == "__main__":
    main()
