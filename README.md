# RL‑MLDL‑25

**Reinforcement Learning with Uniform & Adaptive Domain Randomization on the MuJoCo Hopper**

A compact research framework to **train, evaluate and analyse** policy‑gradient agents on the *Hopper* task, study the impact of **Uniform Domain Randomization (UDR)** and compare several flavours of **SimOpt adaptive randomization** (CMA‑ES, PSO, DE).

Everything is orchestrated by a single launcher, `main.py`, so reproducing the paper’s results is a one‑liner.

---

## Features

* PPO baseline, REINFORCE variants and Actor‑Critic implementations
* Plug‑and‑play UDR on thigh/leg/foot masses
* Hyper‑parameter sweeps for PPO and UDR bounds
* Adaptive SimOpt training with three evolutionary optimisers and three trajectory‑discrepancy metrics
* End‑to‑end testing pipeline that builds the full source→source / source→target / target→target transfer matrix
* Automatic logging of checkpoints (`models_weights/`) and per‑episode returns (`models_data/`)
* Ready‑made Jupyter notebook for plotting and statistical analysis

---

## Quick start

```bash
# clone & enter
git clone https://github.com/Emanuele-Elias/rl_mldl_25.git
cd rl_mldl_25

# create env (example with conda)
conda env create -f environment.yml
conda activate rl_mldl_25

# 1) TRAIN – PPO on source domain with UDR, 2 M steps on GPU
python main.py --run_training --agent PPO --use-udr \
               --env source --episodes 2000000 --device cuda

# 2) TEST  – run the full transfer matrix (S→S, S→T, T→T)
python main.py --run_testing --agent PPO --all-testing

# 3) HYPER‑PARAMETER SWEEPS (optional)
python main.py --ppo_tuning          # grid search for PPO
python main.py --udr_tuning          # grid search for UDR bounds

# 4) ADAPTIVE DOMAIN RANDOMIZATION (SimOpt)
python main.py --simopt_train --simopt_optimizer cma    # CMA‑ES
python main.py --simopt_test                            # evaluate the optimised model
```

All CLI flags are self‑documented with `-h`.

---

## Project layout

```
agentsandpolicies/     ← algorithms: PPO, REINFORCE*, Actor‑Critic, SimOpt loops
env/                   ← CustomHopper, MuJoCo XML assets & wrappers
tuning/                ← grid‑search scripts for PPO and UDR
models_weights/        ← saved checkpoints (auto‑created)
models_data/           ← CSV logs (auto‑created)
notebooks/analysis.ipynb
main.py                ← pipeline launcher (entry point)
```

* `models_weights/` contains every trained policy as a `.zip` file.
* `models_data/` stores the episode‑return CSVs that the notebook uses for plotting.

---

## Typical recipes

| Goal                                        | Command                                                                       |
| ------------------------------------------- | ----------------------------------------------------------------------------- |
| **Baseline PPO (no UDR) on *target***       | `python main.py --run_training --agent PPO --env target --episodes 2000000`   |
| **Evaluate a PPO+UDR model with rendering** | `python main.py --run_testing --agent PPO --use-udr --render --episodes 5000` |
| **Run SimOpt with PSO + Wasserstein**       | `python main.py --simopt_train --simopt_optimizer pso --discrepancy score3`   |
| **Benchmark SimOpt models**                 | `python main.py --simopt_test --episodes 100000`                              |

---

## Authors

*Emanuele Francesco Elias* — s344489
*Dalia Lemmi* — s344440

---

## License

This project is released for academic purposes only. See `LICENSE` for details.
