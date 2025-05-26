from itertools import product
from stable_baselines3 import PPO

# Define the hyperparameter grid
param_grid = {
    "n_steps":    [2048, 4096, 8192],
    "batch_size": [32, 64, 128],
    "gae_lambda": [0.8, 0.9, 0.95],
    "gamma":      [0.95, 0.99],
    "n_epochs":   [10, 15, 20],
    "clip_range": [0.1, 0.2],
    "ent_coef":   [0.0, 0.005],
    "vf_coef":    [0.5, 1.0],
    "max_grad_norm": [0.5, 1.0]
}

# Prepare an iterable over all combinations
combinations = list(product(*param_grid.values()))

best_score = -float('inf')
best_params = None

for vals in combinations:
    # Map each hyperparameter name to its chosen value
    hp = dict(zip(param_grid.keys(), vals))
    print(f"Testing: {hp}")

    # Instantiate the model with English‐commented arguments
    model = PPO(
        policy='MlpPolicy',       # use a multilayer‐perceptron policy network
        env=vec_env,              # training environment (vectorized + normalized)
        seed=SEED,                # random seed for reproducibility
        verbose=0,                # no logging to stdout
        n_steps=hp["n_steps"],    # number of steps to collect before each policy update
        batch_size=hp["batch_size"],      # minibatch size for each gradient update
        gae_lambda=hp["gae_lambda"],      # GAE‐λ parameter for advantage estimation
        gamma=hp["gamma"],                # discount factor for rewards
        n_epochs=hp["n_epochs"],          # number of epochs to optimize the surrogate loss
        clip_range=hp["clip_range"],      # clipping parameter for PPO’s objective
        ent_coef=hp["ent_coef"],          # entropy coefficient (encourages exploration)
        vf_coef=hp["vf_coef"],            # value function loss coefficient
        max_grad_norm=hp["max_grad_norm"],# gradient norm clipping value
        learning_rate=lr_schedule         # schedule for the learning rate (linearly decaying)
    )

    # Train for a fixed small budget (e.g., 200k steps)
    model.learn(total_timesteps=200_000)

    # Evaluate policy performance
    mean_reward, _ = evaluate_policy(model, eval_vec, n_eval_episodes=5, deterministic=True)
    print(f"→ Mean reward: {mean_reward:.1f}")

    # Track best
    if mean_reward > best_score:
        best_score, best_params = mean_reward, hp

print("Best hyperparameters:", best_params)
print("Best mean reward:", best_score)
