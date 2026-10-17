"""Week 1: train an AI to land Lunar Lander.

Run:  python train.py            (full run, about 1M steps)
      python train.py --steps 50000   (quick test)
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1_000_000)
args = parser.parse_args()

# 16 copies of the simulator run side by side, so it learns 16x faster.
# monitor_dir: every finished episode (score, length, time) goes to logs/.
env = make_vec_env("LunarLander-v3", n_envs=16, monitor_dir="logs/")

# Save a snapshot of the AI every ~100k steps, to show how it learned.
checkpoints = CheckpointCallback(
    save_freq=100_000 // 16, save_path="models/checkpoints", name_prefix="lander"
)

# PPO: the algorithm. "MlpPolicy": a small neural network.
# It reads 8 numbers (position, speed, tilt, leg contact)
# and picks one of 4 actions (nothing, left, main, right engine).
# Settings are the tuned ones from RL Baselines3 Zoo.
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    verbose=1,
)

model.learn(total_timesteps=args.steps, callback=checkpoints)
model.save("models/lander_ppo")

# Score: above 200 counts as solved.
eval_env = make_vec_env("LunarLander-v3", n_envs=1)
mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"\nAverage score over 10 landings: {mean:.0f} (+/- {std:.0f}). 200+ = solved.")
