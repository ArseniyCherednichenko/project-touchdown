"""Watch the trained AI land. Opens a window.

Run:  python watch.py
      python watch.py --random   (no AI, random buttons, for comparison)
"""
import argparse

import gymnasium as gym
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("--random", action="store_true")
parser.add_argument("--episodes", type=int, default=5)
args = parser.parse_args()

env = gym.make("LunarLander-v3", render_mode="human")
model = None if args.random else PPO.load("models/lander_ppo")

for episode in range(args.episodes):
    obs, _ = env.reset()
    total, done = 0.0, False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"Landing {episode + 1}: score {total:.0f}")

env.close()
