"""Stage 2 extra: find the cheapest landing. Train one AI per fuel price,
then measure how often it lands and how long it burns the main engine.

Train one:  python sweep_fuel.py 1.5
Compare:    python sweep_fuel.py compare
"""
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rewards import Points
import gymnasium as gym

torch.set_num_threads(1)
PRICES = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]  # main engine cost per frame; 0.3 is the original


def make(price):
    # side engines keep the original ratio: 1/10 of the main engine
    return Points(gym.make("LunarLander-v3"), main_engine=price, side_engine=price / 10)


def path(price):
    if price == 0.3:
        return "models/variants/default"
    if price == 3.0:
        return "models/variants/expensive_fuel"
    return f"models/fuel/fuel_{price}"


def train(price):
    env = make_vec_env(lambda: make(price), n_envs=16)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, gae_lambda=0.98,
                gamma=0.999, n_epochs=4, ent_coef=0.01, seed=0, verbose=0)
    model.learn(total_timesteps=1_000_000)
    model.save(path(price))


def test(price, flights=50):
    model = PPO.load(path(price))
    env = make(price)
    landed, engine = 0, []
    for i in range(flights):
        obs, _ = env.reset(seed=1000 + i)
        burns, done = 0, False
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            burns += action == 2
            done = term or trunc
        landed += info.get("outcome") == "rest" and abs(obs[0]) < 0.2
        engine.append(burns / 50)
    return landed / flights, np.mean(engine)


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        print(f"{'fuel price':>10} {'lands on pad':>13} {'engine s':>9}")
        for p in PRICES:
            land, eng = test(p)
            print(f"{p:>10} {land:>13.0%} {eng:>9.1f}")
    else:
        train(float(sys.argv[1]))
