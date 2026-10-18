"""Stage 2 extra: find the cheapest landing. Train one AI per fuel price,
then measure how often it lands and how long it burns the main engine.

Train one:     python sweep_fuel.py 1.5        (seed 0)
Another seed:  python sweep_fuel.py 1.5 2      (same price, different luck)
Compare:       python sweep_fuel.py compare
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


def path(price, seed=0):
    if price == 0.3 and seed == 0:
        return "models/variants/default.zip"
    if price == 3.0 and seed == 0:
        return "models/variants/expensive_fuel.zip"
    return f"models/fuel/fuel_{price}_seed{seed}.zip"


def train(price, seed=0):
    env = make_vec_env(lambda: make(price), n_envs=16)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, gae_lambda=0.98,
                gamma=0.999, n_epochs=4, ent_coef=0.01, seed=seed, verbose=0)
    model.learn(total_timesteps=1_000_000)
    model.save(path(price, seed))


def test(price, seed=0, flights=50):
    model = PPO.load(path(price, seed))
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
        import glob, os, re
        print(f"{'fuel price':>10} {'runs':>5} {'lands on pad (each run)':>28} {'avg':>5} {'engine s':>9}")
        for p in sorted(set(PRICES) | {float(re.search(r"fuel_([0-9.]+)_seed", f).group(1))
                                       for f in glob.glob("models/fuel/*_seed*.zip")}):
            seeds = [s for s in range(10) if os.path.exists(path(p, s))]
            results = [test(p, s) for s in seeds]
            lands = " ".join(f"{r[0]:.0%}" for r in results)
            print(f"{p:>10} {len(seeds):>5} {lands:>28} {np.mean([r[0] for r in results]):>5.0%}"
                  f" {np.mean([r[1] for r in results]):>9.1f}")
    else:
        train(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
