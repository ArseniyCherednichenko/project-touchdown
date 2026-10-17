"""Stage 2: train one AI per set of points, then compare them on the same test.

Run one:   python train_variants.py free_fuel
Run all:   python train_variants.py all     (one after another, slow)
Compare:   python train_variants.py compare
"""
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rewards import VARIANTS, make

torch.set_num_threads(1)  # so several variants can train side by side
STEPS = 1_000_000


def train(variant):
    env = make_vec_env(lambda: make(variant), n_envs=16,
                       monitor_dir=f"logs/{variant}")
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64,
                gae_lambda=0.98, gamma=0.999, n_epochs=4, ent_coef=0.01,
                seed=0, verbose=0)
    model.learn(total_timesteps=STEPS)
    model.save(f"models/variants/{variant}")
    print(f"{variant}: done")


def test(variant, episodes=50):
    """Score every AI on the ORIGINAL points, so they're comparable."""
    model = PPO.load(f"models/variants/{variant}")
    env = make(variant)
    out = {"crash": 0, "rest": 0, "timeout": 0}
    scores, fuel, on_pad = [], [], 0
    for seed in range(episodes):
        obs, _ = env.reset(seed=1000 + seed)
        total, burns, done = 0.0, 0, False
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            total += info["original_reward"]
            burns += action == 2
            done = term or trunc
        out[info["outcome"]] += 1
        on_pad += info["outcome"] == "rest" and abs(obs[0]) < 0.2
        scores.append(total)
        fuel.append(burns / 50)  # seconds of main engine
    n = episodes
    return dict(score=np.mean(scores), pad=on_pad / n, crash=out["crash"] / n,
                timeout=out["timeout"] / n, fuel=np.mean(fuel))


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "compare":
        print(f"{'variant':<18} {'score':>6} {'on pad':>7} {'crash':>6} "
              f"{'hover':>6} {'engine s':>9}")
        for v in VARIANTS:
            r = test(v)
            print(f"{v:<18} {r['score']:>6.0f} {r['pad']:>7.0%} "
                  f"{r['crash']:>6.0%} {r['timeout']:>6.0%} {r['fuel']:>9.1f}")
    else:
        for v in VARIANTS if arg == "all" else [arg]:
            train(v)
