"""Phone-format video of a real training run, plus the README GIF.

Needs: python train.py run first (writes logs/ and models/checkpoints/).
Run:   python make_post_media.py
Writes to videos/:
  progress.mp4   1080x1920 (9:16): the same AI at 4 points in training
and docs/touchdown.gif, a small landscape version for the README.
"""
import glob
import os
import re

import gymnasium as gym
import imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

W, H, SIM_H = 1080, 1920, 720
TOP = (H - SIM_H) // 2  # simulator sits in the middle, clear of app buttons
FPS, SPEED = 50, 2  # play at 2x so hovering doesn't take forever
HELV = "/System/Library/Fonts/HelveticaNeue.ttc"  # macOS; other systems fall back


def font(size, bold=False):
    try:
        return ImageFont.truetype(HELV, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default(size)


BIG, MID, SMALL = font(64, bold=True), font(38), font(30)
GREY, WHITE = (150, 150, 150), (255, 255, 255)
os.makedirs("videos", exist_ok=True)

# --- Training log: every finished episode from all 16 simulators -----------
runs = []
for path in glob.glob("logs/*.monitor.csv"):
    runs.append(pd.read_csv(path, skiprows=1))
log = pd.concat(runs).sort_values("t").reset_index(drop=True)
log["steps"] = log["l"].cumsum()
log["attempt"] = np.arange(1, len(log) + 1)
log["score"] = log["r"].rolling(100, min_periods=20).mean()
log["length"] = log["l"].rolling(100, min_periods=20).mean()


def attempts_at(steps):
    return int((log["steps"] <= steps).sum())


# --- Stages: pick checkpoints from the data, not by hand --------------------
ckpts = sorted(
    (int(re.search(r"_(\d+)_steps", p).group(1)), p)
    for p in glob.glob("models/checkpoints/*.zip")
)
env_vec = make_vec_env("LunarLander-v3", n_envs=1)


def load(path):
    if path is None:  # a fresh, untrained network
        return PPO("MlpPolicy", env_vec, seed=0)
    return PPO.load(path)


def play(model, seed):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, total, done = [env.render()], 0.0, False
    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, _ = env.step(action)
        total += r
        frames.append(env.render())
        done = term or trunc
    env.close()
    return frames, total


def typical(model):
    """Honest pick: play 9 episodes, show the one with the median score."""
    eps = sorted((play(model, s) for s in range(9)), key=lambda e: e[1])
    return eps[4]


def gif_frame(frame, title, sub):
    img = Image.fromarray(frame).resize((480, 320), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    d.text((16, 12), title, font=font(24, bold=True), fill=WHITE)
    d.text((16, 42), sub, font=font(16), fill=GREY)
    return np.array(img)


def card(frame, title, sub, score, progress):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((60, TOP - 230), "PROJECT TOUCHDOWN", font=SMALL, fill=GREY)
    d.text((60, TOP - 180), title, font=BIG, fill=WHITE)
    d.text((60, TOP - 95), sub, font=MID, fill=GREY)
    sim = Image.fromarray(frame).resize((W, SIM_H), Image.LANCZOS)
    img.paste(sim, (0, TOP))
    # training progress bar
    y = TOP + SIM_H + 90
    d.text((60, y), "Training progress", font=SMALL, fill=GREY)
    d.rounded_rectangle((60, y + 50, W - 60, y + 62), 6, fill=(60, 60, 60))
    d.rounded_rectangle((60, y + 50, 60 + max(12, (W - 120) * progress), y + 62),
                        6, fill=WHITE)
    if score is not None:
        d.text((60, y + 100), f"Score {score:.0f}", font=MID, fill=WHITE)
    d.text((W - 60, y + 100), f"{SPEED}x speed", font=SMALL, fill=GREY, anchor="ra")
    return np.array(img)


# Hovering stage: the first snapshot whose typical flight runs out of time.
hover = None
for steps, path in ckpts:
    frames, score = typical(load(path))
    if len(frames) > 1000:
        hover = ("Hovering", path, steps)
        break
stages = [
    ("Before training", None, 0),
    ("Crashing", ckpts[0][1], ckpts[0][0]),
    *([hover] if hover else []),
    ("Landing", "models/lander_ppo", int(log["steps"].iloc[-1])),
]

total_steps = stages[-1][2]
frames_out = []
gif = []
for title, path, steps in stages:
    frames, score = typical(load(path))
    sub = f"after {attempts_at(steps):,} attempts" if steps else "0 attempts"
    p = steps / total_steps
    shown = frames[::SPEED]
    frames_out += [card(f, title, sub, None, p) for f in shown]
    frames_out += [card(shown[-1], title, sub, score, p)] * int(FPS * 1.2)
    gif += [gif_frame(f, title, f"{sub}  |  score {score:.0f}") for f in frames[::4]]
    gif += [gif[-1]] * 12
    print(f"{title}: {sub}, score {score:.0f}, {len(frames)} frames")

imageio.mimsave("videos/progress.mp4", frames_out, fps=FPS, macro_block_size=8)
print(f"videos/progress.mp4: {len(frames_out) / FPS:.1f}s")

os.makedirs("docs", exist_ok=True)
imageio.mimsave("docs/touchdown.gif", gif, duration=1000 / 25, loop=0)
print("docs/touchdown.gif")
