"""Phone-format media for posts, from a real training run.

Needs: python train.py run first (writes logs/ and models/checkpoints/).
Run:   python make_post_media.py
Writes to videos/:
  progress.mp4   1080x1350: the same AI at 4 points in training
  curve.png      1080x1350: average score over training
"""
import glob
import os
import re

import gymnasium as gym
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

W, H, SIM_H = 1080, 1350, 720
FPS, SPEED = 50, 2  # play at 2x so hovering doesn't take forever
HELV = "/System/Library/Fonts/HelveticaNeue.ttc"
BIG = ImageFont.truetype(HELV, 64, index=1)  # bold
MID = ImageFont.truetype(HELV, 38)
SMALL = ImageFont.truetype(HELV, 30)
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


def card(frame, title, sub, score, progress):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((60, 70), "PROJECT TOUCHDOWN", font=SMALL, fill=GREY)
    d.text((60, 120), title, font=BIG, fill=WHITE)
    d.text((60, 205), sub, font=MID, fill=GREY)
    sim = Image.fromarray(frame).resize((W, SIM_H), Image.LANCZOS)
    img.paste(sim, (0, 300))
    # training progress bar
    y = 300 + SIM_H + 90
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
for title, path, steps in stages:
    frames, score = typical(load(path))
    sub = f"after {attempts_at(steps):,} attempts" if steps else "0 attempts"
    p = steps / total_steps
    shown = frames[::SPEED]
    frames_out += [card(f, title, sub, None, p) for f in shown]
    frames_out += [card(shown[-1], title, sub, score, p)] * int(FPS * 1.2)
    print(f"{title}: {sub}, score {score:.0f}, {len(frames)} frames")

imageio.mimsave("videos/progress.mp4", frames_out, fps=FPS, macro_block_size=8)
print(f"videos/progress.mp4: {len(frames_out) / FPS:.1f}s")

# --- Chart: one series, score over attempts --------------------------------
INK, INK2, SURFACE, SERIES = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
fig, ax = plt.subplots(figsize=(10.8, 13.5), dpi=100)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.13, right=0.94, top=0.80, bottom=0.10)

# Shade where most attempts ran out of time instead of landing or crashing.
log["timeouts"] = (log["l"] >= 1000).rolling(100, min_periods=20).mean()
hov = log[log["timeouts"] > 0.5]
if len(hov):
    ax.axvspan(hov["attempt"].min(), hov["attempt"].max(), color="#e8e7e3", lw=0)
    ax.text((hov["attempt"].min() + hov["attempt"].max()) / 2, -230,
            "mostly hovering\nuntil time ran out", ha="center", va="bottom",
            fontsize=20, color=INK2)
ax.axhline(200, color=INK2, lw=1, ls=(0, (4, 4)))
ax.text(log["attempt"].max(), 210, "solved", ha="right", va="bottom",
        fontsize=20, color=INK2)
ax.plot(log["attempt"], log["score"], color=SERIES, lw=3)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c9c8c3")
ax.tick_params(colors=INK2, labelsize=18, length=0, pad=8)
ax.grid(axis="y", color="#e8e7e3", lw=1)
ax.set_axisbelow(True)
ax.set_xlabel("Landing attempts", fontsize=20, color=INK2, labelpad=12)
ax.set_ylabel("Average score", fontsize=20, color=INK2, labelpad=12)
ax.set_ylim(-250, 300)
ax.set_xlim(0, log["attempt"].max())
fig.text(0.13, 0.91, "An AI learning to land a rocket", fontsize=34,
         color=INK, weight="bold")
fig.text(0.13, 0.86, "Average of the last 100 attempts. Project Touchdown, week 1.",
         fontsize=20, color=INK2)
fig.savefig("videos/curve.png", facecolor=SURFACE)
print("videos/curve.png")
