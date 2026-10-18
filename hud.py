"""Telemetry overlay: see what the AI sees and decides while it flies.

Next to the rocket, every frame shows:
  what it sees     height, speed, tilt, legs (from its 8 numbers)
  pilot            how likely it is to press each of the 4 buttons
  coach            how many points it still expects from here
  points           this frame and the total so far

Run:  python hud.py   ->  videos/telemetry.mp4  (one flight of the trained AI)
"""
import math
import os

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HELV = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
BUTTONS = ["Nothing", "Left engine", "Main engine", "Right engine"]
BG, INK, DIM, LINE = (14, 16, 20), (238, 241, 245), (130, 138, 150), (44, 50, 58)
BLUE, GREEN, RED, AMBER = (90, 140, 255), (76, 194, 135), (235, 92, 92), (240, 167, 58)
PANEL_W = 340


def font(size, mono=False, bold=False):
    try:
        return ImageFont.truetype(MONO if mono else HELV, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default(size)


F_LABEL, F_VALUE, F_HEAD, F_SMALL = font(13), font(15, mono=True), font(12, bold=True), font(12)


def brain(model, obs):
    """What the pilot wants (4 probabilities) and what the coach expects."""
    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(obs)
        probs = model.policy.get_distribution(t).distribution.probs[0].numpy()
        value = float(model.policy.predict_values(t)[0][0])
    return probs, value


def panel(obs, probs, value, chosen, reward, total, t, height):
    img = Image.new("RGB", (PANEL_W, height), BG)
    d = ImageDraw.Draw(img)
    x, y = 20, 14

    def head(text, extra=""):
        nonlocal y
        d.text((x, y), text.upper() + extra, font=F_HEAD, fill=DIM)
        y += 19

    def row(label, value, color=INK):
        nonlocal y
        d.text((x, y), label, font=F_LABEL, fill=DIM)
        d.text((PANEL_W - 20, y - 1), value, font=F_VALUE, fill=color, anchor="ra")
        y += 18

    head("What it sees", f"      {t / 50:4.1f} s")
    row("Height", f"{obs[1]:+.2f}")
    row("Sideways", f"{obs[0]:+.2f}")
    row("Falling speed", f"{-obs[3]:+.2f}", RED if obs[3] < -0.5 else INK)
    row("Sideways speed", f"{obs[2]:+.2f}")
    row("Tilt", f"{math.degrees(obs[4]):+5.1f}°", RED if abs(obs[4]) > 0.4 else INK)
    row("Legs down", f"{'L' if obs[6] else '-'} {'R' if obs[7] else '-'}",
        GREEN if obs[6] and obs[7] else INK)

    y += 6
    head("Pilot: which button?")
    bar_w = PANEL_W - 40
    for i, (name, p) in enumerate(zip(BUTTONS, probs)):
        on = i == chosen
        d.text((x, y), name, font=F_LABEL, fill=INK if on else DIM)
        d.text((PANEL_W - 20, y - 1), f"{p:4.0%}", font=F_VALUE, fill=INK if on else DIM,
               anchor="ra")
        y += 16
        d.rounded_rectangle((x, y, x + bar_w, y + 6), 3, fill=LINE)
        if p > 0.01:
            d.rounded_rectangle((x, y, x + max(6, bar_w * p), y + 6), 3,
                                fill=BLUE if on else (70, 78, 92))
        y += 12

    y += 4
    head("Coach: points still to come")
    row("Expects", f"{value:+.0f}", GREEN if value > 0 else RED)

    y += 6
    head("Points")
    row("This frame", f"{reward:+.2f}", GREEN if reward > 0 else (RED if reward < 0 else INK))
    row("Total", f"{total:+.0f}", GREEN if total >= 200 else INK)
    return img


def fly_with_telemetry(model, seed=1004, speed=1):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, total, t, done = [], 0.0, 0, False
    reward, chosen = 0.0, 0
    while True:
        probs, value = brain(model, obs)
        if t % speed == 0 or done:
            sim = Image.fromarray(env.render())
            out = Image.new("RGB", (sim.width + PANEL_W, sim.height), BG)
            out.paste(sim, (0, 0))
            out.paste(panel(obs, probs, value, chosen, reward, total, t, sim.height),
                      (sim.width, 0))
            frames.append(np.array(out))
        if done:
            break
        chosen = int(model.predict(obs, deterministic=True)[0])
        obs, reward, term, trunc, _ = env.step(chosen)
        total += reward
        t += 1
        done = term or trunc
    env.close()
    return frames, total


if __name__ == "__main__":
    from stable_baselines3 import PPO

    os.makedirs("videos", exist_ok=True)
    frames, total = fly_with_telemetry(PPO.load("models/lander_ppo"))
    frames += [frames[-1]] * 75
    imageio.mimsave("videos/telemetry.mp4", frames, fps=50, macro_block_size=8)
    print(f"videos/telemetry.mp4: {len(frames) / 50:.1f}s, score {total:.0f}")
