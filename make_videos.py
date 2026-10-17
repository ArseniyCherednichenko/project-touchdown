"""Make clips for posts: random vs trained AI, no screen recording needed.

Run:  python make_videos.py
Writes to videos/:
  random.mp4         random buttons, a few crashes
  trained.mp4        the trained AI, a few landings
  side_by_side.mp4   random (left) vs trained (right), landscape
  stacked.mp4        random (top) vs trained (bottom), portrait for phones
"""
import os

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

FPS = 50  # Lunar Lander runs at 50 frames per second
FONT = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 26)
os.makedirs("videos", exist_ok=True)

model = PPO.load("models/lander_ppo")


def run(policy, seed):
    """Play one episode, return its frames and score."""
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, total, done = [env.render()], 0.0, False
    while not done:
        action = env.action_space.sample() if policy is None else policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        frames.append(env.render())
        done = terminated or truncated
    env.close()
    return frames, total


def ai(obs):
    return model.predict(obs, deterministic=True)[0]


def label(frame, text):
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text((20, 16), text, font=FONT, fill=(255, 255, 255))
    return np.array(img)


def hold(frames, seconds=1.0):
    """Freeze the last frame so the ending is visible."""
    return frames + [frames[-1]] * int(FPS * seconds)


def save(name, frames):
    imageio.mimsave(f"videos/{name}", frames, fps=FPS, macro_block_size=8)
    print(f"videos/{name}: {len(frames) / FPS:.1f}s")


# Pick good trained landings: try seeds, keep the first three that score 250+.
trained_eps = []
for seed in range(100):
    frames, score = run(ai, seed)
    if score >= 250:
        trained_eps.append((frames, score))
        print(f"trained seed {seed}: {score:.0f}")
    if len(trained_eps) == 3:
        break

random_eps = [run(None, seed) for seed in range(100, 106)]
for _, score in random_eps:
    print(f"random: {score:.0f}")

# Single clips
save("trained.mp4", [label(f, f"Trained AI  |  score {s:.0f}")
                     for frames, s in trained_eps for f in hold(frames)])
save("random.mp4", [label(f, f"Random buttons  |  score {s:.0f}")
                    for frames, s in random_eps[:3] for f in hold(frames)])

# Comparison: one trained landing; random keeps crashing and restarting
# until the trained one is done.
right, right_score = trained_eps[0]
right = hold(right, 1.5)
left = []
i = 0
while len(left) < len(right):
    frames, _ = random_eps[i % len(random_eps)]
    left += hold(frames, 0.3)
    i += 1
left = left[: len(right)]

left = [label(f, "Random buttons") for f in left]
right = [label(f, "Trained AI, 7 min of training") for f in right]
gap = np.full((left[0].shape[0], 8, 3), 255, dtype=np.uint8)
save("side_by_side.mp4", [np.hstack([l, gap, r]) for l, r in zip(left, right)])
gap = np.full((8, left[0].shape[1], 3), 255, dtype=np.uint8)
save("stacked.mp4", [np.vstack([l, gap, r]) for l, r in zip(left, right)])
