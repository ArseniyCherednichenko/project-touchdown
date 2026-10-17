"""Stage 2 video: one clip per set of points, 9:16 for phones.

Needs: python train_variants.py <variant> for each variant first.
Run:   python make_variant_video.py   ->  videos/variants.mp4
"""
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from rewards import make

W, H, SIM_H = 1080, 1920, 720
TOP = (H - SIM_H) // 2
FPS, SPEED = 50, 2
HELV = "/System/Library/Fonts/HelveticaNeue.ttc"
GREY, WHITE = (150, 150, 150), (255, 255, 255)


def font(size, bold=False):
    try:
        return ImageFont.truetype(HELV, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default(size)


BIG, MID, SMALL = font(64, bold=True), font(38), font(30)

# (variant, title, the rule change in plain words)
CLIPS = [
    ("default", "Normal rules", "the points from stage 1"),
    ("free_fuel", "Fuel is free", "no cost for firing the engine"),
    ("expensive_fuel", "Fuel is 10x expensive", "every engine burn hurts"),
    ("no_crash_penalty", "Crashing is free", "no -100 for a crash"),
    ("only_final_points", "No hints", "only +100 or -100 at the very end"),
]


def play(model, variant, seed):
    env = make(variant, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, total, done = [env.render()], 0.0, False
    while not done:
        obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
        total += info["original_reward"]
        frames.append(env.render())
        done = term or trunc
    env.close()
    return frames, total, info["outcome"]


def typical(variant):
    """Honest pick: the median-scoring flight out of 9."""
    model = PPO.load(f"models/variants/{variant}")
    flights = sorted((play(model, variant, 1000 + s) for s in range(9)),
                     key=lambda f: f[1])
    return flights[4]


def card(frame, title, rule, result):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((60, TOP - 230), "PROJECT TOUCHDOWN  |  STAGE 2", font=SMALL, fill=GREY)
    d.text((60, TOP - 180), title, font=BIG, fill=WHITE)
    d.text((60, TOP - 95), rule, font=MID, fill=GREY)
    img.paste(Image.fromarray(frame).resize((W, SIM_H), Image.LANCZOS), (0, TOP))
    if result:
        d.text((60, TOP + SIM_H + 90), result, font=BIG, fill=WHITE)
    d.text((W - 60, TOP + SIM_H + 110), f"{SPEED}x speed", font=SMALL,
           fill=GREY, anchor="ra")
    return np.array(img)


out = []
for variant, title, rule in CLIPS:
    frames, score, outcome = typical(variant)
    result = {"rest": "Landed", "crash": "Crashed", "timeout": "Hovered"}[outcome]
    shown = frames[::SPEED]
    out += [card(f, title, rule, None) for f in shown]
    out += [card(shown[-1], title, rule, result)] * int(FPS * 1.2)
    print(f"{title}: {result}, score {score:.0f}")

imageio.mimsave("videos/variants.mp4", out, fps=FPS, macro_block_size=8)
print(f"videos/variants.mp4: {len(out) / FPS:.1f}s")
