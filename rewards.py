"""Stage 2: rewrite the points and see what the AI does with them.

Lunar Lander's built-in points, taken from its source code:
  every frame:  change in (-100 * distance to pad
                           -100 * speed
                           -100 * tilt
                           +10 per leg on the ground)
                -0.30 while the main engine fires
                -0.03 while a side engine fires
  at the end:   -100 for a crash or flying off screen
                +100 for coming to rest (anywhere, the pad bonus comes
                     from the distance term above)

This wrapper computes the same points, but every number is a setting.
"""
import gymnasium as gym
import numpy as np

DEFAULT = dict(distance=100, speed=100, tilt=100, legs=10,
               main_engine=0.30, side_engine=0.03, crash=-100, rest=100)

VARIANTS = {
    # The original rules, to check the wrapper changes nothing.
    "default": {},
    # Fuel is free. Does it stop caring about efficiency?
    "free_fuel": dict(main_engine=0, side_engine=0),
    # Fuel is 10x more expensive. Does it get scared to use the engine?
    "expensive_fuel": dict(main_engine=3.0, side_engine=0.3),
    # Crashing costs nothing. Does it just slam into the ground?
    "no_crash_penalty": dict(crash=0),
    # Only +100 or -100 at the very end, no hints along the way.
    "only_final_points": dict(distance=0, speed=0, tilt=0, legs=0,
                              main_engine=0, side_engine=0),
}


class Points(gym.Wrapper):
    def __init__(self, env, **changes):
        super().__init__(env)
        self.w = {**DEFAULT, **changes}
        self.prev = None

    def shaping(self, s):
        w = self.w
        return (-w["distance"] * np.hypot(s[0], s[1])
                - w["speed"] * np.hypot(s[2], s[3])
                - w["tilt"] * abs(s[4])
                + w["legs"] * (s[6] + s[7]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = self.shaping(obs)
        return obs, info

    def step(self, action):
        obs, original, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = original  # the real score, for comparing

        now = self.shaping(obs)
        reward = now - self.prev
        self.prev = now
        if action == 2:
            reward -= self.w["main_engine"]
        elif action in (1, 3):
            reward -= self.w["side_engine"]

        lander = self.env.unwrapped
        if terminated:
            crashed = lander.game_over or abs(obs[0]) >= 1.0
            reward = self.w["crash"] if crashed else self.w["rest"]
            info["outcome"] = "crash" if crashed else "rest"
        elif truncated:
            info["outcome"] = "timeout"
        return obs, reward, terminated, truncated, info


def make(variant, render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    return Points(env, **VARIANTS[variant])
