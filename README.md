# Project Touchdown

**Teaching an AI to land a rocket. First in a simulator, then for real.**

![The same AI before training, crashing, hovering and finally landing](docs/touchdown.gif)

SpaceX lands rockets on their tail and catches boosters out of the air. I had
no idea how you get a machine to do that, so I started building it myself.

Nobody tells this AI how to fly. It figures it out by trial and error.

## How it learns

The AI controls a lander with three engines: main, left and right. Every
frame it sees eight numbers (position, speed, tilt, which legs touch the
ground) and picks one move. Then it gets points:

| What happens | Points |
|---|---|
| Lands and comes to rest | **+100** |
| Crashes | **-100** |
| Gets closer to the pad, slows down, stays level | a few points, every frame |
| A leg touches the ground | +10 per leg |
| Fires an engine | a small cost, every frame |

After each batch of attempts it nudges its neural network toward the moves
that earned more points. That's it. That is the whole teacher.

## What happened

| Phase | Attempts | What it did |
|---|---|---|
| Before training | 0 | Falls out of the sky |
| Crashing | ~1,000 | Tries to land, mostly crashes |
| **Hovering** | ~2,400 | Found a loophole: if you never land, you can never crash. So it hovered until time ran out. |
| Landing | 3,121 | Lands 19 out of 20 times, average score 239 |

Total training time: about 10 minutes on my laptop.

The hovering phase is my favorite part. The AI did exactly what the points
told it to, not what I meant. Designing those points is called **reward
shaping**, and it is one of the hardest problems in training robots.

## Stage 2: what if I change the points?

Same AI, same training, five different sets of points. Then every AI is
tested on the original rules, 50 flights each.

| Rules | Score | Lands on pad | Crashes | Engine time |
|---|---|---|---|---|
| Normal | 265 | 100% | 0% | 3.2 s |
| Fuel is free | 217 | 100% | 0% | 5.9 s |
| Fuel is 10x expensive | -32 | 0% | 100% | 0.5 s |
| Crashing is free | 277 | 98% | 2% | 2.2 s |
| No hints, only +100 / -100 at the end | -371 | 0% | 100% | 2.8 s |

- **Free fuel:** still lands, but burns almost twice as much.
- **Expensive fuel:** so scared of the engine that it barely brakes and
  crashes every time.
- **Crashing is free:** still lands. Crashing ends the flight before the
  +100 for landing, so it avoids crashing anyway.
- **No hints:** never learns. Landing by pure luck is too rare to learn from.
  The small points along the way are what make learning possible.

One training run per variant, so treat the exact numbers as rough. Run it
yourself: `python train_variants.py free_fuel`, then
`python train_variants.py compare`.

## Try it yourself

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArseniyCherednichenko/project-touchdown/blob/main/notebooks/stage1_land_a_rocket.ipynb)
**No install needed:** train your own rocket AI in the browser with the
stage 1 notebook. Free, runs on Google Colab.

Python 3.9 or newer.

```bash
git clone https://github.com/ArseniyCherednichenko/project-touchdown.git
cd project-touchdown
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Watch my trained AI land:

```bash
python watch.py
```

Watch what happens without a brain, for comparison:

```bash
python watch.py --random
```

Train your own from scratch (about 10 minutes):

```bash
python train.py
```

**Can you beat my score of 239?** Change the settings in `train.py`, train
longer, or rewrite the points. If you beat it, open an issue and tell me how.

## Roadmap

- [x] **Stage 1:** Train an AI to land in a 2D simulator
- [x] **Stage 2:** Change the points and see what breaks
- [ ] **Stage 3:** My own rocket with real physics: fuel that burns off, an
      engine that tilts to steer, wind
- [ ] **Stage 4:** AI vs a hand-written controller, the way engineers do it
- [ ] **Stage 5:** A website: land it yourself, then watch the AI, and change
      the rules with sliders
- [ ] **Stage 6:** Tower catch: fly into the arms instead of landing on legs
- [ ] **Stage 7:** Real hardware

## Files

| File | What it does |
|---|---|
| `train.py` | Trains the AI and saves snapshots along the way |
| `watch.py` | Opens a window and shows it landing |
| `make_post_media.py` | Renders the GIF above and a phone video from a training run |
| `rewards.py` | Every point value as a setting, plus the stage 2 variants |
| `train_variants.py` | Trains and compares the variants |
| `make_variant_video.py` | Renders the stage 2 video |
| `make_videos.py` | Renders random vs trained side by side |
| `notebooks/stage1_land_a_rocket.ipynb` | Stage 1 step by step, runs in Colab |
| `models/lander_ppo.zip` | My trained AI |

Built with [Gymnasium](https://gymnasium.farama.org/) (Lunar Lander) and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO).

## Follow along

I post every step on LinkedIn. Built by Arsenii Cherednichenko.

MIT License.
