"""Chart for the fuel price sweep: how reliable is each price, and how much
engine does it use? Tests every trained AI (50 flights each), saves the raw
numbers to docs/fuel_results.csv and the chart to docs/fuel_tradeoff.png.

Run:  python fuel_chart.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sweep_fuel import PRICES, path, test

ROWS = []
for price in sorted(set(PRICES) | {0.7, 0.8, 0.9}):
    for seed in range(10):
        if os.path.exists(path(price, seed)):
            land, engine = test(price, seed)
            ROWS.append(dict(price=price, seed=seed, lands=land, engine_s=engine))
            print(price, seed, f"{land:.0%}", f"{engine:.1f}s")

os.makedirs("docs", exist_ok=True)
with open("docs/fuel_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["price", "seed", "lands", "engine_s"])
    w.writeheader()
    w.writerows(ROWS)

INK, INK2, GRID, SURFACE, SERIES = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb", "#2a78d6"
prices = sorted({r["price"] for r in ROWS})
mean = lambda key, p: np.mean([r[key] for r in ROWS if r["price"] == p])

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.8, 13.5), dpi=100, sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 2], hspace=0.12))
fig.patch.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.12, right=0.95, top=0.84, bottom=0.08)

for ax in (top, bottom):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=17, length=0, pad=8)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)

# Top: landing rate, one dot per training run, line through the averages
xs = [r["price"] / 0.3 for r in ROWS]
top.scatter(xs, [r["lands"] * 100 for r in ROWS], s=90, color=SERIES, alpha=0.35,
            edgecolor=SURFACE, linewidth=2, zorder=3)
top.plot([p / 0.3 for p in prices], [mean("lands", p) * 100 for p in prices],
         color=SERIES, lw=3, zorder=2)
top.axhspan(99, 101.5, color="#e4efe7", lw=0, zorder=0)
top.text(9.9, 96.5, "lands every time", ha="right", va="top", fontsize=17, color="#2f6b45")
top.set_ylim(-4, 104)
top.set_ylabel("Lands on the pad (%)", fontsize=18, color=INK2, labelpad=10)
top.text(1.0, 40, "Up to 2x the price:\nlands every time,\na third less engine",
         fontsize=17, color=INK, va="top")
top.text(1.0, 17, "Around 3x: runs disagree,\n74% to 98%",
         fontsize=17, color=INK, va="top")

# Bottom: engine time
bottom.plot([p / 0.3 for p in prices], [mean("engine_s", p) for p in prices],
            color=SERIES, lw=3, marker="o", markersize=9, markeredgecolor=SURFACE,
            markeredgewidth=2)
bottom.set_ylabel("Engine time (s)", fontsize=18, color=INK2, labelpad=10)
bottom.set_ylim(0, 4.2)
bottom.set_xlabel("Fuel price (1 = normal)", fontsize=18, color=INK2, labelpad=12)
bottom.set_xticks([1, 2, 3, 5, 6.7, 10])
bottom.set_xticklabels(["1x", "2x", "3x", "5x", "6.7x", "10x"])

fig.text(0.12, 0.93, "How cheap can landing get?", fontsize=32, color=INK, weight="bold")
fig.text(0.12, 0.885, "Each dot is one trained AI, tested on 50 flights. Project Touchdown, stage 2.",
         fontsize=17, color=INK2)
fig.savefig("docs/fuel_tradeoff.png", facecolor=SURFACE)
print("docs/fuel_tradeoff.png")
