"""Trade-off chart from docs/fuel_results.csv (made by fuel_chart.py):
each dot is one trained AI, fuel used vs how often it lands.

Run:  python fuel_pareto.py  ->  docs/fuel_pareto.png
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open("docs/fuel_results.csv"))]

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"
BLUE, GREY = "#2a78d6", "#a8a7a1"

fig, ax = plt.subplots(figsize=(10.8, 13.5), dpi=100)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.12, right=0.95, top=0.8, bottom=0.09)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c9c8c3")
ax.tick_params(colors=INK2, labelsize=17, length=0, pad=8)
ax.grid(color=GRID, lw=1)
ax.set_axisbelow(True)

# the goal: little fuel, lands every time
ax.axhspan(99, 103, color="#e4efe7", lw=0, zorder=0)
ax.text(4.85, 101, "lands every time", ha="right", va="center", fontsize=16, color="#2f6b45")

def group(price):
    if abs(price - 0.3) < 1e-9:
        return "normal price (1x)", "#444441", 130
    if abs(price - 0.6) < 1e-9:
        return "2x price", BLUE, 170
    return "more than 2x", GREY, 100

seen = set()
for r in rows:
    name, color, size = group(r["price"])
    ax.scatter(r["engine_s"], r["lands"] * 100, s=size, color=color, edgecolor=SURFACE,
               linewidth=2, zorder=3, label=None if name in seen else name)
    seen.add(name)
ax.legend(loc="lower right", fontsize=16, frameon=False, labelcolor=INK2)
for price, name, dx in [(1.5, "5x", 0.1), (2.0, "6.7x", 0.1), (3.0, "10x", 0.1)]:
    r = next(r for r in rows if abs(r["price"] - price) < 1e-9)
    ax.text(r["engine_s"] + dx, r["lands"] * 100, name, fontsize=16, color=INK2, va="center")

ax.set_xlim(0.2, 4.9)
ax.set_ylim(-5, 106)
ax.set_xlabel("Engine time per landing (s): less is cheaper", fontsize=18, color=INK2, labelpad=12)
ax.set_ylabel("Lands on the pad (%)", fontsize=18, color=INK2, labelpad=10)

ax.text(0.35, 64, "2x is the sweet spot: all 3 runs\nland every time, with about a third\nless engine than normal.\n\nGoing cheaper than 2x barely saves\nmore fuel, but it starts crashing.",
        fontsize=17, color=INK, va="top")

fig.text(0.12, 0.92, "Less fuel or a safe landing?", fontsize=32, color=INK, weight="bold")
fig.text(0.12, 0.875, "Each dot is one trained AI, tested on 50 flights. 3 runs at 1x to 3.3x.",
         fontsize=17, color=INK2)
fig.text(0.12, 0.845, "Project Touchdown, stage 2.", fontsize=17, color=INK2)
fig.savefig("docs/fuel_pareto.png", facecolor=SURFACE)
print("docs/fuel_pareto.png")
