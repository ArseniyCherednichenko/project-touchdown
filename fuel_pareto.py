"""Trade-off chart (Pareto front) from docs/fuel_results.csv: cost on the
x-axis (engine time per landing), performance on the y-axis (landing rate).

One point per fuel price, with its 95% confidence range. The line is the
Pareto front: the options where you cannot land more often without burning
more fuel. A point below or right of it is a worse deal than one on it.

Run:  python fuel_pareto.py  ->  docs/fuel_pareto.png
"""
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLIGHTS_PER_RUN = 50


def wilson(successes, n, z=1.96):
    """95% confidence range for a success rate (Wilson score interval)."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open("docs/fuel_results.csv"))]
points = []
for price in sorted({r["price"] for r in rows}):
    runs = [r for r in rows if r["price"] == price]
    n = FLIGHTS_PER_RUN * len(runs)
    landed = round(sum(r["lands"] for r in runs) * FLIGHTS_PER_RUN)
    lo, hi = wilson(landed, n)
    points.append(dict(price=price, x=sum(r["engine_s"] for r in runs) / len(runs),
                       y=landed / n * 100, lo=lo * 100, hi=hi * 100))

# Pareto front: no other point uses less (or equal) fuel AND lands more often.
front = [p for p in points
         if not any(q is not p and q["x"] <= p["x"] and q["y"] >= p["y"]
                    and (q["x"] < p["x"] or q["y"] > p["y"]) for q in points)]
front.sort(key=lambda p: p["x"])

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"
BLUE, GREY = "#2a78d6", "#a8a7a1"

fig, ax = plt.subplots(figsize=(10.8, 13.5), dpi=100)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.12, right=0.95, top=0.8, bottom=0.1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9c8c3")
ax.tick_params(colors=INK2, labelsize=16, length=0, pad=8)
ax.grid(color=GRID, lw=1)
ax.set_axisbelow(True)

ax.plot([p["x"] for p in front], [p["y"] for p in front], color=BLUE, lw=2.5, zorder=2,
        label="Pareto front: best landing rate\nfor that much fuel")
for p in points:
    color = BLUE if p in front else GREY
    ax.plot([p["x"], p["x"]], [p["lo"], p["hi"]], color=color, lw=3, alpha=0.5,
            solid_capstyle="round", zorder=2)
    ax.scatter(p["x"], p["y"], s=150 if p["price"] == 0.6 else 100, color=color,
               edgecolor=SURFACE, linewidth=2, zorder=3)


def label(price, text, dx, dy, ha="left", bold=False):
    p = next(p for p in points if abs(p["price"] - price) < 1e-9)
    ax.annotate(text, (p["x"], p["y"]), xytext=(p["x"] + dx, p["y"] + dy), fontsize=16,
                color=BLUE if bold else INK2, weight="bold" if bold else "normal", ha=ha,
                va="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))


label(0.6, "2x price: sweet spot", -0.3, 8, ha="right", bold=True)
label(0.3, "1x normal: just as reliable,\nbut more fuel. Not on the front", 0.2, -30, ha="right")
label(0.9, "3x", -0.4, -6, ha="right")
label(1.5, "5x", 0.3, 0)
label(2.0, "6.7x", 0.3, 0)
label(3.0, "10x", 0.3, 4)

ax.annotate("better", xy=(0.25, 111), xytext=(0.25, 96), fontsize=14, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.annotate("cheaper", xy=(0.15, 60), xytext=(0.95, 60), fontsize=14, color=INK2, va="center",
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))

ax.set_xlim(0, 4.1)
ax.set_ylim(-5, 114)
ax.set_xlabel("Cost: engine time per landing (s)", fontsize=18, color=INK2, labelpad=12)
ax.set_ylabel("Performance: lands on the pad (%)", fontsize=18, color=INK2, labelpad=10)
ax.legend(loc="lower right", fontsize=15, frameon=False, labelcolor=INK2)

fig.text(0.12, 0.92, "Fuel or reliability: the trade-off", fontsize=31, color=INK, weight="bold")
fig.text(0.12, 0.88, "One point per fuel price, with its 95% confidence range.", fontsize=16, color=INK2)
fig.text(0.12, 0.855, "Grey: beaten by a blue point that uses less fuel and lands as often.",
         fontsize=16, color=INK2)
fig.text(0.12, 0.03, "Project Touchdown, stage 2. 150 test flights per price up to 3.3x, 50 above.",
         fontsize=13, color=INK2)
fig.savefig("docs/fuel_pareto.png", facecolor=SURFACE)
print("front:", [f"{p['price'] / 0.3:.1f}x" for p in front])
