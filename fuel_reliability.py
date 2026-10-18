"""The fuel result the way engineers report reliability: for each fuel price,
the share of successful landings with a 95% confidence range, checked against
a requirement. Reads docs/fuel_results.csv (made by fuel_chart.py).

Why a range and not one number: test 150 flights and you only estimate the
true landing rate. Train or test again and you get a slightly different
number. The range shows how sure we can be.

Run:  python fuel_reliability.py  ->  docs/fuel_reliability.png
"""
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLIGHTS_PER_RUN = 50
REQUIREMENT = 95  # an example requirement: land at least 95% of the time

rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open("docs/fuel_results.csv"))]


def wilson(successes, n, z=1.96):
    """95% confidence range for a success rate (Wilson score interval)."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


prices = sorted({r["price"] for r in rows})
stats = []
for price in prices:
    runs = [r for r in rows if r["price"] == price]
    n = FLIGHTS_PER_RUN * len(runs)
    landed = round(sum(r["lands"] for r in runs) * FLIGHTS_PER_RUN)
    lo, hi = wilson(landed, n)
    engine = sum(r["engine_s"] for r in runs) / len(runs)
    stats.append(dict(price=price, n=n, rate=landed / n * 100, lo=lo * 100, hi=hi * 100,
                      engine=engine, passes=lo * 100 >= REQUIREMENT))

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"
PASS, FAIL = "#2a78d6", "#a8a7a1"

fig, ax = plt.subplots(figsize=(10.8, 13.5), dpi=100)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.2, right=0.8, top=0.78, bottom=0.1)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c3")
ax.tick_params(axis="x", colors=INK2, labelsize=16, length=0, pad=8)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=1)
ax.set_axisbelow(True)

ys = list(range(len(stats)))[::-1]
for y, s in zip(ys, stats):
    color = PASS if s["passes"] else FAIL
    ax.plot([s["lo"], s["hi"]], [y, y], color=color, lw=5, solid_capstyle="round", zorder=2)
    ax.scatter(s["rate"], y, s=160, color=color, edgecolor=SURFACE, linewidth=2.5, zorder=3)
    name = f"{s['price'] / 0.3:.1f}x".replace(".0x", "x")
    if s["price"] == 0.3:
        name = "1x normal"
    ax.text(-3, y, name, ha="right", va="center", fontsize=18,
            color=INK if s["passes"] else INK2, weight="bold" if s["price"] == 0.6 else "normal")
    ax.text(106, y, f"{s['engine']:.1f} s", ha="left", va="center", fontsize=17, color=INK2)
    ax.text(106, y - 0.32, f"{s['n']} flights", ha="left", va="center", fontsize=12, color=INK2)

ax.axvline(REQUIREMENT, color=INK, lw=1.2, ls=(0, (5, 4)), zorder=1)
ax.text(REQUIREMENT, len(stats) - 0.35, f"requirement\n{REQUIREMENT}%", ha="center",
        va="bottom", fontsize=15, color=INK)
ax.text(-3, len(stats) - 0.35, "Fuel price", ha="right", va="bottom", fontsize=15, color=INK2)
ax.text(106, len(stats) - 0.35, "Engine\ntime", ha="left", va="bottom", fontsize=15, color=INK2)

ax.set_xlim(0, 104)
ax.set_ylim(-0.8, len(stats) + 0.6)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Lands on the pad", fontsize=18, color=INK2, labelpad=12)

fig.text(0.06, 0.93, "Which fuel prices land reliably?", fontsize=31, color=INK, weight="bold")
fig.text(0.06, 0.895, "Dot: share of test flights that landed. Bar: 95% confidence range.",
         fontsize=16, color=INK2)
fig.text(0.06, 0.87, "Blue: even the low end of the range meets the requirement.",
         fontsize=16, color=INK2)
fig.text(0.06, 0.03, "Project Touchdown, stage 2. 3 trained AIs per price up to 3.3x, 1 above.",
         fontsize=13, color=INK2)
fig.savefig("docs/fuel_reliability.png", facecolor=SURFACE)

for s in stats:
    print(f"{s['price']:>4}  {s['rate']:5.1f}%  range {s['lo']:5.1f} to {s['hi']:5.1f}  "
          f"engine {s['engine']:.1f}s  {'meets' if s['passes'] else 'no'}")
