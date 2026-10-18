"""The fuel result in one clean picture: raise the fuel price, and see what
happens to how often it lands (top) and how much engine it uses (bottom).

Reads docs/fuel_results.csv (made by fuel_chart.py).
Run:  python fuel_pareto.py  ->  docs/fuel_pareto.png
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open("docs/fuel_results.csv"))]
prices = sorted({r["price"] for r in rows})
avg = lambda key, p: sum(r[key] for r in rows if r["price"] == p) / sum(r["price"] == p for r in rows)
lands = [avg("lands", p) * 100 for p in prices]
engine = [avg("engine_s", p) for p in prices]
names = [("1x" if p == 0.3 else f"{p / 0.3:.1f}x".replace(".0x", "x")) for p in prices]
xs = list(range(len(prices)))  # tested prices, evenly spaced
sweet = names.index("2x")

INK, INK2, SURFACE, LINE, BAND = "#14171c", "#6b7280", "#ffffff", "#2a78d6", "#e8f3ec"

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.8, 13.5), dpi=100, sharex=True,
                                  gridspec_kw=dict(hspace=0.25))
fig.patch.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.13, right=0.94, top=0.78, bottom=0.11)

for ax, ys, fmt in [(top, lands, "{:.0f}%"), (bottom, engine, "{:.1f} s")]:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d6d8dc")
    ax.tick_params(colors=INK2, labelsize=16, length=0, pad=10)
    ax.grid(axis="y", color="#eef0f2", lw=1)
    ax.set_axisbelow(True)
    ax.axvspan(-0.4, sweet + 0.4, color=BAND, lw=0, zorder=0)
    ax.plot(xs, ys, color=LINE, lw=3.5, zorder=2)
    ax.scatter(xs, ys, s=90, color=LINE, edgecolor=SURFACE, linewidth=2.5, zorder=3)
    for i in (0, sweet, len(xs) - 1):
        ax.text(xs[i], ys[i] + (max(ys) * 0.07), fmt.format(ys[i]), ha="center",
                fontsize=17, color=INK, weight="bold" if i == sweet else "normal")
    ax.set_xlim(-0.5, len(xs) - 0.5)

top.set_ylim(-8, 118)
top.set_yticks([0, 25, 50, 75, 100])
top.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
bottom.set_ylim(0, 4.6)
bottom.set_yticks([0, 1, 2, 3, 4])
bottom.set_yticklabels(["0 s", "1 s", "2 s", "3 s", "4 s"])
top.set_title("Lands on the pad", loc="left", fontsize=19, color=INK2, pad=14)
bottom.set_title("Engine time per landing", loc="left", fontsize=19, color=INK2, pad=14)
top.text(sweet / 2, 55, "safe zone", ha="center", fontsize=16, color="#2f6b45")
bottom.set_xticks(xs)
bottom.set_xticklabels(names)
bottom.set_xlabel("Fuel price", fontsize=18, color=INK2, labelpad=12)

fig.text(0.1, 0.92, "Expensive fuel saves fuel,", fontsize=32, color=INK, weight="bold")
fig.text(0.1, 0.885, "until the AI starts crashing", fontsize=32, color=INK, weight="bold")
fig.text(0.1, 0.85, "Up to 2x the price it still lands every time, with a third less engine.",
         fontsize=17, color=INK2)
fig.text(0.1, 0.015, "Project Touchdown. Averages of 150 test flights per price (50 from 5x up).",
         fontsize=13, color=INK2)
fig.savefig("docs/fuel_pareto.png", facecolor=SURFACE)
print("docs/fuel_pareto.png")
