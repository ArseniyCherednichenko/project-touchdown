"""The fuel result in one clean picture: raise the fuel price, and see what
happens to how often it lands (top) and how much engine it uses (bottom).

The x-axis is to scale (logarithmic, so 1x to 3x is not squeezed). Every
tested price is a point, labelled with its measured value.

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
xs = [p / 0.3 for p in prices]                      # price as a multiple of normal
lands = [avg("lands", p) * 100 for p in prices]
engine = [avg("engine_s", p) for p in prices]
names = [f"{x:.1f}x".replace(".0x", "x") for x in xs]

INK, INK2, SURFACE, LINE, BAND, GREEN = "#14171c", "#6b7280", "#ffffff", "#2a78d6", "#e8f3ec", "#2f6b45"

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.8, 13.5), dpi=100, sharex=True,
                                  gridspec_kw=dict(hspace=0.3))
fig.patch.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.13, right=0.95, top=0.8, bottom=0.11)

for ax, ys, fmt, dy in [(top, lands, "{:.0f}%", 6), (bottom, engine, "{:.1f}", 0.22)]:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d6d8dc")
    ax.tick_params(colors=INK2, labelsize=15, length=0, pad=8)
    ax.grid(axis="y", color="#eef0f2", lw=1)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.axvspan(0.88, 2.12, color=BAND, lw=0, zorder=0)
    ax.plot(xs, ys, color=LINE, lw=3, zorder=2)
    ax.scatter(xs, ys, s=80, color=LINE, edgecolor=SURFACE, linewidth=2.5, zorder=3)
    for x, y in zip(xs, ys):
        # the middle is crowded: between 2x and 3x the dots speak for themselves
        if round(x, 1) in (2.3, 2.7, 3.3):
            continue
        ax.text(x, y + dy, fmt.format(y), ha="center", fontsize=14, color=INK)

top.set_ylim(-8, 120)
top.set_yticks([0, 25, 50, 75, 100])
top.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
top.set_title("How often it lands on the pad", loc="left", fontsize=19, color=INK2, pad=14)
top.text(1.38, 45, "lands every\ntime", ha="center", fontsize=15, color=GREEN)
top.annotate("starts crashing", xy=(5, 46), xytext=(6.2, 78), fontsize=15, color=INK,
             arrowprops=dict(arrowstyle="-", color="#b8bcc4", lw=1))

bottom.set_ylim(0, 4.6)
bottom.set_yticks([0, 1, 2, 3, 4])
bottom.set_yticklabels(["0 s", "1 s", "2 s", "3 s", "4 s"])
bottom.set_title("How long it fires the engine per landing", loc="left", fontsize=19, color=INK2, pad=14)
bottom.annotate("a third less\nthan normal", xy=(2, 2.5), xytext=(2.6, 3.6), fontsize=15, color=GREEN,
                arrowprops=dict(arrowstyle="-", color="#b8bcc4", lw=1))

ticks = [1, 2, 3, 5, 10]
bottom.set_xticks(ticks)
bottom.set_xticklabels([f"{t}x" for t in ticks])
bottom.minorticks_off()
bottom.set_xlabel("Fuel price (1x = normal)", fontsize=17, color=INK2, labelpad=12)

fig.text(0.13, 0.925, "Expensive fuel saves fuel,", fontsize=31, color=INK, weight="bold")
fig.text(0.13, 0.89, "until the AI starts crashing", fontsize=31, color=INK, weight="bold")
fig.text(0.13, 0.855, "Sweet spot: 2x the price. Still lands every time, with a third less engine.",
         fontsize=16, color=INK2)
fig.text(0.13, 0.025, "Project Touchdown. Averages of 150 test flights per price (50 from 5x up).",
         fontsize=12, color=INK2)
fig.savefig("docs/fuel_pareto.png", facecolor=SURFACE)
print("docs/fuel_pareto.png")
