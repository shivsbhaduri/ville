"""False positive rate of a z test looked at daily, against ville's
asymptotic confidence sequence, on A/B tests with no true effect."""

from pathlib import Path

import numpy as np

import ville
from _style import GREY, INK, RED, TEAL, setup

plt = setup()
OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

out = ville.simulate.peeking(n_max=10_000, n_looks=30, n_sims=4000, alpha=0.05, seed=2026)
days = np.arange(1, 31)

print("day  users  z test  ville")
for d in (1, 2, 5, 10, 20, 30):
    print(f"{d:>3}  {out['looks'][d - 1]:>5}  {out['fixed'][d - 1]:6.3f}  {out['anytime'][d - 1]:.3f}")

fig, ax = plt.subplots()
ax.plot(days, out["fixed"], color=RED, lw=2.4, marker="o", ms=4, label="z test, checked every day")
ax.plot(days, out["anytime"], color=TEAL, lw=2.4, marker="o", ms=4, label="ville.AsymptoticABTest")
ax.axhline(0.05, color=INK, lw=1, ls="--")
ax.text(30, 0.056, "5% target", ha="right", va="bottom", color=INK)
ax.set_xlim(0.5, 30.5)
ax.set_ylim(0, max(out["fixed"]) * 1.12)
ax.set_xlabel("Day of the experiment (about 333 users arrive each day)")
ax.set_ylabel("Share of null experiments declared significant")
ax.set_title("Daily peeking at a z test against a confidence sequence")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(loc="upper left")
fig.text(0.01, -0.02, "4,000 simulated A/B tests with no true effect, N(0, 1) outcomes, alpha = 0.05.",
         color=GREY, fontsize=9)
fig.savefig(OUT / "peeking.png")
