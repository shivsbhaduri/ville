"""Average width of each confidence sequence for the mean of Beta(2, 5)
data, with the fixed n CLT interval for reference."""

from pathlib import Path

import numpy as np
from scipy.stats import norm

import ville
from _style import AMBER, BLUE, GREY, INK, RED, TEAL, setup

plt = setup()
OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

N, SIMS, ALPHA = 20_000, 40, 0.05
methods = {
    "GaussianMixtureCS (sigma = 1/2)": (lambda: ville.GaussianMixtureCS(sigma=0.5, alpha=ALPHA, t_opt=1000), GREY),
    "AsymptoticCS": (lambda: ville.AsymptoticCS(alpha=ALPHA, t_opt=1000, intersect=True), AMBER),
    "EmpiricalBernsteinCS": (lambda: ville.EmpiricalBernsteinCS(alpha=ALPHA), BLUE),
    "BettingCS": (lambda: ville.BettingCS(alpha=ALPHA), TEAL),
}

t = np.arange(1, N + 1)
widths = {name: np.zeros(N) for name in methods}
clt = np.zeros(N)
for seed in range(SIMS):
    x = np.random.default_rng(seed).beta(2, 5, N)
    for name, (make, _) in methods.items():
        cs = make().fit(x)
        widths[name] += (cs.upper_ - cs.lower_) / SIMS
    sd = np.sqrt(np.maximum(np.cumsum(x**2) / t - (np.cumsum(x) / t) ** 2, 0))
    clt += 2 * norm.ppf(1 - ALPHA / 2) * sd / np.sqrt(t) / SIMS

print(f"{'n':>6}  " + "  ".join(f"{k.split(' ')[0]:>20}" for k in methods) + f"  {'CLT (fixed n)':>14}")
for n in (100, 1000, 10_000, 20_000):
    print(f"{n:>6}  " + "  ".join(f"{widths[k][n - 1]:20.4f}" for k in methods) + f"  {clt[n - 1]:14.4f}")

fig, ax = plt.subplots()
start = 20
for name, (_, colour) in methods.items():
    ax.plot(t[start:], widths[name][start:], color=colour, lw=2.2, label=name)
ax.plot(t[start:], clt[start:], color=RED, lw=1.6, ls="--", label="CLT interval (fixed n only, not time uniform)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Observations")
ax.set_ylabel("Average interval width")
ax.set_title("Width of 95% confidence sequences for a bounded mean")
ax.legend(loc="lower left")
fig.text(0.01, -0.02, f"Beta(2, 5) data on [0, 1], true sd 0.16, average over {SIMS} streams.",
         color=GREY, fontsize=9)
fig.savefig(OUT / "widths.png")
