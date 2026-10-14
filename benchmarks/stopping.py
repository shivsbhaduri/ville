"""How soon a continuously monitored confidence sequence detects real
effects, against a fixed horizon z test sized for 80% power."""

from pathlib import Path

import numpy as np
from scipy.stats import norm

import ville
from _style import AMBER, BLUE, GREY, INK, TEAL, setup

plt = setup()
OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

ALPHA, POWER, DESIGN = 0.05, 0.8, 0.05
# Two arm z test with a 50/50 split and unit variance: Var(diff) = 4 / n.
n_fixed = int(np.ceil(4 * (norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)) ** 2 / DESIGN**2))
N_MAX = 4 * n_fixed

effects = {0.04: AMBER, 0.05: BLUE, 0.08: TEAL}
print(f"fixed horizon test for effect {DESIGN}: n = {n_fixed}")
print(f"{'effect':>7}  {'stopped by n_fixed':>18}  {'median stop':>11}  {'fixed n power':>13}")

fig, ax = plt.subplots()
grid = np.arange(1, N_MAX + 1)
for effect, colour in effects.items():
    tau = ville.simulate.stopping_times(n_max=N_MAX, n_sims=1000, effect=effect,
                                        alpha=ALPHA, t_opt=n_fixed, seed=11)
    share = np.searchsorted(np.sort(tau), grid, side="right") / tau.size
    fixed_power = norm.cdf(effect * np.sqrt(n_fixed) / 2 - norm.ppf(1 - ALPHA / 2))
    median = np.median(tau)
    print(f"{effect:7.2f}  {np.mean(tau <= n_fixed):18.3f}  {median:11.0f}  {fixed_power:13.3f}")
    ax.plot(grid, share, color=colour, lw=2.4, label=f"effect {effect} (fixed n power {fixed_power:.0%})")

ax.axvline(n_fixed, color=INK, lw=1, ls="--")
ax.text(n_fixed * 1.02, 0.04, f"fixed horizon n = {n_fixed:,}\n(80% power at effect {DESIGN})", color=INK)
ax.set_xlim(0, N_MAX)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Users observed")
ax.set_ylabel("Share of experiments stopped")
ax.set_title("When ville.AsymptoticABTest stops on real effects, monitoring after every user")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.legend(loc="lower right")
fig.text(0.01, -0.02, "1,000 simulated A/B tests per effect, N(0, 1) outcomes, alpha = 0.05, t_opt = fixed horizon n.",
         color=GREY, fontsize=9)
fig.savefig(OUT / "stopping.png")
