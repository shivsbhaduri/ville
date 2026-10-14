"""E-processes, anytime valid p-values and optional stopping."""

import numpy as np

import ville

rng = np.random.default_rng(3)

x = rng.normal(0.15, 1.0, 5000)
msprt = ville.MSPRT(theta0=0.0, sigma=1.0, alpha=0.05, t_opt=1000).fit(x)
print(msprt)
print(f"  e-value first reached 20 at t = {msprt.stopping_time()}")
for t in (100, 200, 300, 1000):
    print(f"  anytime valid p-value at t = {t:>4}: {msprt.p_values_[t - 1]:.2e}")

scores = rng.beta(6, 4, 3000)
bet = ville.BettingTest(m0=0.55, bounds=(0, 1), alternative="greater").fit(scores)
print(bet)
print(f"  rejected mean <= 0.55 at t = {bet.stopping_time()} (true mean 0.6)")

null_runs = [ville.BettingTest(m0=0.5).fit(rng.random(2000)).rejected_ for _ in range(500)]
print(f"Null runs ever rejected at alpha = 0.05: {np.mean(null_runs):.3f}")
