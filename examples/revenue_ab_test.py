"""A/B test on a skewed, unbounded metric (revenue per visitor)."""

import numpy as np

import ville

rng = np.random.default_rng(7)
n = 100_000
treated = (rng.random(n) < 0.5).astype(float)
buys = rng.random(n) < np.where(treated == 1, 0.10, 0.08)
basket = rng.lognormal(mean=3.5, sigma=0.9, size=n)
revenue = np.where(buys, basket, 0.0)

test = ville.AsymptoticABTest(alpha=0.05, t_opt=20_000).fit(revenue, treated)
print(test)
stop = test.stopping_time()
print(f"Stopped after {stop:,} visitors")
lo, hi = test.lower_[stop - 1], test.upper_[stop - 1]
print(f"Effect at stopping: {test.estimate_[stop - 1]:+.3f} per visitor, 95% CS [{lo:+.3f}, {hi:+.3f}]")
true_effect = 0.02 * np.exp(3.5 + 0.9**2 / 2)
print(f"True effect: {true_effect:+.3f} per visitor")
