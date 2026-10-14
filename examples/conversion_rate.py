"""Monitor a conversion rate and an A/B difference in conversion rates."""

import numpy as np

import ville

rng = np.random.default_rng(42)
n = 40_000
treated = (rng.random(n) < 0.5).astype(float)
converted = (rng.random(n) < np.where(treated == 1, 0.050, 0.040)).astype(float)

control = ville.BernoulliCS(alpha=0.05).fit(converted[treated == 0])
print("Control conversion rate")
for t in (500, 2000, 10_000, control.n_):
    print(f"  after {t:>6,} users: {control.estimate_[t - 1]:.4f}  "
          f"[{control.lower_[t - 1]:.4f}, {control.upper_[t - 1]:.4f}]")

test = ville.BettingABTest(bounds=(0, 1), alpha=0.05).fit(converted, treated)
fixed = ville.AsymptoticABTest(alpha=0.05).fit(converted, treated)
print("\nDifference in conversion rate, treatment minus control")
for t in (5000, 10_000, 20_000, n):
    print(f"  after {t:>6,} users: {test.estimate_[t - 1]:+.4f}  "
          f"[{test.lower_[t - 1]:+.4f}, {test.upper_[t - 1]:+.4f}]")
print(f"\nZero first excluded after {test.stopping_time():,} users (betting)")
print(f"Zero first excluded after {fixed.stopping_time():,} users (asymptotic)")
