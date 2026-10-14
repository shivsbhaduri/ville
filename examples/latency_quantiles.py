"""Track the median and 95th percentile of response times as they stream in."""

import numpy as np

import ville

rng = np.random.default_rng(11)
latency_ms = rng.lognormal(mean=4.0, sigma=0.6, size=10_000)
true_p50 = np.exp(4.0)
true_p95 = np.exp(4.0 + 0.6 * 1.6448536)

for q, truth in ((0.5, true_p50), (0.95, true_p95)):
    cs = ville.QuantileCS(q=q, alpha=0.05, t_opt=2000).fit(latency_ms)
    print(f"q = {q}: true {truth:.1f} ms")
    for t in (200, 1000, 10_000):
        print(f"  after {t:>6,} requests: {cs.estimate_[t - 1]:7.1f} ms  "
              f"[{cs.lower_[t - 1]:7.1f}, {cs.upper_[t - 1]:7.1f}]")

var = ville.VarianceCS(bounds=(0, 1), alpha=0.05).fit(rng.random(10_000))
print(f"Variance of U(0, 1), true {1 / 12:.4f}: estimate {var.estimate_[-1]:.4f}, "
      f"95% CS [{var.lower_[-1]:.4f}, {var.upper_[-1]:.4f}]")
