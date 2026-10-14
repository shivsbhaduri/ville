# Quick start

All confidence sequences share one interface. `fit(x)` processes a stream from the start, `update(x)` appends observations, and `estimate_`, `lower_` and `upper_` hold one value per observation.

```python
import numpy as np
import ville

rng = np.random.default_rng(42)
n = 40_000
treated = (rng.random(n) < 0.5).astype(float)
converted = (rng.random(n) < np.where(treated == 1, 0.050, 0.040)).astype(float)

control = ville.BernoulliCS(alpha=0.05).fit(converted[treated == 0])
test = ville.BettingABTest(bounds=(0, 1), alpha=0.05).fit(converted, treated)

for t in (5000, 10_000, 20_000, n):
    print(t, test.estimate_[t - 1], test.lower_[t - 1], test.upper_[t - 1])
print(test.stopping_time())
```

Output from [`examples/conversion_rate.py`](https://github.com/shivsbhaduri/ville/blob/main/examples/conversion_rate.py):

```text
Control conversion rate
  after    500 users: 0.0480  [0.0214, 0.0699]
  after  2,000 users: 0.0410  [0.0277, 0.0605]
  after 10,000 users: 0.0408  [0.0335, 0.0492]
  after 20,147 users: 0.0399  [0.0348, 0.0459]

Difference in conversion rate, treatment minus control
  after  5,000 users: +0.0126  [-0.0073, +0.0331]
  after 10,000 users: +0.0118  [-0.0025, +0.0271]
  after 20,000 users: +0.0101  [-0.0001, +0.0223]
  after 40,000 users: +0.0098  [+0.0019, +0.0189]

Zero first excluded after 20,579 users (betting)
Zero first excluded after 16,532 users (asymptotic)
```

The intervals hold simultaneously over all $t$, so they may be inspected at any time, and the decision to continue may depend on what was observed. The true difference is 0.010.

Observations may also be supplied in batches:

```python
cs = ville.EmpiricalBernsteinCS(bounds=(0, 1))
for batch in daily_batches:
    cs.update(batch)
    lo, hi = cs.interval()
```
