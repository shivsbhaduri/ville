"""Twenty metrics on one experiment, with e-BH for the false discovery rate."""

import numpy as np

import ville
from ville.evalues import bh, e_to_p, ebh

rng = np.random.default_rng(5)
pairs, metrics = 20_000, 20
effects = np.zeros(metrics)
effects[:4] = [0.10, 0.08, 0.06, 0.04]

# Each user has one shock shared by all their metrics, so the twenty
# e-values are correlated. e-BH does not care.
shock_t = 0.5 * rng.standard_normal(pairs)
shock_c = 0.5 * rng.standard_normal(pairs)
e_values = np.empty(metrics)
for k in range(metrics):
    y_t = shock_t + rng.standard_normal(pairs) + effects[k]
    y_c = shock_c + rng.standard_normal(pairs)
    test = ville.MSPRT(theta0=0.0, sigma=np.sqrt(2 * 1.25), t_opt=pairs).fit(y_t - y_c)
    e_values[k] = test.e_values_[-1]

print("metric  true effect     e-value")
for k in range(7):
    print(f"{k:>6}  {effects[k]:11.2f}  {e_values[k]:10.3g}")
print("   ...")
print("Rejected by e-BH at 10%:", np.flatnonzero(ebh(e_values, alpha=0.1)).tolist())
print("Rejected by BH on p = 1/e at 10%:", np.flatnonzero(bh(e_to_p(e_values), alpha=0.1)).tolist())
