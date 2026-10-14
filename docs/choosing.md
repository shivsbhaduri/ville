# Choosing a method

| Your data | Use |
|---|---|
| Binary outcomes (conversion, click, churn) | `BernoulliCS`, `BernoulliTest`; `BettingABTest` for a difference |
| Bounded outcomes with a known range (scores, ratings, capped values) | `BettingCS`; `EmpiricalBernsteinCS` when speed matters |
| Unbounded or heavy tailed (revenue, time spent) | `AsymptoticCS`, `AsymptoticABTest` |
| Gaussian measurements with known noise level | `GaussianMixtureCS`, `MSPRT` |
| Medians, percentiles, latency | `QuantileCS` |
| Spread of a bounded metric | `VarianceCS` |
| Many hypotheses at once | e-values from any of the above, then `evalues.ebh` |

`t_opt` should be close to the sample size at which most decisions are expected. It affects width, not validity.

## Assumptions and limitations

* **Independence.** The confidence sequences assume i.i.d. observations, or for the A/B tests units that are independently randomised with a known probability. The betting and normal mixture constructions hold under weaker martingale conditions (a constant conditional mean, for example), but the documentation and tests here cover the i.i.d. case.
* **Bounds.** `BettingCS`, `EmpiricalBernsteinCS`, `VarianceCS` and `BettingABTest` raise an error on observations outside the stated range. Bounds chosen after inspecting the data invalidate the guarantee.
* **Asymptotic validity.** `AsymptoticCS` and `AsymptoticABTest` have the same large sample guarantee as the CLT interval. Increase `min_samples` for heavily skewed data.
* **Bias at stopping times.** Point estimates at data dependent stopping times are biased. Report the confidence sequence at the stopping time.
* **Grid resolution.** `BettingCS` is computed on a grid of candidate means and widened to the adjacent grid points. Increase `grid` if the interval width approaches the grid spacing.
* **`VarianceCS` efficiency.** Exactly valid, but uses half the available information.
* **Design issues.** Sample ratio mismatch, novelty effects and interference between units violate the randomisation assumptions and are not detected by these methods.
