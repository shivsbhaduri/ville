# ville

Confidence sequences, sequential tests and e-values for anytime valid inference in Python.

Classical confidence intervals and p-values assume a sample size fixed in advance and a single analysis. Under repeated analysis of accumulating data (continuous monitoring, interim looks, optional stopping) the type I error rate exceeds the nominal level. The methods implemented here control error rates uniformly over time: the coverage and type I error guarantees hold simultaneously for all sample sizes, so the data may be monitored continuously and the experiment stopped or extended at any point.

The package is named after Jean Ville, whose maximal inequality for nonnegative supermartingales (Ville 1939) underlies all of the methods.

![Daily peeking at a z test against a confidence sequence](figures/peeking.png)

*Figure 1. Cumulative false positive rate over 4,000 simulated A/B tests with no true effect, analysed once per day for 30 days. Two sample z test at α = 0.05: 29.4% by day 30. `ville.AsymptoticABTest` at the same looks: 1.6%. Script: [`benchmarks/peeking.py`](https://github.com/shivsbhaduri/ville/blob/main/benchmarks/peeking.py).*

## Installation

```bash
pip install ville
```

To work on the code, install from a clone instead:

```bash
git clone https://github.com/shivsbhaduri/ville.git
cd ville
pip install -e ".[test,examples]"
```

It needs Python 3.10 or later, NumPy and SciPy. Matplotlib is only needed to redraw the figures.

## What is in the library

| Class | Target | Assumptions | Method |
|---|---|---|---|
| `GaussianMixtureCS` | mean | sub-Gaussian with known scale | Robbins (1970); Howard et al. (2021) |
| `AsymptoticCS` | mean | finite variance, large sample | Waudby-Smith et al. (2024) |
| `EmpiricalBernsteinCS` | mean | known bounds | Waudby-Smith and Ramdas (2024), Theorem 2 |
| `BettingCS` | mean | known bounds | Waudby-Smith and Ramdas (2024), Theorem 3 |
| `BernoulliCS` | proportion | binary data | beta-binomial mixture; Howard et al. (2021) |
| `QuantileCS` | any quantile | none beyond i.i.d. | Howard and Ramdas (2022) |
| `VarianceCS` | variance | known bounds | paired differences with `BettingCS` |
| `AsymptoticABTest` | average treatment effect | randomised, finite variance | Waudby-Smith et al. (2024) |
| `BettingABTest` | average treatment effect | randomised, bounded outcome | Waudby-Smith and Ramdas (2024) |
| `MSPRT` | Gaussian mean test | Gaussian or sub-Gaussian | Johari et al. (2022) |
| `BettingTest` | bounded mean test | known bounds | Waudby-Smith and Ramdas (2024) |
| `BernoulliTest` | proportion test | binary data | beta-binomial Bayes factor |
| `evalues.p_to_e`, `e_to_p`, `combine` | e-value calculus | see below | Vovk and Wang (2021) |
| `evalues.ebh` | false discovery rate | arbitrary dependence | Wang and Ramdas (2022) |
| `simulate.peeking`, `simulate.stopping_times` | diagnostics | | |

Confidence sequences inherit from `ville.ConfidenceSequence` and sequential tests from `ville.SequentialTest`. Tests expose `e_values_`, anytime valid `p_values_` and `stopping_time(alpha)`.
