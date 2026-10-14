# Benchmarks

Each figure is produced by a script in [`benchmarks/`](https://github.com/shivsbhaduri/ville/tree/main/benchmarks/) with a fixed seed.

## False positives under peeking

Figure 1. Two sample z test at $\alpha = 0.05$, analysed once per day for 30 days, on experiments with no true effect.

| Day | Users | z test, ever significant | `AsymptoticABTest`, ever significant |
|---:|---:|---:|---:|
| 1 | 333 | 4.9% | 0.1% |
| 2 | 666 | 8.3% | 0.1% |
| 5 | 1,666 | 14.4% | 0.5% |
| 10 | 3,333 | 20.0% | 0.8% |
| 20 | 6,666 | 25.9% | 1.3% |
| 30 | 10,000 | 29.4% | 1.6% |

The confidence sequence controls the error rate at 5% under monitoring after every observation; with daily looks it is well below that.

## Width

![Width of 95% confidence sequences for a bounded mean](figures/widths.png)

Average width over 40 streams of Beta(2, 5) data (support $[0, 1]$, standard deviation 0.16).

| n | GaussianMixtureCS | AsymptoticCS | EmpiricalBernsteinCS | BettingCS | CLT interval (fixed n) |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.382 | 0.120 | 0.166 | 0.099 | 0.061 |
| 1,000 | 0.095 | 0.027 | 0.034 | 0.031 | 0.020 |
| 10,000 | 0.032 | 0.009 | 0.012 | 0.012 | 0.006 |
| 20,000 | 0.023 | 0.006 | 0.009 | 0.009 | 0.004 |

Without variance information (`GaussianMixtureCS` with the Hoeffding scale) the sequence is 2.3 to 3.9 times the width of the variance adaptive ones. The betting and empirical Bernstein sequences are nonasymptotic; betting is narrower at small $n$. `AsymptoticCS` is the narrowest time uniform method at large $n$, with an asymptotic guarantee only. The CLT interval is not time uniform and is shown for reference.

## Time to a decision

![When ville.AsymptoticABTest stops on real effects](figures/stopping.png)

A fixed horizon z test sized for 80% power at an effect of 0.05 needs 12,559 users. The confidence sequence is monitored after every user, with `t_opt` set to the same horizon.

| True effect | Stopped by 12,559 users | Median stopping time | Power of the fixed horizon test |
|---:|---:|---:|---:|
| 0.04 | 34.3% | 17,558 | 61.1% |
| 0.05 | 55.1% | 11,351 | 80.0% |
| 0.08 | 95.6% | 5,013 | 99.4% |

At its design sample size the fixed horizon test has higher power than a sequence that must be valid at all sample sizes (80.0% against 55.1% at effect 0.05). The sequence stops earlier on larger effects (median 5,013 users against 12,559 at effect 0.08) and allows the experiment to continue past the horizon when the effect is smaller than planned. The type I error rate is controlled in all cases.

## Reproducing the results

```bash
pip install -e ".[test,examples]"
pytest                                # 41 tests, about 15 seconds
python examples/conversion_rate.py    # and the other examples
cd benchmarks && python peeking.py    # redraws docs/figures/peeking.png
```

The test suite checks, for each confidence sequence, time uniform coverage by simulation (a run fails if the true value is excluded at any time, and the failure rate must not exceed $\alpha$ beyond Monte Carlo error). It also checks closed forms against direct computation, the unit mean of each e-process under the null, agreement between streaming and batch results, and e-BH against a hand calculation and in simulation.
