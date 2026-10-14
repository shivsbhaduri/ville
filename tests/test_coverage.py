"""Time uniform coverage, checked by simulation.

A sequence fails a run if it excludes the truth at any time. The failure
rate must stay below alpha, allowing three standard errors of Monte Carlo
noise.
"""

import numpy as np
import pytest

import ville

ALPHA = 0.1
SIMS = 300
N = 1000
TOLERANCE = 3 * np.sqrt(ALPHA * (1 - ALPHA) / SIMS)


def failure_rate(make, draw, truth, sims=SIMS):
    misses = 0
    for seed in range(sims):
        x = draw(np.random.default_rng(seed))
        misses += make().fit(x).excludes(truth) is not None
    return misses / sims


@pytest.mark.parametrize("make", [
    lambda: ville.GaussianMixtureCS(sigma=0.5, alpha=ALPHA, t_opt=100),
    lambda: ville.EmpiricalBernsteinCS(alpha=ALPHA),
    lambda: ville.BettingCS(alpha=ALPHA, grid=400),
], ids=["gaussian-mixture", "empirical-bernstein", "betting"])
def test_bounded_mean(make):
    rate = failure_rate(make, lambda r: r.beta(2, 5, N), 2 / 7)
    assert rate <= ALPHA + TOLERANCE


def test_asymptotic_mean_on_skewed_data():
    rate = failure_rate(lambda: ville.AsymptoticCS(alpha=ALPHA, t_opt=200),
                        lambda r: r.exponential(2.0, 3000), 2.0)
    assert rate <= ALPHA + TOLERANCE


def test_bernoulli():
    rate = failure_rate(lambda: ville.BernoulliCS(alpha=ALPHA),
                        lambda r: (r.random(N) < 0.07).astype(float), 0.07)
    assert rate <= ALPHA + TOLERANCE


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_quantile(q):
    truth = -np.log(1 - q)
    rate = failure_rate(lambda: ville.QuantileCS(q=q, alpha=ALPHA),
                        lambda r: r.exponential(1.0, N), truth)
    assert rate <= ALPHA + TOLERANCE


def test_quantile_with_atoms():
    # Half the mass sits on 0, so the median is 0 and ties are everywhere.
    rate = failure_rate(lambda: ville.QuantileCS(q=0.5, alpha=ALPHA),
                        lambda r: np.where(r.random(N) < 0.5, 0.0, r.exponential(1.0, N)), 0.0)
    assert rate <= ALPHA + TOLERANCE


def test_variance():
    rate = failure_rate(lambda: ville.VarianceCS(alpha=ALPHA, grid=400),
                        lambda r: r.random(N), 1 / 12, sims=150)
    assert rate <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / 150)


@pytest.mark.parametrize("make", [
    lambda: ville.AsymptoticABTest(alpha=ALPHA, t_opt=200),
    lambda: ville.BettingABTest(alpha=ALPHA, grid=800),
], ids=["asymptotic", "betting"])
def test_ab_under_the_null(make):
    misses = 0
    for seed in range(SIMS):
        r = np.random.default_rng(seed)
        a = (r.random(N) < 0.5).astype(float)
        y = (r.random(N) < 0.2).astype(float)
        misses += make().fit(y, a).stopping_time() is not None
    assert misses / SIMS <= ALPHA + TOLERANCE
