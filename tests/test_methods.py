import numpy as np
import pytest
from scipy.special import betaln

import ville
from ville.evalues import bh, combine, e_to_p, ebh, p_to_e


@pytest.mark.parametrize("make", [
    lambda: ville.GaussianMixtureCS(),
    lambda: ville.AsymptoticCS(),
    lambda: ville.EmpiricalBernsteinCS(),
    lambda: ville.BettingCS(grid=300),
    lambda: ville.QuantileCS(q=0.3),
    lambda: ville.VarianceCS(grid=300),
])
def test_streaming_matches_batch(make):
    x = np.random.default_rng(1).random(501)
    batch = make().fit(x)
    stream = make()
    for chunk in np.array_split(x, 9):
        stream.update(chunk)
    assert np.allclose(batch.lower_, stream.lower_)
    assert np.allclose(batch.upper_, stream.upper_)


def test_intervals_shrink_and_contain_the_estimate():
    x = np.random.default_rng(2).beta(3, 3, 4000)
    cs = ville.EmpiricalBernsteinCS().fit(x)
    width = cs.upper_ - cs.lower_
    assert np.all(np.diff(width) <= 1e-12)
    assert width[-1] < 0.05
    assert cs.lower_[-1] <= x.mean() <= cs.upper_[-1]


def test_bounds_are_enforced():
    with pytest.raises(ValueError):
        ville.BettingCS(bounds=(0, 1)).fit([0.2, 1.5])


def test_bernoulli_limits_sit_on_the_rejection_threshold():
    x = (np.random.default_rng(3).random(800) < 0.4).astype(float)
    cs = ville.BernoulliCS(alpha=0.05, intersect=False).fit(x)
    lo, hi = cs.interval()
    for p0 in (lo, hi):
        e = ville.BernoulliTest(p0=p0).fit(x).e_values_[-1]
        assert np.isclose(e, 20.0, rtol=1e-6)


def test_bernoulli_e_value_is_beta_binomial_bayes_factor():
    x = np.array([1, 0, 0, 1, 1, 1, 0, 1], dtype=float)
    s, t = x.sum(), x.size
    expected = np.exp(betaln(1 + s, 1 + t - s) - betaln(1, 1)) / (0.3**s * 0.7 ** (t - s))
    assert np.isclose(ville.BernoulliTest(0.3).fit(x).e_values_[-1], expected)


def test_msprt_matches_johari_formula():
    x = np.random.default_rng(4).normal(0.2, 2.0, 50)
    test = ville.MSPRT(theta0=0.1, sigma=2.0, tau=0.5).fit(x)
    t = np.arange(1, 51)
    xbar = np.cumsum(x) / t
    s2, tau2 = 4.0, 0.25
    expected = np.sqrt(s2 / (s2 + t * tau2)) * np.exp(
        t**2 * tau2 * (xbar - 0.1) ** 2 / (2 * s2 * (s2 + t * tau2)))
    assert np.allclose(test.e_values_, expected)


@pytest.mark.parametrize("make, draw", [
    (lambda: ville.MSPRT(theta0=0.0), lambda r: r.standard_normal(200)),
    (lambda: ville.BettingTest(0.5), lambda r: r.random(200)),
])
def test_e_processes_have_unit_mean_under_the_null(make, draw):
    finals = [make().fit(draw(np.random.default_rng(s))).e_values_[-1] for s in range(4000)]
    assert abs(np.mean(finals) - 1) < 0.15


def test_bernoulli_e_value_has_mean_exactly_one():
    # Exact summation over the binomial distribution. Monte Carlo is
    # unreliable: the mean is dominated by sequences of probability ~2^-200.
    from scipy.stats import binom
    t, p0 = 200, 0.3
    s = np.arange(t + 1)
    log_e = betaln(1 + s, 1 + t - s) - betaln(1, 1) - s * np.log(p0) - (t - s) * np.log(1 - p0)
    assert np.isclose(np.sum(binom.pmf(s, t, p0) * np.exp(log_e)), 1.0)


def test_one_sided_betting_test_ignores_the_wrong_direction():
    x = np.random.default_rng(5).beta(2, 8, 2000)
    assert ville.BettingTest(0.5, alternative="greater").fit(x).stopping_time() is None
    assert ville.BettingTest(0.5, alternative="less").fit(x).stopping_time() is not None


def test_p_values_are_monotone():
    x = np.random.default_rng(6).normal(0.05, 1, 3000)
    p = ville.MSPRT().fit(x).p_values_
    assert np.all(np.diff(p) <= 0)


def test_mixture_calibrator_is_the_integral_of_kappa_calibrators():
    p = np.array([1e-6, 0.003, 0.04, 0.5, 0.97])
    kappas = np.linspace(0, 1, 200_001)[1:]
    numeric = np.trapezoid(kappas[:, None] * p ** (kappas[:, None] - 1), kappas, axis=0)
    assert np.allclose(p_to_e(p), numeric, rtol=1e-3)
    assert p_to_e(1.0) == 0.5


def test_calibrated_e_values_have_mean_at_most_one():
    p = np.random.default_rng(7).random(1_000_000)
    assert p_to_e(p).mean() < 1.01
    assert p_to_e(p, kappa=0.3).mean() < 1.01


def test_e_to_p_and_combine():
    assert np.allclose(e_to_p([0.5, 4, 40]), [1, 0.25, 0.025])
    assert combine([2, 4, 6]) == 4
    assert combine([2, 4, 6], method="product") == 48


def test_ebh_matches_hand_calculation():
    # m = 5, alpha = 0.1, so the k-th largest must satisfy e_(k) >= 50 / k.
    e = np.array([60.0, 30.0, 14.0, 1.0, 0.2])
    assert ebh(e, alpha=0.1).tolist() == [True, True, False, False, False]
    e[2] = 17.0
    assert ebh(e, alpha=0.1).tolist() == [True, True, True, False, False]


def test_ebh_controls_fdr_in_simulation():
    rng = np.random.default_rng(8)
    m, nonnull, alpha = 200, 40, 0.1
    fdp = []
    for _ in range(300):
        z = rng.standard_normal(m)
        z[:nonnull] += 3.0
        # Likelihood ratio e-values against N(0, 1) with alternative N(3, 1).
        e = np.exp(3 * z - 4.5)
        rej = ebh(e, alpha)
        fdp.append(rej[nonnull:].sum() / max(rej.sum(), 1))
    assert np.mean(fdp) <= alpha


def test_bh_reference():
    assert bh([0.001, 0.02, 0.03, 0.5], alpha=0.05).tolist() == [True, True, True, False]


def test_ab_effect_is_recovered():
    rng = np.random.default_rng(9)
    a = (rng.random(20_000) < 0.5).astype(float)
    y = 50 + 3 * a + rng.normal(0, 10, a.size)
    test = ville.AsymptoticABTest().fit(y, a)
    lo, hi = test.interval()
    assert lo < 3 < hi
    assert test.stopping_time() is not None


def test_peeking_simulation_reproduces_the_problem():
    out = ville.simulate.peeking(n_max=2000, n_looks=20, n_sims=400, seed=1)
    assert out["fixed"][-1] > 0.15
    assert out["anytime"][-1] <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)
