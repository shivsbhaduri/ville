import numpy as np
from scipy.stats import norm

from ville import best_rho, normal_mixture


def test_two_sided_boundary_matches_mixture_martingale():
    v, alpha, rho = np.array([1.0, 10.0, 1000.0]), 0.05, 7.0
    s = normal_mixture(v, alpha, rho)
    mixture = np.sqrt(rho / (v + rho)) * np.exp(s**2 / (2 * (v + rho)))
    assert np.allclose(mixture, 1 / alpha)


def test_one_sided_boundary_solves_its_equation():
    v, alpha, rho = np.array([0.5, 25.0, 4000.0]), 0.025, 3.0
    s = normal_mixture(v, alpha, rho, one_sided=True)
    w = v + rho
    mixture = 2 * np.sqrt(rho / w) * np.exp(s**2 / (2 * w)) * norm.cdf(s / np.sqrt(w))
    assert np.allclose(mixture, 1 / alpha)


def test_one_sided_is_tighter_than_two_sided():
    v = np.geomspace(1, 1e6, 20)
    assert np.all(normal_mixture(v, 0.05, 10, one_sided=True) < normal_mixture(v, 0.05, 10))


def test_best_rho_solves_stationarity_condition():
    # d/drho of the two sided boundary vanishes where r - log r = 1 + 2 log(1/alpha),
    # with r = (v + rho) / rho.
    v, alpha = 1000.0, 0.05
    rho = best_rho(v, alpha)
    r = (v + rho) / rho
    assert abs(r - np.log(r) - (1 + 2 * np.log(1 / alpha))) < 1e-4


def test_two_sided_crossing_rate_is_below_alpha():
    rng = np.random.default_rng(0)
    n, sims, alpha = 2000, 3000, 0.1
    rho = best_rho(200, alpha)
    s = np.cumsum(rng.standard_normal((sims, n)), axis=1)
    u = normal_mixture(np.arange(1, n + 1), alpha, rho)
    assert (np.abs(s) >= u).any(axis=1).mean() <= alpha
