"""Monte Carlo helpers for comparing continuous monitoring with fixed n tests."""

import numpy as np
from scipy.stats import norm

from ._base import check_alpha
from ._boundaries import best_rho, normal_mixture


def _ab_paths(y, a, alpha, rho, min_samples=100, propensity=0.5):
    """Running effect estimates, z test p-values and anytime valid radii,
    for a batch of simulated A/B tests stored row-wise."""
    n = y.shape[1]
    t = np.arange(1, n + 1)

    n1 = np.cumsum(a, axis=1)
    n0 = t - n1
    s1 = np.cumsum(y * a, axis=1)
    s0 = np.cumsum(y * (1 - a), axis=1)
    q1 = np.cumsum(y * y * a, axis=1)
    q0 = np.cumsum(y * y * (1 - a), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        m1, m0 = s1 / n1, s0 / n0
        v1 = (q1 - n1 * m1**2) / (n1 - 1)
        v0 = (q0 - n0 * m0**2) / (n0 - 1)
        z = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
    p_fixed = 2 * norm.sf(np.abs(z))
    p_fixed = np.where(np.isfinite(p_fixed), p_fixed, 1.0)

    # Centred pseudo outcomes, as in ville.ab.AsymptoticABTest.
    past = np.concatenate((np.zeros((y.shape[0], 1)), np.cumsum(y, axis=1)[:, :-1]), axis=1)
    centre = np.where(t > 1, past / np.maximum(t - 1, 1), 0.0)
    phi = (y - centre) * (a / propensity - (1 - a) / (1 - propensity))
    s = np.cumsum(phi, axis=1)
    q = np.cumsum(phi * phi, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / t
        sd = np.sqrt(np.maximum(q - t * mean**2, 0) / (t - 1))
        radius = sd * normal_mixture(t, alpha, rho) / t
    radius = np.where(t < max(min_samples, 2), np.inf, radius)
    return mean, radius, p_fixed


def peeking(n_max=10_000, n_looks=30, n_sims=2000, effect=0.0, alpha=0.05,
            t_opt=None, seed=0, batch=200):
    """Simulate A/B tests on N(0, 1) outcomes with a shift of `effect` under
    treatment, looked at n_looks times at equal intervals.

    Returns the look times and, for each look, the share of simulations that
    have declared an effect at or before it, both for a z test applied at
    every look and for the asymptotic confidence sequence.
    """
    alpha = check_alpha(alpha)
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_max / n_looks, n_max, n_looks).astype(int)
    rho = best_rho(t_opt or n_max // 4, alpha)

    fixed_hits = np.zeros(n_looks)
    anytime_hits = np.zeros(n_looks)
    done = 0
    while done < n_sims:
        b = min(batch, n_sims - done)
        a = (rng.random((b, n_max)) < 0.5).astype(float)
        y = rng.standard_normal((b, n_max)) + effect * a
        mean, radius, p_fixed = _ab_paths(y, a, alpha, rho)
        idx = looks - 1
        fixed = np.logical_or.accumulate(p_fixed[:, idx] < alpha, axis=1)
        anytime = np.logical_or.accumulate(np.abs(mean[:, idx]) > radius[:, idx], axis=1)
        fixed_hits += fixed.sum(axis=0)
        anytime_hits += anytime.sum(axis=0)
        done += b

    return {"looks": looks, "fixed": fixed_hits / n_sims, "anytime": anytime_hits / n_sims}


def stopping_times(n_max=10_000, n_sims=2000, effect=0.05, alpha=0.05,
                   t_opt=None, seed=0, batch=200):
    """Monitor after every observation and record when the confidence
    sequence first excludes zero. np.inf for runs that did not stop."""
    alpha = check_alpha(alpha)
    rng = np.random.default_rng(seed)
    rho = best_rho(t_opt or n_max // 4, alpha)
    out = np.full(n_sims, np.inf)
    done = 0
    while done < n_sims:
        b = min(batch, n_sims - done)
        a = (rng.random((b, n_max)) < 0.5).astype(float)
        y = rng.standard_normal((b, n_max)) + effect * a
        mean, radius, _ = _ab_paths(y, a, alpha, rho)
        hit = np.abs(mean) > radius
        first = np.argmax(hit, axis=1)
        out[done:done + b] = np.where(hit.any(axis=1), first + 1, np.inf)
        done += b
    return out
