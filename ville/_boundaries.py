import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr


def bisect(f, lo, hi, iters=100):
    """Elementwise bisection for an increasing f with f(lo) < 0 < f(hi)."""
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = f(mid) > 0
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def normal_mixture(v, alpha, rho, one_sided=False):
    """Robbins' normal mixture boundary u(v).

    For a process S_t with exp(lambda S_t - lambda^2 V_t / 2) a supermartingale
    for every lambda, P(exists t: S_t >= u(V_t)) <= alpha in the one sided case
    and P(exists t: |S_t| >= u(V_t)) <= alpha in the two sided case.
    rho is the precision of the mixing distribution over lambda.
    """
    v = np.asarray(v, dtype=float)
    w = v + rho
    if not one_sided:
        return np.sqrt(w * np.log(w / (rho * alpha**2)))

    # Half normal mixture over lambda >= 0 (Howard et al. 2021). No closed
    # form for the boundary, so solve M(s) = 1/alpha for s.
    target = -np.log(alpha)

    def excess(s):
        return (np.log(2.0) + 0.5 * np.log(rho / w) + s**2 / (2 * w)
                + log_ndtr(s / np.sqrt(w)) - target)

    hi = np.sqrt(2 * w * (target + 0.5 * np.log(w / rho)))
    return bisect(excess, np.zeros_like(w), hi)


def best_rho(v_opt, alpha, one_sided=False):
    """Mixture precision that minimises the boundary at intrinsic time v_opt."""
    centre = np.log(v_opt)

    def width(log_rho):
        return float(normal_mixture(v_opt, alpha, np.exp(log_rho), one_sided))

    res = minimize_scalar(width, bounds=(centre - 15, centre + 15), method="bounded")
    return float(np.exp(res.x))
