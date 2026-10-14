"""Working with e-values directly: calibration, combination and e-BH."""

import numpy as np

from ._base import check_alpha


def p_to_e(p, kappa=None):
    """Turn p-values into e-values with a calibrator (Vovk and Wang 2021).

    With kappa in (0, 1) this is kappa * p^(kappa - 1). Without it, the
    calibrator is averaged over kappa in (0, 1), which gives
        (1 - p + p log p) / (p (log p)^2).
    """
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    with np.errstate(divide="ignore", invalid="ignore"):
        if kappa is not None:
            if not 0 < kappa < 1:
                raise ValueError("kappa must lie in (0, 1)")
            return kappa * p ** (kappa - 1)
        log_p = np.log(p)
        e = (1 - p + p * log_p) / (p * log_p**2)
    e = np.where(p == 0, np.inf, e)
    return np.where(p == 1, 0.5, e)


def e_to_p(e):
    """The p-value implied by Markov's inequality, min(1, 1/e)."""
    e = np.asarray(e, dtype=float)
    if np.any(e < 0):
        raise ValueError("e-values must be nonnegative")
    with np.errstate(divide="ignore"):
        return np.minimum(1.0, 1.0 / e)


def combine(e, method="mean", weights=None, axis=-1):
    """Merge e-values for the same null.

    "mean" is valid under any dependence. "product" needs the e-values to be
    independent, or sequential so that each is valid given the ones before.
    """
    e = np.asarray(e, dtype=float)
    if method == "mean":
        return np.average(e, axis=axis, weights=weights)
    if method == "product":
        if weights is not None:
            raise ValueError("weights are only used by the mean")
        return np.prod(e, axis=axis)
    raise ValueError('method must be "mean" or "product"')


def ebh(e, alpha=0.05):
    """The e-BH procedure of Wang and Ramdas (2022).

    Rejects the k hypotheses with the largest e-values, where k is the
    largest index with k * e_(k) / m >= 1 / alpha. Controls the false
    discovery rate at alpha under arbitrary dependence between the e-values.
    Returns a boolean mask of rejections.
    """
    alpha = check_alpha(alpha)
    e = np.asarray(e, dtype=float)
    m = e.size
    order = np.argsort(-e)
    k = np.arange(1, m + 1)
    passing = np.flatnonzero(k * e[order] / m >= 1 / alpha)
    reject = np.zeros(m, dtype=bool)
    if passing.size:
        reject[order[: passing[-1] + 1]] = True
    return reject


def bh(p, alpha=0.05):
    """Benjamini and Hochberg (1995) for comparison with e-BH."""
    alpha = check_alpha(alpha)
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    k = np.arange(1, m + 1)
    passing = np.flatnonzero(p[order] <= k * alpha / m)
    reject = np.zeros(m, dtype=bool)
    if passing.size:
        reject[order[: passing[-1] + 1]] = True
    return reject
