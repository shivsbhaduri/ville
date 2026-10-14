import numpy as np
from scipy.special import betaln, xlogy

from ._base import ConfidenceSequence, SequentialTest
from ._boundaries import bisect


def _check_binary(x):
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("observations must be 0 or 1")


def _log_marginal(s, t, a, b):
    return betaln(a + s, b + t - s) - betaln(a, b)


def _log_likelihood(s, t, p):
    return xlogy(s, p) + xlogy(t - s, 1 - p)


class BernoulliCS(ConfidenceSequence):
    """Beta-binomial mixture confidence sequence for a proportion.

    The e-process for H0: p = p0 is the Bayes factor of a Beta(a, b) mixture
    against the point null,
        E_t(p0) = B(a + S_t, b + t - S_t) / B(a, b) / (p0^S_t (1 - p0)^(t - S_t)),
    and the sequence collects every p0 it has not yet rejected (Howard et al.
    2021). The prior affects width only.
    """

    def __init__(self, alpha=0.05, prior=(1.0, 1.0), intersect=True):
        self.prior = tuple(map(float, prior))
        if min(self.prior) <= 0:
            raise ValueError("prior parameters must be positive")
        super().__init__(alpha, intersect)

    def _reset(self):
        self._s = 0.0

    def _process(self, x):
        _check_binary(x)
        a, b = self.prior
        t = self.n_ + np.arange(1, x.size + 1, dtype=float)
        s = self._s + np.cumsum(x)
        self._s = s[-1]
        p_hat = s / t

        # p0 is rejected when -loglik(p0) > log(1/alpha) - log marginal.
        level = -np.log(self.alpha) - _log_marginal(s, t, a, b)

        def lower_gap(p):
            return level + _log_likelihood(s, t, p)

        def upper_gap(p):
            return -_log_likelihood(s, t, p) - level

        lower = np.where(s > 0, bisect(lower_gap, np.zeros_like(t), p_hat), 0.0)
        upper = np.where(s < t, bisect(upper_gap, p_hat, np.ones_like(t)), 1.0)
        return p_hat, lower, upper


class BernoulliTest(SequentialTest):
    """Beta-binomial mixture e-process for H0: p = p0."""

    def __init__(self, p0, alpha=0.05, prior=(1.0, 1.0)):
        if not 0 < p0 < 1:
            raise ValueError("p0 must lie in (0, 1)")
        self.p0 = float(p0)
        self.prior = tuple(map(float, prior))
        super().__init__(alpha)

    def _reset(self):
        self._s = 0.0

    def _process(self, x):
        _check_binary(x)
        a, b = self.prior
        t = self.n_ + np.arange(1, x.size + 1, dtype=float)
        s = self._s + np.cumsum(x)
        self._s = s[-1]
        return _log_marginal(s, t, a, b) - _log_likelihood(s, t, self.p0)
