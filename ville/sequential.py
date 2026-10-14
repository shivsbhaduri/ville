import numpy as np

from ._base import SequentialTest, check_bounds
from ._boundaries import best_rho
from .means import _PlugIn, _plugin_lambda

ALTERNATIVES = ("two-sided", "greater", "less")


class MSPRT(SequentialTest):
    """Mixture sequential probability ratio test for a Gaussian mean.

    Johari et al. (2022). With a N(theta0, tau^2) mixture over the
    alternative,
        Lambda_t = sqrt(sigma^2 / (sigma^2 + t tau^2))
                   * exp(t^2 tau^2 (xbar_t - theta0)^2 / (2 sigma^2 (sigma^2 + t tau^2))).
    If tau is not given it is chosen to make the test most sensitive around
    t_opt observations.
    """

    def __init__(self, theta0=0.0, sigma=1.0, tau=None, alpha=0.05, t_opt=500):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.theta0 = float(theta0)
        self.sigma = float(sigma)
        self.t_opt = t_opt
        super().__init__(alpha)
        if tau is None:
            rho = best_rho(self.sigma**2 * t_opt, self.alpha)
            tau = self.sigma**2 / np.sqrt(rho)
        self.tau = float(tau)

    def _reset(self):
        self._sum = 0.0

    def _process(self, x):
        t = self.n_ + np.arange(1, x.size + 1)
        s = self._sum + np.cumsum(x - self.theta0)
        self._sum = s[-1]
        s2, tau2 = self.sigma**2, self.tau**2
        return (0.5 * np.log(s2 / (s2 + t * tau2))
                + tau2 * s**2 / (2 * s2 * (s2 + t * tau2)))


class BettingTest(SequentialTest):
    """Betting e-process for the mean of bounded observations.

    Waudby-Smith and Ramdas (2024). Under H0: mean = m0 the wealth
        K_t = prod (1 + lambda_i (X_i - m0))
    is a nonnegative martingale for any predictable lambda_i, so it is an
    e-process. The two sided test splits capital between an upward and a
    downward bettor. One sided alternatives keep only one of them, which
    also makes the test valid for the composite nulls mean <= m0 or >= m0.
    """

    def __init__(self, m0, bounds=(0.0, 1.0), alternative="two-sided",
                 alpha=0.05, c=0.5, theta=0.5):
        lo, hi = check_bounds(bounds)
        if not lo < m0 < hi:
            raise ValueError("m0 must lie strictly inside bounds")
        if alternative not in ALTERNATIVES:
            raise ValueError(f"alternative must be one of {ALTERNATIVES}")
        self.m0 = float(m0)
        self.bounds = (lo, hi)
        self.alternative = alternative
        self.c = c
        self.theta = theta
        super().__init__(alpha)

    def _reset(self):
        self._plug = _PlugIn()
        self._log_up = 0.0
        self._log_down = 0.0

    def _process(self, x):
        lo, hi = self.bounds
        if np.any(x < lo) or np.any(x > hi):
            raise ValueError(f"observations must lie in {self.bounds}")
        z = (x - lo) / (hi - lo)
        m = (self.m0 - lo) / (hi - lo)

        i, _, var_prev = self._plug.step(z)
        lam = _plugin_lambda(i, var_prev, self.alpha)
        log_up = self._log_up + np.cumsum(np.log1p(np.minimum(lam, self.c / m) * (z - m)))
        log_down = self._log_down + np.cumsum(np.log1p(-np.minimum(lam, self.c / (1 - m)) * (z - m)))
        self._log_up, self._log_down = log_up[-1], log_down[-1]

        if self.alternative == "greater":
            return log_up
        if self.alternative == "less":
            return log_down
        return np.logaddexp(np.log(self.theta) + log_up, np.log1p(-self.theta) + log_down)
