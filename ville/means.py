import numpy as np

from ._base import ConfidenceSequence, check_bounds
from ._boundaries import best_rho, normal_mixture


class GaussianMixtureCS(ConfidenceSequence):
    """Confidence sequence for the mean of sigma sub-Gaussian observations.

    Uses Robbins' two sided normal mixture boundary (Robbins 1970;
    Howard et al. 2021). The mixing precision minimises the width at t_opt.

    Parameters
    ----------
    sigma : float
        Sub-Gaussian scale. For data in [a, b], sigma = (b - a) / 2 is valid.
    alpha : float
        The sequence covers the mean at every time with probability >= 1 - alpha.
    t_opt : int
        Sample size at which the interval should be narrowest.
    intersect : bool
        Report the running intersection of the intervals.
    """

    def __init__(self, sigma=1.0, alpha=0.05, t_opt=500, intersect=True):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)
        self.t_opt = t_opt
        super().__init__(alpha, intersect)
        self.rho_ = best_rho(self.sigma**2 * t_opt, self.alpha)

    def _reset(self):
        self._sum = 0.0

    def _process(self, x):
        t = self.n_ + np.arange(1, x.size + 1)
        s = self._sum + np.cumsum(x)
        self._sum = s[-1]
        mean = s / t
        radius = normal_mixture(self.sigma**2 * t, self.alpha, self.rho_) / t
        return mean, mean - radius, mean + radius


class AsymptoticCS(ConfidenceSequence):
    """Asymptotic confidence sequence for a mean (Waudby-Smith et al. 2024).

    Time uniform analogue of the CLT interval. Requires a finite variance;
    no bounds or scale parameter. The guarantee is asymptotic, so the
    interval is infinite until min_samples observations are available.
    """

    def __init__(self, alpha=0.05, t_opt=500, min_samples=100, intersect=False):
        self.t_opt = t_opt
        self.min_samples = max(int(min_samples), 2)
        super().__init__(alpha, intersect)
        self.rho_ = best_rho(t_opt, self.alpha)

    def _reset(self):
        self._shift = None
        self._s = 0.0
        self._q = 0.0

    def _process(self, x):
        if self._shift is None:
            self._shift = x[0]
        y = x - self._shift
        t = self.n_ + np.arange(1, x.size + 1)
        s = self._s + np.cumsum(y)
        q = self._q + np.cumsum(y * y)
        self._s, self._q = s[-1], q[-1]

        mean = s / t
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.maximum(q - t * mean**2, 0.0) / (t - 1)
            radius = np.sqrt(var) * normal_mixture(t, self.alpha, self.rho_) / t
        radius = np.where(t < self.min_samples, np.inf, radius)
        mean = mean + self._shift
        return mean, mean - radius, mean + radius


class _PlugIn:
    """Predictable mean and variance estimates used by the betting methods.

    mu_t = (1/2 + sum X_i) / (t + 1)
    var_t = (1/4 + sum (X_i - mu_i)^2) / (t + 1)
    on data rescaled to [0, 1]. step() returns the values at time t - 1,
    i.e. the predictable estimates used for X_t.
    """

    def __init__(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_dev = 0.0
        self.mu = 0.5
        self.var = 0.25

    def step(self, x):
        i = self.n + np.arange(1, x.size + 1)
        mu = (0.5 + self.sum_x + np.cumsum(x)) / (i + 1)
        dev = np.cumsum((x - mu) ** 2)
        var = (0.25 + self.sum_dev + dev) / (i + 1)
        mu_prev = np.concatenate(([self.mu], mu[:-1]))
        var_prev = np.concatenate(([self.var], var[:-1]))
        self.n += x.size
        self.sum_x += x.sum()
        self.sum_dev += dev[-1]
        self.mu, self.var = mu[-1], var[-1]
        return i, mu_prev, var_prev


def _plugin_lambda(i, var_prev, alpha):
    return np.sqrt(2 * np.log(2 / alpha) / (var_prev * i * np.log1p(i)))


class _Bounded(ConfidenceSequence):
    def __init__(self, bounds, alpha, intersect):
        self.bounds = check_bounds(bounds)
        super().__init__(alpha, intersect)

    def _scale(self, x):
        lo, hi = self.bounds
        if np.any(x < lo) or np.any(x > hi):
            raise ValueError(f"observations must lie in {self.bounds}")
        return (x - lo) / (hi - lo)

    def _unscale(self, *arrays):
        lo, hi = self.bounds
        return tuple(lo + a * (hi - lo) for a in arrays)


class EmpiricalBernsteinCS(_Bounded):
    """Predictable plug-in empirical Bernstein confidence sequence.

    Waudby-Smith and Ramdas (2024), Theorem 2. Closed form, valid for any
    distribution on known bounds, and adapts to the variance.
    """

    def __init__(self, bounds=(0.0, 1.0), alpha=0.05, c=0.5, intersect=True):
        if not 0 < c < 1:
            raise ValueError("c must lie in (0, 1)")
        self.c = c
        super().__init__(bounds, alpha, intersect)

    def _reset(self):
        self._plug = _PlugIn()
        self._lam = 0.0
        self._lam_x = 0.0
        self._penalty = 0.0
        self._sum = 0.0

    def _process(self, x):
        z = self._scale(x)
        i, mu_prev, var_prev = self._plug.step(z)
        lam = np.minimum(_plugin_lambda(i, var_prev, self.alpha), self.c)
        v = 4 * (z - mu_prev) ** 2
        psi = (-np.log1p(-lam) - lam) / 4

        lam_sum = self._lam + np.cumsum(lam)
        lam_x = self._lam_x + np.cumsum(lam * z)
        penalty = self._penalty + np.cumsum(v * psi)
        s = self._sum + np.cumsum(z)
        self._lam, self._lam_x = lam_sum[-1], lam_x[-1]
        self._penalty, self._sum = penalty[-1], s[-1]

        centre = lam_x / lam_sum
        radius = (np.log(2 / self.alpha) + penalty) / lam_sum
        lower = np.maximum(centre - radius, 0.0)
        upper = np.minimum(centre + radius, 1.0)
        return self._unscale(s / i, lower, upper)


class BettingCS(_Bounded):
    """Hedged capital confidence sequence for a bounded mean.

    Waudby-Smith and Ramdas (2024), Theorem 3. A candidate mean m is
    excluded once the hedged capital process K_t(m) reaches 1/alpha. Capital
    is tracked on a grid of candidate means; with intersect=True only grid
    points inside the current interval are updated.

    Parameters
    ----------
    bounds : (float, float)
        Known range of the observations.
    alpha : float
    c : float
        Bets are truncated at c / m and c / (1 - m), keeping capital positive.
    theta : float
        Share of capital placed on the upward bet.
    grid : int
        Number of candidate means. The interval is widened to the adjacent
        grid points, so the discretisation is conservative.
    """

    def __init__(self, bounds=(0.0, 1.0), alpha=0.05, c=0.5, theta=0.5,
                 grid=10_000, intersect=True):
        if not 0 < c < 1:
            raise ValueError("c must lie in (0, 1)")
        if not 0 < theta < 1:
            raise ValueError("theta must lie in (0, 1)")
        self.c = c
        self.theta = theta
        self.grid = int(grid)
        super().__init__(bounds, alpha, intersect)

    def _reset(self):
        self._plug = _PlugIn()
        self._m = np.linspace(0.0, 1.0, self.grid)
        with np.errstate(divide="ignore"):
            self._cap_up = self.c / self._m
            self._cap_down = self.c / (1 - self._m)
        self._log_up = np.zeros(self.grid)
        self._log_down = np.zeros(self.grid)
        self._alive = (0, self.grid - 1)
        self._sum = 0.0

    def _process(self, x):
        z = self._scale(x)
        i, _, var_prev = self._plug.step(z)
        lam = _plugin_lambda(i, var_prev, self.alpha)
        s = self._sum + np.cumsum(z)
        self._sum = s[-1]

        threshold = -np.log(self.alpha)
        log_theta, log_rest = np.log(self.theta), np.log1p(-self.theta)
        lower = np.empty(z.size)
        upper = np.empty(z.size)

        start = 0
        while start < z.size:
            # Under the running intersection excluded grid points cannot
            # re-enter, so only the current window is updated.
            a, b = self._alive
            w = slice(a, b + 1)
            sl = slice(start, start + max(1, 2_000_000 // (b - a + 1)))
            diff = z[sl, None] - self._m[None, w]
            up = np.minimum(lam[sl, None], self._cap_up[w])
            down = np.minimum(lam[sl, None], self._cap_down[w])
            log_up = self._log_up[w] + np.cumsum(np.log1p(up * diff), axis=0)
            log_down = self._log_down[w] + np.cumsum(np.log1p(-down * diff), axis=0)
            self._log_up[w], self._log_down[w] = log_up[-1], log_down[-1]

            log_k = np.maximum(log_theta + log_up, log_rest + log_down)
            inside = log_k < threshold
            any_in = inside.any(axis=1)
            first = np.argmax(inside, axis=1)
            last = inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1)
            # No grid point inside: the set lies between the neighbours of
            # the grid point with the smallest capital.
            j = np.argmin(log_k, axis=1)
            lo_idx = np.maximum(a + np.where(any_in, first, j) - 1, 0)
            hi_idx = np.minimum(a + np.where(any_in, last, j) + 1, self.grid - 1)
            lower[sl] = self._m[lo_idx]
            upper[sl] = self._m[hi_idx]

            if self.intersect:
                a = max(a, int(lo_idx.max()))
                self._alive = (a, max(a, min(b, int(hi_idx.min()))))
            start = sl.stop

        return self._unscale(s / i, lower, upper)
