import numpy as np

from ._base import as_stream, check_bounds
from .means import AsymptoticCS, BettingCS


class _ABTest:
    """Confidence sequence for the average treatment effect of a randomised test.

    Each unit is treated with known probability pi. The pseudo outcome
        phi_i = (Y_i - m_{i-1}) * (A_i / pi - (1 - A_i) / (1 - pi))
    has conditional mean E[Y(1)] - E[Y(0)] whatever the predictable centre
    m_{i-1} is, so a confidence sequence for the mean of phi is one for the
    effect. Centring on the running mean of past outcomes removes the
    variance an uncentred inverse propensity weight would carry when the
    outcomes have a large baseline.
    """

    def __init__(self, propensity, alpha):
        if not 0 < propensity < 1:
            raise ValueError("propensity must lie in (0, 1)")
        self.propensity = float(propensity)
        self.alpha = alpha
        self._clear()

    def _clear(self):
        self._cs = self._make_cs()
        self._sum_y = 0.0
        self._count = 0

    def fit(self, y, treated):
        self._clear()
        return self.update(y, treated)

    def update(self, y, treated):
        y = as_stream(y)
        a = as_stream(treated)
        if y.shape != a.shape:
            raise ValueError("y and treated must have the same length")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("treated must be 0 or 1")
        if y.size == 0:
            return self

        past = self._sum_y + np.concatenate(([0.0], np.cumsum(y)[:-1]))
        seen = self._count + np.arange(y.size)
        centre = np.where(seen > 0, past / np.maximum(seen, 1), self._initial_centre())
        self._sum_y += y.sum()
        self._count += y.size

        pi = self.propensity
        phi = (y - centre) * (a / pi - (1 - a) / (1 - pi))
        self._cs.update(phi)
        return self

    @property
    def n_(self):
        return self._cs.n_

    @property
    def estimate_(self):
        return self._cs.estimate_

    @property
    def lower_(self):
        return self._cs.lower_

    @property
    def upper_(self):
        return self._cs.upper_

    def interval(self):
        return self._cs.interval()

    def stopping_time(self):
        """First time (1 indexed) the sequence excludes zero, or None."""
        return self._cs.excludes(0.0)

    def __repr__(self):
        if self.n_ == 0:
            return f"{type(self).__name__}(alpha={self.alpha})"
        lo, hi = self.interval()
        return (f"{type(self).__name__}(n={self.n_}, effect={self.estimate_[-1]:.6g}, "
                f"interval=({lo:.6g}, {hi:.6g}))")


class AsymptoticABTest(_ABTest):
    """A/B test with an asymptotic confidence sequence (Waudby-Smith et al. 2024).

    Needs no bounds on the outcome, only a finite variance. Validity is
    asymptotic, like the usual z test.
    """

    def __init__(self, alpha=0.05, propensity=0.5, t_opt=500, min_samples=100):
        self.t_opt = t_opt
        self.min_samples = min_samples
        super().__init__(propensity, alpha)

    def _make_cs(self):
        return AsymptoticCS(alpha=self.alpha, t_opt=self.t_opt, min_samples=self.min_samples)

    def _initial_centre(self):
        return 0.0


class BettingABTest(_ABTest):
    """A/B test for bounded outcomes with a betting confidence sequence.

    Valid at every sample size, not only asymptotically. Suited to conversion
    rates and other outcomes with known bounds.
    """

    def __init__(self, bounds=(0.0, 1.0), alpha=0.05, propensity=0.5, grid=20_000):
        self.bounds = check_bounds(bounds)
        self.grid = grid
        super().__init__(propensity, alpha)

    def _make_cs(self):
        lo, hi = self.bounds
        reach = (hi - lo) / min(self.propensity, 1 - self.propensity)
        return BettingCS(bounds=(-reach, reach), alpha=self.alpha, grid=self.grid)

    def _initial_centre(self):
        return 0.5 * sum(self.bounds)

    def update(self, y, treated):
        lo, hi = self.bounds
        y_arr = as_stream(y)
        if np.any(y_arr < lo) or np.any(y_arr > hi):
            raise ValueError(f"outcomes must lie in {self.bounds}")
        return super().update(y_arr, treated)
