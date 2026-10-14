import numpy as np

from ._base import ConfidenceSequence, check_bounds
from .means import BettingCS


class VarianceCS(ConfidenceSequence):
    """Confidence sequence for the variance of bounded i.i.d. data.

    Consecutive pairs give Y_k = (X_2k - X_2k-1)^2 / 2, which has mean
    sigma^2 and lies in [0, (b - a)^2 / 2]. A betting confidence sequence on
    the Y_k is therefore one for sigma^2. Exactly valid, but uses half the
    information of a full U-statistic. The upper bound is capped at
    (b - a)^2 / 4, the maximum variance on [a, b].
    """

    def __init__(self, bounds=(0.0, 1.0), alpha=0.05, grid=10_000, intersect=True):
        self.bounds = check_bounds(bounds)
        self.grid = grid
        super().__init__(alpha, intersect)

    def _reset(self):
        lo, hi = self.bounds
        self._cap = (hi - lo) ** 2 / 4
        self._pairs = BettingCS(bounds=(0.0, (hi - lo) ** 2 / 2), alpha=self.alpha,
                                grid=self.grid)
        self._carry = None
        self._s = 0.0
        self._q = 0.0
        self._shift = None

    def _process(self, x):
        lo, hi = self.bounds
        if np.any(x < lo) or np.any(x > hi):
            raise ValueError(f"observations must lie in {self.bounds}")

        stream = x if self._carry is None else np.concatenate(([self._carry], x))
        n_pairs = stream.size // 2
        if n_pairs:
            first, second = stream[0:2 * n_pairs:2], stream[1:2 * n_pairs:2]
            self._pairs.update((second - first) ** 2 / 2)
        self._carry = stream[-1] if stream.size % 2 else None

        t = self.n_ + np.arange(1, x.size + 1)
        k = t // 2
        pair_lo = np.concatenate(([0.0], self._pairs.lower_))
        pair_hi = np.concatenate(([self._cap], self._pairs.upper_))
        lower = pair_lo[k]
        upper = np.minimum(pair_hi[k], self._cap)

        if self._shift is None:
            self._shift = x[0]
        y = x - self._shift
        s = self._s + np.cumsum(y)
        q = self._q + np.cumsum(y * y)
        self._s, self._q = s[-1], q[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            est = np.where(t > 1, (q - s**2 / t) / (t - 1), np.nan)
        return est, lower, upper
