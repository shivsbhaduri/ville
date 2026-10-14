import bisect as _bisect

import numpy as np

from ._base import ConfidenceSequence
from ._boundaries import best_rho, normal_mixture


class QuantileCS(ConfidenceSequence):
    """Confidence sequence for the q-quantile of any distribution.

    Follows Howard and Ramdas (2022). The count of observations at or below
    the true quantile is a sum of Bernoulli(q) variables, so a time uniform
    boundary u_t for that count pins the quantile between two order
    statistics:
        [ X_(ceil(t q - u_t)),  X_(floor(t q + u_t) + 1) ].
    Here u_t is a one sided normal mixture boundary at alpha / 2 on each
    side, with the Hoeffding variance 1/4. No assumptions on the
    distribution, including ties and atoms.
    """

    def __init__(self, q=0.5, alpha=0.05, t_opt=500, intersect=True):
        if not 0 < q < 1:
            raise ValueError("q must lie in (0, 1)")
        self.q = float(q)
        self.t_opt = t_opt
        super().__init__(alpha, intersect)
        self.rho_ = best_rho(0.25 * t_opt, self.alpha / 2, one_sided=True)

    def _reset(self):
        self._sorted = []

    def _process(self, x):
        t = self.n_ + np.arange(1, x.size + 1)
        u = normal_mixture(0.25 * t, self.alpha / 2, self.rho_, one_sided=True)
        lo_rank = np.ceil(t * self.q - u).astype(int)
        hi_rank = np.floor(t * self.q + u).astype(int) + 1
        mid_rank = np.maximum(np.ceil(t * self.q).astype(int), 1)

        data = self._sorted
        est = np.empty(x.size)
        lower = np.empty(x.size)
        upper = np.empty(x.size)
        for k, value in enumerate(x):
            _bisect.insort(data, value)
            n = len(data)
            est[k] = data[mid_rank[k] - 1]
            lower[k] = data[lo_rank[k] - 1] if lo_rank[k] >= 1 else -np.inf
            upper[k] = data[hi_rank[k] - 1] if hi_rank[k] <= n else np.inf
        return est, lower, upper
