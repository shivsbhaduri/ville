import numpy as np


def as_stream(x):
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("observations must be finite")
    return x


def check_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return float(alpha)


def check_bounds(bounds):
    lo, hi = map(float, bounds)
    if not lo < hi:
        raise ValueError("bounds must satisfy lo < hi")
    return lo, hi


def _join(chunks):
    if not chunks:
        return np.empty(0)
    if len(chunks) > 1:
        chunks[:] = [np.concatenate(chunks)]
    return chunks[0]


class ConfidenceSequence:
    """Base class. Subclasses implement _reset() and _process(x).

    _process receives a new block of observations and returns the running
    estimate, lower and upper bounds for each of them.
    """

    def __init__(self, alpha=0.05, intersect=True):
        self.alpha = check_alpha(alpha)
        self.intersect = intersect
        self._clear()

    def _clear(self):
        self.n_ = 0
        self._est, self._lo, self._hi = [], [], []
        self._reset()

    def _reset(self):
        pass

    def _process(self, x):
        raise NotImplementedError

    def fit(self, x):
        """Process a whole stream from scratch."""
        self._clear()
        return self.update(x)

    def update(self, x):
        """Add one or more observations to the stream."""
        x = as_stream(x)
        if x.size == 0:
            return self
        est, lo, hi = self._process(x)
        if self.intersect:
            if self._lo:
                lo = np.maximum(lo, self._lo[-1][-1])
                hi = np.minimum(hi, self._hi[-1][-1])
            lo = np.maximum.accumulate(lo)
            hi = np.minimum.accumulate(hi)
        self._est.append(est)
        self._lo.append(lo)
        self._hi.append(hi)
        self.n_ += x.size
        return self

    @property
    def estimate_(self):
        return _join(self._est)

    @property
    def lower_(self):
        return _join(self._lo)

    @property
    def upper_(self):
        return _join(self._hi)

    def interval(self):
        """The current interval (lower, upper)."""
        if self.n_ == 0:
            raise RuntimeError("no observations yet")
        return float(self._lo[-1][-1]), float(self._hi[-1][-1])

    def excludes(self, value):
        """First time (1 indexed) at which value falls outside the sequence, or None."""
        out = (self.lower_ > value) | (self.upper_ < value)
        hits = np.flatnonzero(out)
        return int(hits[0]) + 1 if hits.size else None

    def __repr__(self):
        if self.n_ == 0:
            return f"{type(self).__name__}(alpha={self.alpha})"
        lo, hi = self.interval()
        return f"{type(self).__name__}(n={self.n_}, interval=({lo:.6g}, {hi:.6g}))"


class SequentialTest:
    """Base class for e-processes. Subclasses implement _reset() and _process(x),
    returning log e-values for each new observation."""

    def __init__(self, alpha=0.05):
        self.alpha = check_alpha(alpha)
        self._clear()

    def _clear(self):
        self.n_ = 0
        self._log_e = []
        self._reset()

    def _reset(self):
        pass

    def _process(self, x):
        raise NotImplementedError

    def fit(self, x):
        self._clear()
        return self.update(x)

    def update(self, x):
        x = as_stream(x)
        if x.size == 0:
            return self
        self._log_e.append(self._process(x))
        self.n_ += x.size
        return self

    @property
    def log_e_values_(self):
        return _join(self._log_e)

    @property
    def e_values_(self):
        return np.exp(self.log_e_values_)

    @property
    def p_values_(self):
        """Anytime valid p-values, p_t = min(1, 1 / max_{s <= t} E_s)."""
        running = np.maximum.accumulate(self.log_e_values_)
        return np.minimum(1.0, np.exp(-running))

    def stopping_time(self, alpha=None):
        """First time (1 indexed) the e-process reaches 1/alpha, or None."""
        alpha = self.alpha if alpha is None else check_alpha(alpha)
        hits = np.flatnonzero(self.log_e_values_ >= -np.log(alpha))
        return int(hits[0]) + 1 if hits.size else None

    @property
    def rejected_(self):
        return self.stopping_time() is not None

    def __repr__(self):
        if self.n_ == 0:
            return f"{type(self).__name__}(alpha={self.alpha})"
        e = float(np.exp(self.log_e_values_[-1]))
        p = float(self.p_values_[-1])
        return f"{type(self).__name__}(n={self.n_}, e={e:.4g}, p={p:.4g})"
