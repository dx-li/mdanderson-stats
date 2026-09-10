"""Stable Breslow partial likelihood and sparse monotone-likelihood check."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array

from ._validation import FloatArray


class _CoxLikelihood:
    def __init__(self, time: FloatArray, event: FloatArray, x: FloatArray):
        order = np.argsort(-time, kind="stable")
        self.x, self.event = x[order], event[order]
        self.time = time[order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.time)) + 1]
        self.ends = np.r_[self.starts[1:], time.size] - 1
        self.group = np.repeat(np.arange(self.starts.size), self.ends - self.starts + 1)
        self.deaths = np.add.reduceat(self.event, self.starts)
        self.event_x = self.x.T @ self.event

    def evaluate(self, beta: FloatArray) -> tuple[float, FloatArray, FloatArray, FloatArray]:
        x, d = self.x, self.deaths
        eta = x @ beta
        if not np.isfinite(eta).all():
            raise ArithmeticError("Cox linear predictor exceeds numerical range")
        if np.ptp(eta) > 500:
            return self._rescaled(eta)
        shift = eta.max()
        w = np.exp(eta - shift)
        risk = np.cumsum(w)[self.ends]
        mean = np.cumsum(w[:, None] * x, axis=0)[self.ends] / risk[:, None]
        log_risk = np.log(risk) + shift
        nll = float(np.sum(self.event * (np.log(risk[self.group]) - (eta - shift))))
        gradient = mean.T @ d - self.event_x
        exposure = np.cumsum((d / risk)[::-1])[::-1][self.group]
        info = x.T @ ((w * exposure)[:, None] * x) - mean.T @ (d[:, None] * mean)
        return nll, gradient, (info + info.T) / 2, log_risk

    def _rescaled(self, eta: FloatArray) -> tuple[float, FloatArray, FloatArray, FloatArray]:
        # Weighted central moments with a new log scale for each risk-set block.
        p = self.x.shape[1]
        mean, cov = np.zeros(p), np.zeros((p, p))
        gradient, info = np.zeros(p), np.zeros((p, p))
        log_total, nll = -np.inf, 0.0
        log_risk = np.empty(self.starts.size)
        for j, (start, end) in enumerate(zip(self.starts, self.ends, strict=True)):
            xx, ee = self.x[start : end + 1], eta[start : end + 1]
            w = np.exp(ee - ee.max())
            total = w.sum()
            block_log = float(ee.max() + np.log(total))
            block_mean = w @ xx / total
            centered = xx - block_mean
            block_cov = centered.T @ (w[:, None] * centered) / total
            combined = float(np.logaddexp(log_total, block_log))
            fraction = np.exp(block_log - combined)
            previous = np.exp(log_total - combined)
            delta = block_mean - mean
            cov = (
                previous * cov + fraction * block_cov + previous * fraction * np.outer(delta, delta)
            )
            mean = previous * mean + fraction * block_mean
            log_total = combined
            log_risk[j] = combined
            dd = self.deaths[j]
            nll += float(np.sum(self.event[start : end + 1] * (combined - ee)))
            gradient += dd * mean
            info += dd * cov
        return nll, gradient - self.event_x, (info + info.T) / 2, log_risk

    def check_separation(self, null_gradient: FloatArray) -> None:
        # One auxiliary risk maximum per time replaces all event/risk pairs.
        x = self.x
        n, p = x.shape
        m = self.starts.size
        deaths = np.flatnonzero(self.event)
        ne = deaths.size
        rows = np.r_[
            np.repeat(np.arange(n), p),
            np.arange(n),
            np.repeat(n + np.arange(ne), p),
            n + np.arange(ne),
            n + ne + np.arange(m - 1),
            n + ne + np.arange(m - 1),
        ]
        columns = np.r_[
            np.tile(np.arange(p), n),
            p + self.group,
            np.tile(np.arange(p), ne),
            p + self.group[deaths],
            p + np.arange(m - 1),
            p + np.arange(1, m),
        ]
        values = np.r_[
            x.ravel(), -np.ones(n), -x[deaths].ravel(), np.ones(ne), np.ones(m - 1), -np.ones(m - 1)
        ]
        constraints = coo_array((values, (rows, columns)), shape=(n + ne + m - 1, p + m)).tocsr()
        objective = np.r_[null_gradient / max(1, np.max(np.abs(null_gradient))), np.zeros(m)]
        result = linprog(
            objective,
            A_ub=constraints,
            b_ub=np.zeros(constraints.shape[0]),
            bounds=[(-1, 1)] * p + [(None, None)] * m,
            method="highs",
        )
        if not result.success:
            raise ArithmeticError(f"Cox separation check failed: {result.message}")
        if result.fun < -1e-7:
            raise ValueError("Cox monotone likelihood: no finite coefficient maximum")
