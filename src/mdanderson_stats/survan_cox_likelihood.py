"""Stable Breslow partial likelihood and sparse monotone-likelihood check."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array
from scipy.special import logsumexp

from ._validation import FloatArray


class _CoxLikelihood:
    def __init__(self, time: FloatArray, event: FloatArray, x: FloatArray, ties: str = "breslow"):
        if ties not in ("breslow", "efron"):
            raise ValueError("ties must be 'breslow' or 'efron'")
        self.ties = ties
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
        if self.ties == "efron":
            return self._evaluate_efron(eta)
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

    def _evaluate_efron(self, eta: FloatArray) -> tuple[float, FloatArray, FloatArray, FloatArray]:
        """Evaluate Efron's tied partial likelihood with online risk moments."""
        p = self.x.shape[1]
        zero_mean = np.zeros(p)
        zero_cov = np.zeros((p, p))
        risk_log = -np.inf
        risk_mean, risk_cov = zero_mean.copy(), zero_cov.copy()
        gradient = np.zeros(p)
        information = np.zeros((p, p))
        nll = 0.0
        log_risk = np.empty(self.starts.size)

        for group, (start, end) in enumerate(zip(self.starts, self.ends, strict=True)):
            x_block = self.x[start : end + 1]
            eta_block = eta[start : end + 1]
            event_block = self.event[start : end + 1].astype(bool)
            shift = float(np.max(eta_block))
            weight = np.exp(eta_block - shift)
            block_log, block_mean, block_cov = _weighted_moments(x_block, weight, shift)

            deaths = int(self.deaths[group])
            if deaths:
                death_x = x_block[event_block]
                death_eta = eta_block[event_block]
                death_shift = float(np.max(death_eta))
                death_weight = np.exp(death_eta - death_shift)
                death_log, death_mean, death_cov = _weighted_moments(
                    death_x, death_weight, death_shift
                )
                nondeath = ~event_block
                if np.any(nondeath):
                    nondeath_eta = eta_block[nondeath]
                    nondeath_shift = float(np.max(nondeath_eta))
                    nond_log, nond_mean, nond_cov = _weighted_moments(
                        x_block[nondeath],
                        np.exp(nondeath_eta - nondeath_shift),
                        nondeath_shift,
                    )
                else:
                    nond_log, nond_mean, nond_cov = -np.inf, zero_mean, zero_cov
                event_eta_sum = float(np.sum(death_eta))
                for tied_index in range(deaths):
                    fraction = tied_index / deaths
                    components = [(risk_log, risk_mean, risk_cov), (nond_log, nond_mean, nond_cov)]
                    if fraction < 1:
                        components.append(
                            (
                                death_log + float(np.log1p(-fraction)),
                                death_mean,
                                death_cov,
                            )
                        )
                    log_denom, mean, covariance = _combine_moments(components)
                    nll += log_denom - event_eta_sum / deaths
                    gradient += mean
                    information += covariance

            risk_log, risk_mean, risk_cov = _combine_moments(
                [(risk_log, risk_mean, risk_cov), (block_log, block_mean, block_cov)]
            )
            log_risk[group] = risk_log

        # The event linear-predictor term is subtracted once per observed event.
        gradient -= self.event_x
        if (
            not np.isfinite(nll)
            or not np.isfinite(gradient).all()
            or not np.isfinite(information).all()
        ):
            raise ArithmeticError("Efron Cox likelihood moments exceed numerical range")
        return nll, gradient, (information + information.T) / 2, log_risk

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


def _weighted_moments(
    x: FloatArray, weights: FloatArray, log_scale: float
) -> tuple[float, FloatArray, FloatArray]:
    total = float(weights.sum())
    if not total > 0 or not np.isfinite(total):
        raise ArithmeticError("Cox tied-risk weights are not representable")
    mean = weights @ x / total
    centered = x - mean
    covariance = centered.T @ (weights[:, None] * centered) / total
    return log_scale + float(np.log(total)), mean, (covariance + covariance.T) / 2


def _combine_moments(
    components: list[tuple[float, FloatArray, FloatArray]],
) -> tuple[float, FloatArray, FloatArray]:
    finite = [component for component in components if np.isfinite(component[0])]
    if not finite:
        raise ArithmeticError("empty Cox risk set in Efron likelihood")
    logs = np.asarray([component[0] for component in finite])
    log_total = float(logsumexp(logs))
    fractions = np.exp(logs - log_total)
    means = [component[1] for component in finite]
    mean = sum(
        (fraction * item for fraction, item in zip(fractions, means, strict=True)),
        np.zeros_like(means[0]),
    )
    covariance = np.zeros_like(finite[0][2])
    for fraction, (_, component_mean, component_cov) in zip(fractions, finite, strict=True):
        delta = component_mean - mean
        covariance += fraction * (component_cov + np.outer(delta, delta))
    return log_total, mean, (covariance + covariance.T) / 2
