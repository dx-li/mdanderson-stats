"""BOP2 ordinal and multiple efficacy: Dirichlet monitoring and exact paired-count OC."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import BayesianMonitoringDesign, _owned
from .bop2_binary import bop2_binary_design


def _cells(rates: ArrayLike, joint_rate: float | None, endpoint: str) -> FloatArray:
    p = finite(rates, "rates")
    if p.shape != (2,) or np.any((p < 0) | (p > 1)):
        raise ValueError("rates must contain two probabilities in [0,1]")
    if endpoint == "ordinal":
        if joint_rate is not None or p[0] > p[1]:
            raise ValueError("ordinal requires CR <= CR+PR and no joint_rate")
        cells = np.array([p[0], p[1] - p[0], 1 - p[1]])
    elif endpoint == "multiple":
        if joint_rate is None:
            raise ValueError("multiple efficacy requires an explicit joint_rate")
        joint = scalar(joint_rate, "joint_rate")
        if not max(0, p[0] + p[1] - 1) <= joint <= min(p):
            raise ValueError("joint_rate violates the joint probability bounds")
        # Use the same sum as the lower-bound check for the final cell.
        cells = np.array([joint, p[0] - joint, p[1] - joint, 1 - (p[0] + p[1]) + joint])
    else:
        raise ValueError("endpoint must be ordinal or multiple")
    return cells


def _increments(endpoint: str) -> tuple[tuple[int, int], ...]:
    return ((1, 1), (0, 1), (0, 0)) if endpoint == "ordinal" else ((1, 1), (1, 0), (0, 1), (0, 0))


@dataclass(frozen=True)
class BOP2PairedState:
    sample_size: FloatArray
    counts: FloatArray
    posterior_shapes: FloatArray
    marginal_success_probability: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BOP2PairedOperatingCharacteristics:
    category_probability: FloatArray
    looks: NDArray[np.int64]
    stop_probability: FloatArray
    success_probability: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray
    sample_size_sd: FloatArray

    @property
    def early_stop_probability(self) -> FloatArray:
        return self.stop_probability[..., :-1].sum(axis=-1)


@dataclass(frozen=True)
class BOP2PairedDesign:
    """Both marginal futility conditions must hold, including at the final look.

    Category order: ordinal (CR, PR, other); multiple (11, 10, 01, 00).
    Construct with bop2_paired_design. All observations must be fully evaluated.
    """

    endpoint: str
    prior: FloatArray
    marginals: tuple[BayesianMonitoringDesign, BayesianMonitoringDesign]

    @property
    def max_subjects(self) -> int:
        return self.marginals[0].max_subjects

    @property
    def looks(self) -> NDArray[np.int64]:
        return self.marginals[0].looks

    @property
    def futility_max(self) -> NDArray[np.int64]:
        bounds = np.stack([m.futility_max for m in self.marginals], axis=-1)
        bounds.flags.writeable = False
        return bounds

    def monitor(self, counts: ArrayLike) -> BOP2PairedState:
        x = count(counts, "counts")
        if x.ndim == 0 or x.shape[-1] != self.prior.size:
            raise ValueError("counts must have a final category axis matching the prior")
        n = x.sum(axis=-1)
        if np.any(n > self.max_subjects):
            raise ValueError("total counts exceed max_subjects")
        events = x @ np.array(_increments(self.endpoint))
        states = [m.monitor(events[..., j], n) for j, m in enumerate(self.marginals)]
        bad = np.logical_and.reduce(
            [(s.decision == "stop_futility") | (s.decision == "final_negative") for s in states]
        )
        decisions = np.full(n.shape, "continue", dtype="U24")
        decisions[bad & (n < self.max_subjects)] = "stop_futility"
        decisions[(n == self.max_subjects) & bad] = "final_negative"
        decisions[(n == self.max_subjects) & ~bad] = "final_positive"
        decisions.flags.writeable = False
        return BOP2PairedState(
            _owned(n),
            _owned(x),
            _owned(x + self.prior),
            _owned(np.stack([s.high_probability for s in states], axis=-1)),
            decisions,
        )

    def monitor_outcomes(self, outcomes: ArrayLike) -> BOP2PairedState:
        """Zero-based category codes on the final patient axis; retain the first stop."""
        y = count(outcomes, "outcomes")
        if y.ndim == 0 or y.shape[-1] > self.max_subjects or np.any(y >= self.prior.size):
            raise ValueError(
                "outcomes require a patient axis with category codes 0..K-1, at most N"
            )
        state = self.monitor(np.cumsum(y[..., None] == np.arange(self.prior.size), axis=-2))
        decisions = state.decision.copy()
        for j in range(1, y.shape[-1]):
            decisions[..., j] = np.where(
                decisions[..., j - 1] != "continue", decisions[..., j - 1], decisions[..., j]
            )
        decisions.flags.writeable = False
        return BOP2PairedState(
            state.sample_size,
            state.counts,
            state.posterior_shapes,
            state.marginal_success_probability,
            decisions,
        )

    def operating_characteristics(
        self, category_probability: ArrayLike
    ) -> BOP2PairedOperatingCharacteristics:
        """Exact forward recursion over the two marginal counts, preserving dependence."""
        p = finite(category_probability, "category_probability")
        if p.ndim == 0 or p.shape[-1] != self.prior.size or np.any((p < 0) | (p > 1)):
            raise ValueError(
                "category_probability requires a final category axis with probabilities"
            )
        total = p.sum(axis=-1)
        if np.any(np.abs(total - 1) > 1e-14):
            raise ValueError("category probabilities must sum to one")
        p = p / total[..., None]
        if total.size * (self.max_subjects + 1) ** 2 > 5_000_000:
            raise ValueError("scenario batch too large; split into smaller batches")
        surviving = np.ones((*p.shape[:-1], 1, 1))
        stop = np.zeros((*p.shape[:-1], self.looks.size))
        bounds = self.futility_max
        index = 0
        for n in range(1, self.max_subjects + 1):
            arriving = np.zeros((*p.shape[:-1], n + 1, n + 1))
            for k, (a, b) in enumerate(_increments(self.endpoint)):
                arriving[..., a : a + n, b : b + n] += surviving * p[..., k, None, None]
            if n == self.looks[index]:
                a, b = bounds[index] + 1
                stop[..., index] = arriving[..., :a, :b].sum(axis=(-2, -1))
                arriving[..., :a, :b] = 0
                index += 1
            surviving = arriving
        success = surviving.sum(axis=(-2, -1))
        pmf = stop.copy()
        pmf[..., -1] += success
        expected = pmf @ self.looks
        sd = np.sqrt(np.sum(pmf * (self.looks - expected[..., None]) ** 2, axis=-1))
        return BOP2PairedOperatingCharacteristics(
            _owned(p),
            self.looks,
            _owned(stop),
            _owned(success),
            _owned(pmf),
            _owned(expected),
            _owned(sd),
        )


def bop2_paired_design(
    max_subjects: int,
    null_rates: ArrayLike,
    *,
    cutoff_scale: float,
    gamma: float,
    endpoint: str = "ordinal",
    null_joint_rate: float | None = None,
    looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2PairedDesign:
    """Specified-parameter ordinal/multiple efficacy design; default prior has ESS one."""
    cells = _cells(null_rates, null_joint_rate, endpoint)
    shapes = cells if prior is None else finite(prior, "prior")
    if shapes.shape != cells.shape or np.any(shapes <= 0) or not np.isfinite(shapes.sum()):
        raise ValueError("prior requires one positive finite shape per category with a finite sum")
    if (
        isinstance(max_subjects, (bool, np.bool_))
        or not isinstance(max_subjects, (int, np.integer))
        or not 1 <= max_subjects <= 200
    ):
        raise ValueError("max_subjects must be an integer in [1,200]")
    masks = np.array(_increments(endpoint), dtype=bool)
    marginal = [
        bop2_binary_design(
            max_subjects,
            float(np.asarray(null_rates)[j]),
            cutoff_scale=cutoff_scale,
            gamma=gamma,
            looks=looks,
            prior=[shapes[masks[:, j]].sum(), shapes[~masks[:, j]].sum()],
            min_subjects=min_subjects,
            cohort_size=cohort_size,
        )
        for j in range(2)
    ]
    return BOP2PairedDesign(endpoint, _owned(shapes), (marginal[0], marginal[1]))
