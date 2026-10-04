"""Across-analysis summaries for the six-dose Phase I/II probability vector."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ._cdflib import _freeze
from ._validation import FloatArray
from .parallel_phase12_importance import Phase12ImportanceFit

_DOSES = 6
_COMPONENTS = 4 * _DOSES + _DOSES * _DOSES
_MAX_ANALYSIS_CALLS = 100_000

_COMPONENT_LABELS = (
    *(f"reference_superiority[{dose}]" for dose in range(_DOSES)),
    *(f"efficacy_probability[{dose}]" for dose in range(_DOSES)),
    *(f"future_probability[{dose}]" for dose in range(_DOSES)),
    *(
        f"pairwise_superiority[{dose},{comparator}]"
        for dose in range(_DOSES)
        for comparator in range(_DOSES)
    ),
    *(f"response_probability_second_moment[{dose}]" for dose in range(_DOSES)),
)


@dataclass(frozen=True)
class Phase12ProbabilitySummary:
    """Componentwise summary over an explicitly supplied sequence of fits.

    Component arrays follow ``component_labels``. Variances are sample
    variances across analysis calls, not posterior variances or Monte Carlo
    standard errors. A single supplied fit has zero variance, matching the
    archived running accumulator.
    """

    analysis_call_count: int
    nonconverged_analysis_call_count: int
    component_labels: tuple[str, ...]
    component_means: FloatArray
    component_sample_variances: FloatArray


class _Phase12ProbabilityMoments:
    """Fixed-size streaming/merge accumulator used by the calendar workflow."""

    def __init__(self, *, max_count: int | None = None) -> None:
        self._max_count = max_count
        self.count = 0
        self.nonconverged_count = 0
        self.mean = np.zeros(_COMPONENTS)
        self.m2 = np.zeros(_COMPONENTS)

    def _add(self, observation: np.ndarray, nonconverged: bool) -> None:
        if self._max_count is not None and self.count >= self._max_count:
            raise ValueError("analysis-call summary exceeds its bounded limit")
        self.count += 1
        self.nonconverged_count += int(nonconverged)
        delta = observation - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (observation - self.mean)

    def add_fit(self, fit: Phase12ImportanceFit) -> None:
        observation, nonconverged = _fit_vector(fit)
        self._add(observation, nonconverged)

    def merge(self, summary: Phase12ProbabilitySummary) -> None:
        if not isinstance(summary, Phase12ProbabilitySummary):
            raise TypeError("summary must be a Phase12ProbabilitySummary")
        if summary.analysis_call_count < 1:
            raise ValueError("summary must contain at least one analysis call")
        if summary.component_labels != _COMPONENT_LABELS:
            raise ValueError("summary component labels do not match the six-dose kernel")
        if summary.component_means.shape != (
            _COMPONENTS,
        ) or summary.component_sample_variances.shape != (_COMPONENTS,):
            raise ValueError("summary component arrays must each have 60 values")
        n_b = summary.analysis_call_count
        if self._max_count is not None and self.count + n_b > self._max_count:
            raise ValueError("analysis-call summary exceeds its bounded limit")
        if self.count == 0:
            self.count = n_b
            self.nonconverged_count = summary.nonconverged_analysis_call_count
            self.mean = np.array(summary.component_means, copy=True)
            self.m2 = np.array(summary.component_sample_variances, copy=True) * max(n_b - 1, 0)
            return
        n_a = self.count
        n = n_a + n_b
        delta = summary.component_means - self.mean
        self.mean += delta * (n_b / n)
        self.m2 += np.asarray(summary.component_sample_variances) * max(
            n_b - 1, 0
        ) + delta * delta * (n_a * n_b / n)
        self.count = n
        self.nonconverged_count += summary.nonconverged_analysis_call_count

    def to_summary(self, *, allow_empty: bool = False) -> Phase12ProbabilitySummary | None:
        if self.count == 0:
            if allow_empty:
                return None
            raise ValueError("at least one analysis call is required")
        variance = self.m2 / (self.count - 1) if self.count > 1 else np.zeros(_COMPONENTS)
        return Phase12ProbabilitySummary(
            self.count,
            self.nonconverged_count,
            _COMPONENT_LABELS,
            _freeze(self.mean),
            _freeze(variance),
        )


def _probability_vector(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype.kind not in "fiu"
        or np.iscomplexobj(value)
    ):
        raise ValueError(f"{name} must be a real numeric array with shape {shape}")
    if np.any(~np.isfinite(value)) or np.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must contain finite probabilities in [0,1]")
    return value


def _fit_vector(fit: Phase12ImportanceFit) -> tuple[np.ndarray, bool]:
    if not isinstance(fit, Phase12ImportanceFit):
        raise TypeError("fits must contain Phase12ImportanceFit objects")
    reference = _probability_vector(fit.reference_superiority, (6,), "reference_superiority")
    efficacy = _probability_vector(fit.efficacy_probability, (6,), "efficacy_probability")
    future = _probability_vector(fit.future_probability, (6,), "future_probability")
    pairwise = _probability_vector(fit.pairwise_superiority, (6, 6), "pairwise_superiority")
    second_moment = _probability_vector(
        fit.response_probability_second_moment, (6,), "response_probability_second_moment"
    )
    if not isinstance(fit.converged, (bool, np.bool_)):
        raise ValueError("fit.converged must be boolean")

    observation = np.empty(_COMPONENTS)
    offset = 0
    for values in (reference, efficacy, future):
        observation[offset : offset + _DOSES] = values
        offset += _DOSES
    observation[offset : offset + _DOSES * _DOSES] = pairwise.reshape(-1)
    offset += _DOSES * _DOSES
    observation[offset : offset + _DOSES] = second_moment
    return observation, not bool(fit.converged)


def summarize_phase12_importance_fits(
    fits: Iterable[Phase12ImportanceFit], *, max_fits: int = _MAX_ANALYSIS_CALLS
) -> Phase12ProbabilitySummary:
    """Stream a bounded sequence of successful importance-fit results.

    The component order matches the archived six-dose kernel: reference
    superiority, efficacy-threshold indicators, future-study-threshold
    indicators, row-major ordered pairwise superiority, then squared response
    probabilities. Nonconverged fits are included, as estimates returned at
    the source integration cap are included by the native aggregator; their
    count is reported explicitly. This function does not imply independent
    analysis calls or calculate Monte Carlo errors.
    """
    if isinstance(max_fits, (bool, np.bool_)) or not isinstance(max_fits, (int, np.integer)):
        raise ValueError("max_fits must be an integer")
    limit = int(max_fits)
    if not 1 <= limit <= _MAX_ANALYSIS_CALLS:
        raise ValueError(f"max_fits must be in [1,{_MAX_ANALYSIS_CALLS}]")
    try:
        iterator: Iterator[Phase12ImportanceFit] = iter(fits)
    except TypeError as exc:
        raise TypeError("fits must be an iterable of Phase12ImportanceFit objects") from exc

    moments = _Phase12ProbabilityMoments(max_count=limit)
    for fit in iterator:
        if moments.count == limit:
            raise ValueError(f"fits exceeds the bounded limit of {limit} analysis calls")
        moments.add_fit(fit)
    summary = moments.to_summary()
    assert summary is not None
    return summary
