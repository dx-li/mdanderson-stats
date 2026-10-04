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

    mean = np.zeros(_COMPONENTS)
    m2 = np.zeros(_COMPONENTS)
    count = 0
    nonconverged_count = 0
    for fit in iterator:
        if count == limit:
            raise ValueError(f"fits exceeds the bounded limit of {limit} analysis calls")
        observation, nonconverged = _fit_vector(fit)
        count += 1
        nonconverged_count += int(nonconverged)
        delta = observation - mean
        mean += delta / count
        m2 += delta * (observation - mean)

    if count == 0:
        raise ValueError("fits must contain at least one Phase12ImportanceFit")
    sample_variance = m2 / (count - 1) if count > 1 else np.zeros(_COMPONENTS)
    return Phase12ProbabilitySummary(
        count,
        nonconverged_count,
        _COMPONENT_LABELS,
        _freeze(mean),
        _freeze(sample_variance),
    )
