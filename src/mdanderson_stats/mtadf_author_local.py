"""Author-reference local logistic MTADF policy.

This module is separate from the paper-based local logistic API because the
author R reference uses ordinal dose scaling and different slope-probability
gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .hierarchical_binomial import summarize_chains
from .mtadf_author import (
    MTADFAuthorDecision,
    _author_limits,
    _author_safety_state,
    _mtadf_author_admissible_dose_count,
    _scalar,
    mtadf_author_decision,
)
from .mtadf_logistic import (
    MTADFLocalLogisticPosterior,
    _counts,
    _mcmc_options,
    _preflight_local_decision,
    _preflight_work,
    _sampler,
)

_MAX_DOSES = 20
_MAX_SUBJECTS = 10_000
_AUTHOR_CAUCHY_SCALES = np.array([10.0, 2.5], dtype=np.float64)


def _author_mcmc_options(draws: int, warmup: int, chains: int) -> tuple[int, int, int]:
    if any(isinstance(value, (bool, np.bool_)) for value in (draws, warmup, chains)):
        raise ValueError("draws, warmup, and chains must be integer settings, not booleans")
    return _mcmc_options(draws, warmup, chains)


def _freeze_bool_array(values: ArrayLike) -> NDArray[np.bool_]:
    contiguous = np.ascontiguousarray(values, dtype=np.bool_)
    return np.frombuffer(contiguous.tobytes(), dtype=np.bool_).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class MTADFAuthorLocalDecision:
    """One source-style local logistic action and its safety/posterior state.

    Dose indices are zero-based. ``admissible`` is the prefix used to cap this
    action; ``admissible_dose_count`` always reports the freshly computed
    prefix, which can differ for the author's lagged simulation convention.
    """

    action: str
    dose: int
    reason: str
    raw_overdose_probability: FloatArray
    adjusted_overdose_probability: FloatArray
    admissible: NDArray[np.bool_]
    admissible_dose_count: int
    admissibility_count_used: int
    backward_posterior: MTADFLocalLogisticPosterior | None
    forward_posterior: MTADFLocalLogisticPosterior | None
    probability_backward_nonpositive: float | None
    probability_forward_positive: float | None
    final_selection: MTADFAuthorDecision | None

    @property
    def fit_count(self) -> int:
        return int(self.backward_posterior is not None) + int(self.forward_posterior is not None)


def _author_local_doses(dose_count: int) -> FloatArray:
    if isinstance(dose_count, (bool, np.bool_)) or not isinstance(dose_count, (int, np.integer)):
        raise ValueError("dose_count must be an integer")
    if not 2 <= int(dose_count) <= _MAX_DOSES:
        raise ValueError(f"author local logistic requires 2..{_MAX_DOSES} dose levels")
    ordinal = np.arange(1, int(dose_count) + 1, dtype=np.float64)
    return (ordinal - np.mean(ordinal)) / (2.0 * np.std(ordinal, ddof=1))


def mtadf_author_local_posterior(
    subjects: ArrayLike,
    responses: ArrayLike,
    *,
    lower_dose: int,
    draws: int = 2_000,
    warmup: int = 1_000,
    chains: int = 4,
    rng: np.random.Generator,
) -> MTADFLocalLogisticPosterior:
    """Fit one author adjacent-dose slope posterior.

    ``lower_dose`` is the zero-based lower index of the two-dose window.
    ``window_doses`` in the returned established result type contains the
    author's globally standardized ordinal coordinates, not supplied dose
    concentrations. A zero-count neighbor is allowed; at least one subject
    across the pair is required. The sampler is the package's bounded Python
    random-walk implementation, not the native ``mcmc::metrop`` stream.
    """

    n, y = _counts(subjects, responses, allow_empty=False)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator")
    if n.size < 2:
        raise ValueError("author local logistic requires at least two dose levels")
    if isinstance(lower_dose, (bool, np.bool_)) or not isinstance(lower_dose, (int, np.integer)):
        raise ValueError("lower_dose must be a zero-based integer dose index")
    lo = int(lower_dose)
    if not 0 <= lo < n.size - 1:
        raise ValueError("lower_dose must identify a valid adjacent-dose pair")
    if not np.any(n[lo : lo + 2]):
        raise ValueError("the fitted adjacent-dose pair must contain an observed subject")
    draws, warmup, chains = _author_mcmc_options(draws, warmup, chains)
    _preflight_work(draws, warmup, chains, 2, 2)
    x = _author_local_doses(n.size)
    indices = slice(lo, lo + 2)
    design = np.column_stack((np.ones(2, dtype=np.float64), x[indices]))
    parameters, efficacy, acceptance = _sampler(
        design,
        n[indices],
        y[indices],
        _AUTHOR_CAUCHY_SCALES,
        draws=draws,
        warmup=warmup,
        chains=chains,
        rng=rng,
    )
    positive = parameters[:, :, 1] > 0.0
    positive_summary = summarize_chains(positive)
    return MTADFLocalLogisticPosterior(
        _freeze(x[indices]),
        _freeze(n[indices]),
        _freeze(y[indices]),
        parameters,
        efficacy,
        summarize_chains(parameters),
        summarize_chains(efficacy),
        _freeze_bool_array(positive),
        positive_summary,
        float(np.mean(positive)),
        float(positive_summary.batch_mean_mcse),
        acceptance,
        warmup,
        2,
    )


def _author_local_gate(
    current_dose: int,
    dose_count: int,
    *,
    backward_nonpositive: float | None,
    forward_positive: float | None,
    next_dose_observed: bool,
    ce1: float,
    ce2: float,
) -> tuple[int, str]:
    """Apply the reference's strict local-slope probability gates."""
    if current_dose == 0:
        if forward_positive is None:
            raise ValueError("a forward slope probability is required at the lowest dose")
        if forward_positive > ce1:
            return current_dose + 1, "lowest_dose_forward_slope_exceeds_ce1"
        return current_dose, "lowest_dose_forward_slope_does_not_exceed_ce1"

    if backward_nonpositive is None:
        raise ValueError("a backward slope probability is required above the lowest dose")
    if current_dose == dose_count - 1:
        if backward_nonpositive > 1.0 - ce1:
            return current_dose - 1, "highest_dose_backward_slope_exceeds_deescalation_gate"
        return current_dose, "highest_dose_backward_slope_within_hold_gate"

    if next_dose_observed:
        if forward_positive is None:
            raise ValueError(
                "a forward slope probability is required when the next dose is observed"
            )
        if backward_nonpositive <= 1.0 - ce2 and forward_positive > ce1:
            return current_dose + 1, "two_sided_escalation_gates_passed"
        if backward_nonpositive > 1.0 - ce1:
            return current_dose - 1, "backward_slope_exceeds_deescalation_gate"
        return current_dose, "two_sided_gates_hold"

    if backward_nonpositive > 1.0 - ce1:
        return current_dose - 1, "untried_next_backward_slope_exceeds_deescalation_gate"
    if backward_nonpositive <= 1.0 - ce2:
        return current_dose + 1, "untried_next_backward_slope_passes_escalation_gate"
    return current_dose, "untried_next_backward_slope_holds"


def _mtadf_author_local_decision_with_count(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    responses: ArrayLike,
    *,
    current_dose: int | None,
    final: bool = False,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    ce1: float = 0.3,
    ce2: float = 0.4,
    draws: int = 2_000,
    warmup: int = 1_000,
    chains: int = 4,
    rng: np.random.Generator,
    admissibility_count: int,
) -> MTADFAuthorLocalDecision:
    n, ytox, raw, adjusted, fresh_count = _author_safety_state(
        subjects,
        toxicities,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
    )
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator")
    n_float, response = _counts(n, responses, allow_empty=True)
    if n_float.shape != n.shape or not np.array_equal(n_float, n.astype(np.float64)):
        raise ValueError("subjects and responses must be matching count vectors")
    if int(n.sum()) != int(n_float.sum()) or int(np.sum(response)) > int(n.sum()):
        raise ValueError("response counts must match subjects and cannot exceed enrollment")
    dose_count = int(n.size)
    _author_local_doses(dose_count)
    phi, cutoff = _author_limits(toxicity_limit, safety_cutoff)
    ce1_value = _scalar(ce1, "ce1")
    ce2_value = _scalar(ce2, "ce2")
    if not 0 <= ce1_value <= ce2_value <= 1:
        raise ValueError("local author cutoffs must satisfy 0 <= ce1 <= ce2 <= 1")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    if isinstance(admissibility_count, (bool, np.bool_)) or not isinstance(
        admissibility_count, (int, np.integer)
    ):
        raise ValueError("admissibility_count must be an integer prefix length")
    used_count = int(admissibility_count)
    if not 1 <= used_count <= dose_count:
        raise ValueError("admissibility_count must lie from 1 to the number of doses")
    if current_dose is not None and (
        isinstance(current_dose, (bool, np.bool_))
        or not isinstance(current_dose, (int, np.integer))
        or not 0 <= int(current_dose) < dose_count
    ):
        raise ValueError("current_dose must be a zero-based dose index")

    admissible = np.arange(dose_count) < used_count
    if final:
        if int(n.sum()) == 0:
            raise ValueError("final author selection requires at least one observed subject")
        selection = mtadf_author_decision(
            n,
            ytox,
            response,
            final=True,
            toxicity_limit=phi,
            safety_cutoff=cutoff,
        )
        assert selection.dose is not None
        return MTADFAuthorLocalDecision(
            "select_obd",
            int(selection.dose),
            "author_all_dose_epsilon_unimodal_final_selection",
            selection.raw_overdose_probability,
            selection.adjusted_overdose_probability,
            selection.admissible,
            selection.admissible_dose_count,
            selection.admissibility_count_used,
            None,
            None,
            None,
            None,
            selection,
        )

    if int(n.sum()) == 0:
        if current_dose is not None:
            raise ValueError("current_dose must be omitted before enrollment")
        return MTADFAuthorLocalDecision(
            "start",
            0,
            "author_local_start_at_lowest_dose",
            _freeze(raw),
            _freeze(adjusted),
            _freeze_bool_array(admissible),
            fresh_count,
            used_count,
            None,
            None,
            None,
            None,
            None,
        )
    if current_dose is None:
        raise ValueError("current_dose is required after enrollment has begun")
    current = int(current_dose)
    if n[current] == 0:
        raise ValueError("current_dose must have observed subjects")
    if used_count == 1:
        return MTADFAuthorLocalDecision(
            "treat",
            0,
            "author_safety_cap_is_lowest_dose",
            _freeze(raw),
            _freeze(adjusted),
            _freeze_bool_array(admissible),
            fresh_count,
            used_count,
            None,
            None,
            None,
            None,
            None,
        )

    draws, warmup, chains = _author_mcmc_options(draws, warmup, chains)
    needs_backward = current > 0
    needs_forward = current == 0 or (current < dose_count - 1 and n[current + 1] > 0)
    generated_count = int(needs_backward) + int(needs_forward)
    _preflight_local_decision(
        2,
        [(chains, draws)] * generated_count,
        generated_count,
        draws,
        warmup,
        chains,
    )

    forward: MTADFLocalLogisticPosterior | None = None
    backward: MTADFLocalLogisticPosterior | None = None
    if needs_forward:
        forward = mtadf_author_local_posterior(
            n,
            response,
            lower_dose=current,
            draws=draws,
            warmup=warmup,
            chains=chains,
            rng=rng,
        )
    if needs_backward:
        backward = mtadf_author_local_posterior(
            n,
            response,
            lower_dose=current - 1,
            draws=draws,
            warmup=warmup,
            chains=chains,
            rng=rng,
        )

    p_backward = (
        None if backward is None else float(np.mean(backward.parameter_draws[:, :, 1] <= 0.0))
    )
    p_forward = None if forward is None else forward.probability_positive_slope
    candidate, reason = _author_local_gate(
        current,
        dose_count,
        backward_nonpositive=p_backward,
        forward_positive=p_forward,
        next_dose_observed=bool(current < dose_count - 1 and n[current + 1] > 0),
        ce1=ce1_value,
        ce2=ce2_value,
    )
    selected = min(candidate, used_count - 1)
    if selected != candidate:
        reason += "_capped_by_admissible_prefix"
    return MTADFAuthorLocalDecision(
        "treat",
        selected,
        reason,
        _freeze(raw),
        _freeze(adjusted),
        _freeze_bool_array(admissible),
        fresh_count,
        used_count,
        backward,
        forward,
        p_backward,
        p_forward,
        None,
    )


def mtadf_author_local_decision(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    responses: ArrayLike,
    *,
    current_dose: int | None = None,
    final: bool = False,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    ce1: float = 0.3,
    ce2: float = 0.4,
    draws: int = 2_000,
    warmup: int = 1_000,
    chains: int = 4,
    rng: np.random.Generator,
) -> MTADFAuthorLocalDecision:
    """Apply the authors' adjacent-window local logistic rule.

    The default cutoffs are the reference values. The sampler is an explicit
    bounded Python MCMC choice; results do not claim the reference R RNG or
    ``mcmc::metrop`` proposal stream.
    """
    fresh = _mtadf_author_admissible_dose_count(
        subjects,
        toxicities,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
    )
    return _mtadf_author_local_decision_with_count(
        subjects,
        toxicities,
        responses,
        current_dose=current_dose,
        final=final,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
        ce1=ce1,
        ce2=ce2,
        draws=draws,
        warmup=warmup,
        chains=chains,
        rng=rng,
        admissibility_count=fresh,
    )
