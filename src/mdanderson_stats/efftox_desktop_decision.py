"""Named current-desktop EffTox selection profile with independent inference.

Original 5.2.3 controller comparisons/reachability are retained here. The Lp
shape is solved accurately by Python rather than at the original coarse root
tolerance. Native posterior integrator and RNG identity are not asserted.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._validation import FloatArray
from .efftox_decision import EffToxContour, _probability, _tail_probability
from .efftox_model import EffToxFit, _integer, _owned


def _admissible(
    counts: np.ndarray,
    toxicity_tail: FloatArray,
    efficacy_tail: FloatArray,
    starting_dose: int,
    toxicity_cutoff: float,
    efficacy_cutoff: float,
) -> np.ndarray:
    """Original RVA0xecc0; counts are complete observations per dose."""
    indices = np.arange(len(counts))
    tried = np.flatnonzero(counts > 0)
    acceptable = toxicity_tail >= toxicity_cutoff
    if tried.size:
        acceptable &= (indices >= tried.min() - 1) & (indices <= tried.max() + 1)
        acceptable &= (efficacy_tail > efficacy_cutoff) | (indices == tried.max() + 1)
    else:
        acceptable &= indices == starting_dose
    return acceptable


def _select(
    counts: np.ndarray, admissible: np.ndarray, utility: FloatArray, start: int
) -> int | None:
    """Original RVA0xfc90: unobserved start bypass, first-index utility ties."""
    if counts[start] == 0:
        return start
    candidates = np.flatnonzero(admissible & (utility > -np.finfo(float).max))
    if not len(candidates):
        return None
    return int(candidates[np.argmax(utility[candidates])])


def _toxicity_summary(
    fit: EffToxFit, profile: Literal["model", "desktop_legacy"]
) -> tuple[FloatArray, FloatArray]:
    if profile not in {"model", "desktop_legacy"}:
        raise ValueError("unknown toxicity_summary_profile")
    if profile == "model":
        return fit.toxicity_logits, fit.toxicity_probabilities
    negative = fit.parameters[:, :, 1, None] < 0
    return (
        np.where(negative, fit.toxicity_logits[:, :, :1], fit.toxicity_logits),
        np.where(negative, fit.toxicity_probabilities[:, :, :1], fit.toxicity_probabilities),
    )


@dataclass(frozen=True)
class EffToxDesktopDecision:
    """Zero-based current-desktop dose and captured posterior summaries.

    ``dose_index=None`` means no acceptable dose. Patient capacity, pending
    consistency and cohort handling belong to the separate trial controller.
    """

    dose_index: int | None
    admissible: np.ndarray
    utility: FloatArray
    efficacy_tail_probability: FloatArray
    toxicity_tail_probability: FloatArray
    mean_efficacy: FloatArray | None = None
    mean_toxicity: FloatArray | None = None


def efftox_desktop_decision(
    fit: EffToxFit,
    contour: EffToxContour,
    *,
    starting_dose_index: int,
    efficacy_limit: float,
    toxicity_limit: float,
    efficacy_probability: float,
    toxicity_probability: float,
    toxicity_summary_profile: Literal["model", "desktop_legacy"] = "model",
) -> EffToxDesktopDecision:
    """Apply current desktop selection to a supplied Python binary posterior.

    Unlike the generic profile, the native toxicity-tail cutoff includes
    equality; exploration exempts max(tried)+1 from the efficacy cutoff; and
    reachability is min(tried)-1 through max(tried)+1 for both interim and final
    selection. An unobserved starting dose is selected unconditionally, matching
    the original selector. These compatibility rules are explicit, not clinical
    recommendations or changes to ``efftox_decision`` defaults.

    ``toxicity_summary_profile="desktop_legacy"`` replaces negative-slope
    toxicity summaries with the first-dose value, as in the original integrand.
    The default "model" preserves each dose's modeled toxicity. Neither option
    changes the fitted likelihood.
    """
    # The named legacy option reproduces the native negative-slope summary clamp
    # without changing the supplied posterior likelihood or generic defaults.
    if not isinstance(fit, EffToxFit) or not isinstance(contour, EffToxContour):
        raise TypeError("current desktop selection requires a binary EffToxFit and Lp contour")
    start = _integer(starting_dose_index, "starting_dose_index", 0, len(fit.doses) - 1)
    e_limit = _probability(efficacy_limit, "efficacy_limit")
    t_limit = _probability(toxicity_limit, "toxicity_limit")
    e_cutoff = _probability(efficacy_probability, "efficacy_probability")
    t_cutoff = _probability(toxicity_probability, "toxicity_probability")
    counts = fit.counts.sum(axis=(1, 2))
    p_e = _tail_probability(fit.efficacy_logits, e_limit, upper=True)
    toxicity_logits, toxicity_probabilities = _toxicity_summary(fit, toxicity_summary_profile)
    p_t = _tail_probability(toxicity_logits, t_limit, upper=False)
    acceptable = _admissible(counts, p_t, p_e, start, t_cutoff, e_cutoff)
    utility = contour.utility(
        fit.efficacy_probabilities.mean(axis=(0, 1)),
        toxicity_probabilities.mean(axis=(0, 1)),
    )
    selected = _select(counts, acceptable, utility, start)
    acceptable.flags.writeable = False
    return EffToxDesktopDecision(
        selected,
        acceptable,
        _owned(utility),
        p_e,
        p_t,
        _owned(fit.efficacy_probabilities.mean(axis=(0, 1))),
        _owned(toxicity_probabilities.mean(axis=(0, 1))),
    )
