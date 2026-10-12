"""Saved desktop TITE settings and its days-based pending follow-up calculator."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .boin import _owned
from .boin_desktop import BOINDesktopStudy, _boolean, _number, _text
from .tite_keyboard import toxicity_followup_weights

_DAYS_PER_MONTH = 30.436666666666667


@dataclass(frozen=True)
class BOINDesktopTITESettings:
    window_months: float
    window_days: float
    accrual_per_month: float
    max_pending_fraction: float
    trimester_weights: tuple[float, float, float] | None
    raw_trimester_weights: tuple[float, float, float] | None
    distribution: str
    late_probability: float | None


@dataclass(frozen=True)
class BOINDesktopFollowup:
    followup_days: np.ndarray
    weights: np.ndarray
    total: float
    window_days: float


def _nodes(study):
    if study.mode != "tite":
        raise ValueError("saved TITE study required")
    sim = ET.fromstring(study.xml).find("SimDat")
    assert sim is not None
    design = sim.find("TrialDesign")
    assert design is not None
    return sim, design


def _windows(design):
    window = _number(design, "DLTAssessmentWindow")
    unit = _text(design, "DLTAssessmentWindow_Unit")
    if unit not in ("Days", "Months") or window <= 0:
        raise ValueError("require positive assessment window in Days or Months")
    months = window / _DAYS_PER_MONTH if unit == "Days" else window
    days = window if unit == "Days" else window * _DAYS_PER_MONTH
    if not np.isfinite([months, days]).all() or min(months, days) <= 0:
        raise ValueError("assessment window conversion is not representable")
    return months, days


def _prior(design):
    if _boolean(design, "UseUniform_ToxPriorProbs"):
        return None
    values = np.array([float(n.text or "") for n in design.findall("ToxPriorProbs/double")])
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not 0 < sum(float(v) for v in values) < np.inf
    ):
        raise ValueError("active trimester prior requires three nonnegative finite weights")
    return tuple(float(v) for v in values)


def desktop_tite_settings(study: BOINDesktopStudy) -> BOINDesktopTITESettings:
    """Resolve simulation settings, normalizing active timing weights as native R does.

    Accrual is per month. Inactive uniform-timing late-half values and calculator
    observations do not alter simulation settings. The calculator separately
    validates raw prior probabilities rather than silently normalizing them.
    """
    sim, design = _nodes(study)
    months, days = _windows(design)
    accrual = _number(design, "AccrualRate")
    pending = _number(design, "MaxPendingOutcomePercent") / 100
    if accrual <= 0 or not 0 <= pending <= 0.65:
        raise ValueError("require positive monthly accrual and pending fraction in [0,.65]")
    raw = _prior(design)
    weights = None if raw is None else tuple(v / sum(raw) for v in raw)
    distribution = {
        "eUniform": "uniform",
        "eWeibull": "weibull",
        "eLogLogistic": "log-logistic",
    }.get(_text(sim, "TimeToToxDistribution"))
    if distribution is None:
        raise ValueError("unsupported saved event-time distribution")
    late = None if distribution == "uniform" else _number(sim, "ProbDLTInLateHalf")
    if late is not None and not 0 < late < 1:
        raise ValueError("nonuniform timing requires late-half probability in (0,1)")
    return BOINDesktopTITESettings(months, days, accrual, pending, weights, raw, distribution, late)


def boin_desktop_stft(study: BOINDesktopStudy) -> BOINDesktopFollowup:
    """Calculate saved follow-up weights; native saved observations always use days.

    Only the assessment window, timing prior and calculator observations are
    active here. Nonuniform raw prior probabilities must sum to one; malformed
    priors are rejected rather than returning an unnormalized pseudo-CDF.
    """
    sim, design = _nodes(study)
    _, days = _windows(design)
    nodes = sim.findall("TITE_BOIN_PendingPatientData")
    if len(nodes) != 1 or any(n.tag != "double" for n in nodes[0]):
        raise ValueError("expected one pending-patient follow-up array")
    followup = np.array([float(n.text or "") for n in nodes[0]])
    if len(followup) > 1000:
        raise ValueError("calculator supports at most 1000 pending follow-up records")
    weights = toxicity_followup_weights(followup, days, trimester_probabilities=_prior(design))
    return BOINDesktopFollowup(_owned(followup), _owned(weights), float(weights.sum()), days)
