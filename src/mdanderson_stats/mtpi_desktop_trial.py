"""Current mTPI trial replay with separate paper and recovered desktop profiles."""

from dataclasses import dataclass

import numpy as np
from scipy.special import betainc

from .boin import _owned
from .mtpi_desktop import MTPIDesktopStudy, _integer
from .mtpi_legacy_selection import MTPILegacySelection, select_mtpi_legacy_mtd


@dataclass(frozen=True)
class MTPIDesktopTrial:
    patients: np.ndarray
    toxicities: np.ndarray
    dose_path: tuple[int, ...]
    cohort_sizes: tuple[int, ...]
    selected: int | None
    stop_reason: str
    first_excluded_dose: int | None
    next_dose_indicator: int | None
    profile: str
    unsafe_assignments: tuple[int, ...]
    eliminated: np.ndarray


def _capacity(study, profile):
    if not isinstance(study, MTPIDesktopStudy):
        raise ValueError("study must be MTPIDesktopStudy")
    if profile not in ("safe", "desktop_legacy"):
        raise ValueError("profile must be safe or desktop_legacy")
    capacity = study.maximum_patients
    if profile == "desktop_legacy":
        if study.starting_dose == len(study.scenarios[0]):
            raise ValueError("native next-dose indicator is undefined when starting at top dose")
        capacity = ((capacity + study.cohort_size - 1) // study.cohort_size) * study.cohort_size
        if capacity > 200:
            raise ValueError("native whole-cohort enrollment exceeds supported 200-patient bound")
    return capacity


def _legacy_step(study, n, y, d, tt, excluded):
    D = len(n)
    target = study.target
    lower = target - study.epsilon_lower
    upper = target + study.epsilon_upper
    if d < D - 1:
        tt = int(1 - betainc(y[d + 1] + 0.5, n[d + 1] + 0.5, target) > 0.95)
        if tt:
            excluded = d + 1
    if tt is None:
        raise ValueError("native next-dose indicator is undefined when starting at top dose")
    a = y[d] + 1
    b = n[d] - y[d] + 1
    q = [
        (1 - betainc(a, b, upper)) / (1 - upper),
        (betainc(a, b, upper) - betainc(a, b, lower)) / (upper - lower),
        betainc(a, b, lower) / lower * (1 - tt),
    ]
    stop = ""
    if d == 0:
        if 1 - betainc(y[d] + 0.5, n[d] - y[d] + 0.5, target) > 0.95:
            stop = "stop_safety"
        elif q[2] > q[0] and q[2] > q[1]:
            d += 1
    elif d == D - 1:
        if betainc(y[d] + 0.5, n[d] - y[d] + 0.5, target) > 0.8:
            stop = "stop_underdose"
        elif q[0] > q[1] and q[0] > q[2]:
            d -= 1
    elif q[0] > q[1] and q[0] > q[2]:
        d -= 1
    elif q[2] > q[0] and q[2] > q[1]:
        d += 1
    return d, tt, excluded, stop


def select_mtpi_desktop_legacy_mtd(
    patients, toxicities, *, target=0.3, first_excluded_dose=None, no_selection=False
):
    """XLL final ranking; explicit legacy estimator, not a safety recommendation."""
    n, y = patients, toxicities
    result = select_mtpi_legacy_mtd(
        n, y, target=target, first_excluded_dose=first_excluded_dose, no_selection=no_selection
    )
    fit = result.isotonic_mean.copy()
    selected, score = _desktop_rank(fit, result.eligible, target)
    return MTPILegacySelection(selected, _owned(fit), _owned(score), result.eligible)


def _desktop_rank(fit, admissible, target):
    score = fit.copy()
    eligible = np.flatnonzero(admissible)
    count = np.isfinite(fit).sum()
    score[1:count] += np.arange(1, count) * 1e-10
    selected = None
    if len(eligible):
        distance = abs(score[eligible] - target)
        # XLL replaces the incumbent at exact equality; R sort chooses first.
        selected = int(eligible[np.flatnonzero(distance == min(distance))[-1]]) + 1
    return selected, score


def replay_mtpi_desktop_trial(study, scenario, uniforms, *, profile="safe"):
    """Replay a zero-based scenario with a supplied [0,1) uniform tape.

    Require enough tape for the worst-case enrollment, including legacy cohort
    overflow. Safe mode clips the last cohort and uses the existing paper APIs.
    Legacy mode preserves native priors, malformed next-dose beta shape, carried
    indicator and overwritten exclusion history. It is not a clinical default.
    Dose labels and unsafe-assignment cohort indices are one-based.
    """
    capacity = _capacity(study, profile)
    scenario = _integer(scenario, "scenario", 0, len(study.scenarios) - 1)
    draw = np.asarray(uniforms, dtype=float)
    if (
        draw.ndim != 1
        or len(draw) < capacity
        or not np.isfinite(draw).all()
        or np.any((draw < 0) | (draw >= 1))
    ):
        raise ValueError("uniform tape must be finite, in [0,1), and cover actual capacity")
    return _replay(study, scenario, draw, profile)


def _replay(study, scenario, draw, profile):
    D = len(study.scenarios[0])
    p = study.scenarios[scenario]
    n = np.zeros(D, dtype=int)
    y = n.copy()
    d = study.starting_dose - 1
    tt = None
    excluded = D
    path, sizes, unsafe = [], [], []
    stop = "capacity"
    mask = np.zeros(D, dtype=bool)
    design = study.design()
    for offset in range(0, study.maximum_patients, study.cohort_size):
        size = (
            study.cohort_size
            if profile == "desktop_legacy"
            else min(study.cohort_size, study.maximum_patients - offset)
        )
        path.append(d + 1)
        sizes.append(size)
        if n[d] > 0 and bool(design.posterior(n[d], y[d]).unsafe):
            unsafe.append(len(path))
        n[d] += size
        y[d] += np.count_nonzero(draw[offset : offset + size] < p[d])
        if profile == "desktop_legacy":
            d, tt, excluded, reason = _legacy_step(study, n, y, d, tt, excluded)
            if reason:
                stop = reason
                break
        else:
            decision = design.next_dose(n, y, d + 1, eliminated=mask)
            mask = decision.eliminated
            if decision.next_dose is None:
                stop = decision.action
                break
            d = decision.next_dose - 1
    if profile == "desktop_legacy":
        selected = select_mtpi_desktop_legacy_mtd(
            n,
            y,
            target=study.target,
            first_excluded_dose=excluded + 1,
            no_selection=stop != "capacity",
        ).dose
    else:
        selected = design.select_mtd(n, y, eliminated=mask).dose
    return MTPIDesktopTrial(
        _owned(n),
        _owned(y),
        tuple(path),
        tuple(sizes),
        selected,
        stop,
        excluded + 1 if profile == "desktop_legacy" else None,
        tt,
        profile,
        tuple(unsafe),
        _owned(mask),
    )
