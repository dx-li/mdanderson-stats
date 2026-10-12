"""Unrounded desktop pending decisions and explicit native-table compatibility."""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import betaincc

from .boin_desktop import _number
from .boin_desktop_single import desktop_single_design
from .boin_desktop_tite import _nodes, _prior
from .tite_keyboard import toxicity_followup_weights


@dataclass(frozen=True)
class BOINDesktopPendingRule:
    patients: int
    toxicities: int
    pending: int
    action: str | None
    escalate_at_least: float | None
    deescalate_at_most: float | None
    lowest_stop: bool
    safety_eliminated: bool
    profile: str

    def decide(self, standardized_followup: float) -> str:
        """Apply inclusive thresholds to a valid standardized follow-up total."""
        if isinstance(standardized_followup, (bool, np.bool_)):
            raise ValueError("follow-up must be a real number")
        value = float(standardized_followup)
        if not math.isfinite(value) or not 0 <= value <= self.pending:
            raise ValueError("follow-up must be finite and between zero and pending count")
        if self.action is not None:
            return self.action
        if self.escalate_at_least is not None and value >= self.escalate_at_least:
            return "escalate"
        if self.deescalate_at_most is not None and value <= self.deescalate_at_most:
            return "deescalate"
        return "stay"


@dataclass(frozen=True)
class BOINDesktopPendingTable:
    study_sha256: str
    profile: str
    weighted: bool
    rows: tuple[BOINDesktopPendingRule, ...]


def _settings(study, profile):
    if profile not in ("safe", "desktop_table_legacy"):
        raise ValueError("profile must be safe or desktop_table_legacy")
    _, node = _nodes(study)
    design = desktop_single_design(study)
    assert design.early_stop_patients is not None
    limit = _number(node, "MaxPendingOutcomePercent") / 100
    if not 0 <= limit <= 0.65:
        raise ValueError("pending fraction must lie in [0,.65]")
    prior = _prior(node)
    if profile == "safe":
        toxicity_followup_weights([], 1, trimester_probabilities=prior)
    elif design.early_stop_patients <= 6:
        raise ValueError("native pending table requires EarlyStop greater than six")
    if (
        profile == "desktop_table_legacy"
        and design.extra_safe
        and betaincc(4, 1, design.target) <= design.elimination_probability - design.safety_offset
    ):
        raise ValueError("native extra-safe table has an undefined stopping flag at n=3")
    return design, limit, prior is not None


def _rule(design, limit, n, y, pending, profile):
    probability = float(betaincc(y + 1, n - y + 1, design.target))
    unsafe = n >= 3 and probability > design.elimination_probability
    lowest_stop = n >= 3 and probability > (
        design.elimination_probability - design.safety_offset
        if design.extra_safe
        else design.elimination_probability
    )
    action = None
    up = down = None
    raw = y / n
    le, ld = design.escalation_boundary, design.deescalation_boundary
    legacy = profile == "desktop_table_legacy"
    if legacy:
        # The table's inclusive integer count represents a strict rate boundary.
        dlt_cutoff = round(ld * n) + 1 if abs(round(ld * n) - ld * n) < 1e-12 else math.ceil(ld * n)
        eliminate = (
            next(
                (
                    v
                    for v in range(1, n + 1)
                    if betaincc(v + 1, n - v + 1, design.target) > design.elimination_probability
                ),
                n,
            )
            if n >= 3
            else None
        )
        if eliminate is not None:
            dlt_cutoff = min(dlt_cutoff, eliminate)
        if y >= dlt_cutoff:
            action = "eliminate" if eliminate is not None and y >= eliminate else "deescalate"
    elif unsafe:
        action = "eliminate"
    elif raw >= ld:
        action = "stay" if design.stay_at_one_of_three and n == 3 and y == 1 else "deescalate"
    if action is None and pending > n * limit:
        action = "suspend"
    if action is None:
        posterior = (y + design.target / 2) / (n - pending + 1) if pending else raw
        if posterior == 0:
            action = "escalate"
        else:
            up = pending - (1 - posterior) / posterior * (le * n - y)
            down = pending - (1 - posterior) / posterior * (ld * n - y)
            if legacy:
                up, down = round(up, 2), round(down, 2)
                up = None if up >= pending else max(0.0, up)
                down = None if down <= 0 else down
            else:
                up = None if up > pending else max(0.0, up)
                down = None if down < 0 else min(float(pending), down)
            if raw > design.target:
                up = None
            if raw < design.target:
                down = None
            if up is not None and up <= 0:
                action = "escalate"
            elif down is not None and down >= pending:
                action = "deescalate"
            elif up is None and down is None:
                action = "stay"
    if action is not None:
        up = down = None
    return BOINDesktopPendingRule(n, y, pending, action, up, down, lowest_stop, unsafe, profile)


def boin_desktop_pending_rule(study, patients, toxicities, pending, *, profile="safe"):
    """Return a current-dose rule; lowest-dose stopping is a separate precedence flag.

    Safe thresholds are unrounded and mirror observed-data calendar priorities,
    including active run-in. The native table profile uses two-decimal cutoffs,
    strict integer de-escalation and no run-in. It is not a simulation profile.
    """
    design, limit, _ = _settings(study, profile)
    values = (patients, toxicities, pending)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
        raise ValueError("counts must be integers")
    n, y, m = map(int, values)
    if not 1 <= n <= 1000 or min(y, m) < 0 or y + m > n:
        raise ValueError("require 1..1000 patients with DLT plus pending no greater than patients")
    return _rule(design, limit, n, y, m, profile)


def boin_desktop_pending_table(study, *, profile="safe") -> BOINDesktopPendingTable:
    """Enumerate cohort-boundary states up to min(planned maximum, precision cap).

    At most20,000 rows are retained; larger tables fail before allocation. Use
    the single-state rule API for a particular larger count. Informative WSTFT
    uses the saved analysis probabilities, not the simulation event distribution.
    """
    design, limit, weighted = _settings(study, profile)
    if study.maximum_patients % study.cohort_size:
        raise ValueError("planned maximum must contain whole cohorts")
    ceiling = min(study.maximum_patients, design.early_stop_patients)
    if ceiling < study.cohort_size:
        raise ValueError("precision cap is below one cohort; use the single-state rule API")
    sizes = range(study.cohort_size, ceiling + 1, study.cohort_size)
    if sum((n + 1) * (n + 2) // 2 for n in sizes) > 20000:
        raise ValueError("pending table exceeds20000 rows; use the single-state rule API")
    rows = tuple(
        _rule(design, limit, n, y, m, profile)
        for n in sizes
        for y in range(n + 1)
        for m in range(n - y + 1)
    )
    return BOINDesktopPendingTable(
        hashlib.sha256(study.xml.encode()).hexdigest(), profile, weighted, rows
    )
