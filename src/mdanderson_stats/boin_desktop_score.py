"""Enumerated combination scores with explicit native table conventions."""

import hashlib
from dataclasses import dataclass

from scipy.special import betaincc
from scipy.stats import rankdata

from .boin_combination import _interval_probability
from .boin_desktop_controller import desktop_combination_design


@dataclass(frozen=True)
class BOINDesktopScoreRow:
    patients: int
    toxicities: int
    eliminated_from: bool
    interval_probability: float | None
    allocation_score: float | None
    rank: float | None


@dataclass(frozen=True)
class BOINDesktopScoreTable:
    study_sha256: str
    profile: str
    ranking_maximum: int
    rows: tuple[BOINDesktopScoreRow, ...]


def boin_desktop_score_table(study, *, profile="safe"):
    """Rank over the full planned maximum, then display through EarlyStop.

    Safe ranks use the allocation score (interval mass + .0005*n) and strict
    posterior safety cutoff. Native-table compatibility ranks raw interval mass
    and eliminates at equality. Neither table alone replaces the controller's
    admissible set, sticky exclusions, run-in, or lowest-dose stopping rule.
    """
    if profile not in ("safe", "desktop_table_legacy"):
        raise ValueError("profile must be safe or desktop_table_legacy")
    design = desktop_combination_design(study)
    maximum = study.maximum_patients // study.cohort_size * study.cohort_size
    counts = range(0, maximum + 1, study.cohort_size)
    if sum(n + 1 for n in counts) > 20000:
        raise ValueError("score table exceeds 20000 candidate rows")
    legacy = profile == "desktop_table_legacy"
    assert design.early_stop_patients is not None
    entries: list[tuple[int, int, bool, float | None, float | None]] = []
    scores = []
    for n in counts:
        for y in range(n + 1):
            tail = float(betaincc(y + 1, n - y + 1, design.target))
            eliminated = n >= 3 and (
                tail >= design.elimination_probability
                if legacy
                else tail > design.elimination_probability
            )
            if eliminated:
                entries.append((n, y, True, None, None))
                break
            mass = _interval_probability(
                y + 0.5, n - y + 0.5, design.escalation_boundary, design.deescalation_boundary
            )
            allocation = mass + 0.0005 * n
            entries.append((n, y, False, mass, allocation))
            scores.append(mass if legacy else allocation)
    ranks = iter(rankdata(scores, method="average"))
    rows = []
    for n, y, eliminated, row_mass, row_allocation in entries:
        rank = None if eliminated else float(next(ranks))
        if n <= design.early_stop_patients:
            rows.append(BOINDesktopScoreRow(n, y, eliminated, row_mass, row_allocation, rank))
    return BOINDesktopScoreTable(
        hashlib.sha256(study.xml.encode()).hexdigest(), profile, maximum, tuple(rows)
    )
