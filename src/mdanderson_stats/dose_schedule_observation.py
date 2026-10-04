"""As-of adjudication of delayed dose-schedule toxicity episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from .dose_schedule import DoseSchedulePatient, _numeric

_MAX_OBSERVATION_RECORDS = 10_000


def _time(value: object, name: str, *, positive: bool = False) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite nonnegative time")
    result = float(raw)
    if not isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _freeze_indices(values: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.frombuffer(np.ascontiguousarray(values, dtype=np.int64).tobytes(), dtype=np.int64)


@dataclass(frozen=True)
class DoseScheduleToxicityEpisode:
    """A toxicity onset and qualification status at an adjudication time.

    ``qualifies`` and ``adjudication_time`` are both ``None`` while the
    episode is unresolved. This value records a caller's adjudication; it does
    not infer toxicity grade, persistence, or dose-reduction causality.
    """

    onset_time: float
    adjudication_time: float | None
    qualifies: bool | None

    def __init__(
        self,
        onset_time: float,
        adjudication_time: float | None = None,
        qualifies: bool | None = None,
    ) -> None:
        onset = _time(onset_time, "onset_time")
        if (adjudication_time is None) != (qualifies is None):
            raise ValueError("adjudication_time and qualifies must be supplied together")
        if adjudication_time is None:
            adjudicated_at = None
            qualification = None
        else:
            adjudicated_at = _time(adjudication_time, "adjudication_time")
            if adjudicated_at < onset:
                raise ValueError("adjudication_time cannot precede onset_time")
            if not isinstance(qualifies, (bool, np.bool_)):
                raise ValueError("qualifies must be boolean when adjudicated")
            qualification = bool(qualifies)
        object.__setattr__(self, "onset_time", onset)
        object.__setattr__(self, "adjudication_time", adjudicated_at)
        object.__setattr__(self, "qualifies", qualification)


def adjudicate_dose_schedule_grade2(
    onset_time: float,
    *,
    as_of: float,
    resolution_time: float | None = None,
    dose_reduction_time: float | None = None,
    day_length: float = 1.0,
) -> DoseScheduleToxicityEpisode:
    """Apply the paper's 14-day grade-2 persistence/dose-reduction rule.

    Times use a common origin and unit; ``day_length`` converts one day to
    that unit. A dose reduction must be explicitly attributed to this episode
    by the caller. Future resolution and reduction times are ignored until
    ``as_of`` reaches them. The onset must already be known at ``as_of``.

    An episode qualifies when an attributable dose reduction is known, or
    when it remains unresolved at the end of the 14-day window. A qualifying
    episode is recorded at its onset, while ``adjudication_time`` is the
    earliest trigger known as of the snapshot. Resolution by the deadline
    without a known attributable reduction is nonqualifying. Recreate the
    episode for each new ``as_of`` snapshot: a later dose reduction can change
    an earlier nonqualifying result.
    """
    onset = _time(onset_time, "onset_time")
    observation_time = _time(as_of, "as_of")
    unit = _time(day_length, "day_length", positive=True)
    if onset > observation_time:
        raise ValueError("onset_time must be known at as_of")
    resolution = None if resolution_time is None else _time(resolution_time, "resolution_time")
    reduction = (
        None if dose_reduction_time is None else _time(dose_reduction_time, "dose_reduction_time")
    )
    if resolution is not None and resolution < onset:
        raise ValueError("resolution_time cannot precede onset_time")
    if reduction is not None and reduction < onset:
        raise ValueError("dose_reduction_time cannot precede onset_time")

    maximum = np.finfo(float).max
    if unit > maximum / 14.0:
        raise ValueError("14-day adjudication window exceeds floating-point range")
    delay = 14.0 * unit
    if not isfinite(delay):
        raise ValueError("14-day adjudication window exceeds floating-point range")
    if onset > maximum - delay:
        raise ValueError("14-day adjudication deadline exceeds floating-point range")
    deadline = onset + delay
    if not isfinite(deadline):
        raise ValueError("14-day adjudication deadline exceeds floating-point range")
    if deadline <= onset:
        raise ValueError("14-day adjudication deadline is below floating-point resolution")

    known_reduction = reduction if reduction is not None and reduction <= observation_time else None
    known_resolution = (
        resolution if resolution is not None and resolution <= observation_time else None
    )
    triggers: list[float] = []
    if known_reduction is not None:
        triggers.append(known_reduction)
    unresolved_at_deadline = observation_time >= deadline and (
        known_resolution is None or known_resolution > deadline
    )
    if unresolved_at_deadline:
        triggers.append(deadline)
    if triggers:
        return DoseScheduleToxicityEpisode(onset, min(triggers), True)
    if known_resolution is not None and known_resolution <= deadline:
        return DoseScheduleToxicityEpisode(onset, known_resolution, False)
    return DoseScheduleToxicityEpisode(onset)


@dataclass(frozen=True)
class DoseScheduleEpisodeStatus:
    """Episode information visible at one observation time."""

    episode_index: int
    onset_time: float
    adjudicated: bool
    adjudication_time: float | None
    qualifies: bool | None


@dataclass(frozen=True)
class DoseSchedulePatientObservation:
    """As-of model record, visible episode statuses, and delivered-dose ledger."""

    patient: DoseSchedulePatient
    as_of: float
    horizon: float
    observation_end: float
    qualifying_episode_index: int | None
    episode_statuses: tuple[DoseScheduleEpisodeStatus, ...]
    pending_episode_indices: tuple[int, ...]
    final_ready: bool
    delivered_administration_times: NDArray[np.float64]
    delivered_dose_indices: NDArray[np.int64]


def observe_dose_schedule_patient(
    episodes: Sequence[DoseScheduleToxicityEpisode],
    *,
    as_of: float,
    horizon: float,
    administration_times: ArrayLike,
    dose_indices: ArrayLike,
) -> DoseSchedulePatientObservation:
    """Build a likelihood-ready patient record from as-of toxicity information.

    A known qualifying onset at or before ``horizon`` is the event time, even
    when its adjudication occurs later. Administrations strictly before that
    onset enter the event likelihood, matching the event-first tie convention
    of the calendar replay. The separate delivered-dose ledger retains every
    administration at or before ``as_of``, including post-onset treatment.

    Only episodes whose onset is at or before ``as_of`` appear in the returned
    statuses. Future adjudication dates and decisions are masked until they
    are known. With no known event, the likelihood record is censored at the
    earlier of ``as_of`` and ``horizon``. ``final_ready`` is false while an
    unresolved episode could move the event earlier, or while risk-horizon
    follow-up/adjudication is incomplete. Pending episodes after the horizon
    remain visible but do not delay a risk-horizon final analysis.
    """
    observation_time = _time(as_of, "as_of")
    horizon_time = _time(horizon, "horizon", positive=True)
    if not isinstance(episodes, (tuple, list)) or len(episodes) > _MAX_OBSERVATION_RECORDS:
        raise ValueError(
            f"episodes must be a bounded sequence of at most {_MAX_OBSERVATION_RECORDS}"
        )
    if any(not isinstance(episode, DoseScheduleToxicityEpisode) for episode in episodes):
        raise ValueError("episodes must contain DoseScheduleToxicityEpisode values")

    times = _numeric(
        administration_times, "administration_times", max_size=_MAX_OBSERVATION_RECORDS
    )
    raw_doses = np.asarray(dose_indices)
    if (
        times.ndim != 1
        or raw_doses.ndim != 1
        or times.shape != raw_doses.shape
        or raw_doses.dtype.kind not in "iu"
        or raw_doses.dtype.kind == "b"
    ):
        raise ValueError("administration_times and integer dose_indices must be matching vectors")
    if np.any(times < 0) or np.any(times[1:] < times[:-1]):
        raise ValueError("administration_times must be nonnegative and nondecreasing")
    if np.any(raw_doses < 0) or np.any(raw_doses >= 20):
        raise ValueError("dose_indices must lie in [0,19]")
    doses = np.asarray(raw_doses, dtype=np.int64)

    statuses: list[DoseScheduleEpisodeStatus] = []
    pending: list[int] = []
    qualifying: list[tuple[float, int]] = []
    for index, episode in enumerate(episodes):
        if episode.onset_time > observation_time:
            continue
        adjudicated = (
            episode.adjudication_time is not None and episode.adjudication_time <= observation_time
        )
        status = DoseScheduleEpisodeStatus(
            episode_index=index,
            onset_time=episode.onset_time,
            adjudicated=adjudicated,
            adjudication_time=episode.adjudication_time if adjudicated else None,
            qualifies=episode.qualifies if adjudicated else None,
        )
        statuses.append(status)
        if not adjudicated:
            pending.append(index)
        elif episode.qualifies and episode.onset_time <= horizon_time:
            qualifying.append((episode.onset_time, index))

    qualifying_index: int | None = None
    qualifying_onset: float | None = None
    if qualifying:
        qualifying_onset, qualifying_index = min(qualifying)
        observation_end = qualifying_onset
        event = True
        final_ready = not any(episodes[index].onset_time < qualifying_onset for index in pending)
    else:
        observation_end = min(observation_time, horizon_time)
        event = False
        final_ready = observation_time >= horizon_time and not any(
            episodes[index].onset_time <= horizon_time for index in pending
        )

    delivered_mask = times <= observation_time
    delivered_times = times[delivered_mask]
    delivered_doses = doses[delivered_mask]
    if event:
        fit_mask = delivered_times < observation_end
    else:
        fit_mask = delivered_times <= observation_end
    patient = DoseSchedulePatient(
        observation_end,
        event,
        delivered_times[fit_mask],
        delivered_doses[fit_mask],
    )
    return DoseSchedulePatientObservation(
        patient=patient,
        as_of=observation_time,
        horizon=horizon_time,
        observation_end=observation_end,
        qualifying_episode_index=qualifying_index,
        episode_statuses=tuple(statuses),
        pending_episode_indices=tuple(pending),
        final_ready=bool(final_ready),
        delivered_administration_times=_freeze(delivered_times),
        delivered_dose_indices=_freeze_indices(delivered_doses),
    )
