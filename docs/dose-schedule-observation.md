# Dose Schedule Finder delayed-toxicity observations

The helper `observe_dose_schedule_patient` turns caller-adjudicated toxicity
episodes and actual administration records into an as-of `DoseSchedulePatient`
for the existing likelihood and posterior fit. All episode, administration,
follow-up, horizon and adjudication times use one common relative time origin
per patient.

The paper's example treats a grade-2 toxicity that cannot be therapeutically
resolved within two weeks of onset, or one that necessitates dose reduction,
as a dose-limiting toxicity at its initial onset. Use
`adjudicate_dose_schedule_grade2` to apply this source rule to an explicit
as-of snapshot. A dose-reduction time must be explicitly attributed to the
episode by the caller; no attribution is inferred from the administration
history.

The function's timing convention is a Python choice, not established native
parity. It treats resolution known at or before the 14-day deadline as
nonqualifying, and an episode still unresolved at the deadline as qualifying.
The paper says “resolved by day 24” for an onset at day 10, but does not settle
automatic adjudication of observations exactly at the 14-day boundary. The
caller supplies the time unit through `day_length` (default 1 means one unit
per day). Recreate the episode at every new `as_of`; later information can
change an earlier nonqualifying snapshot. A future onset is rejected, and
future resolution/reduction times do not affect the current result.

```python
import numpy as np
from mdanderson_stats import (
    DoseSchedulePrior,
    adjudicate_dose_schedule_grade2,
    fit_dose_schedule,
    observe_dose_schedule_patient,
)

# Day 10 onset, unresolved at day 23: the two-week rule is still pending.
pending = adjudicate_dose_schedule_grade2(onset_time=10, as_of=23)
assert pending.qualifies is None

# By day 24 it remains unresolved, so it qualifies and is scored at day 10.
episode = adjudicate_dose_schedule_grade2(onset_time=10, as_of=24)
assert episode.qualifies is True and episode.adjudication_time == 24
interim = observe_dose_schedule_patient(
    [pending],
    as_of=23,
    horizon=30,
    administration_times=[2, 10, 11],
    dose_indices=[0, 0, 1],
)
assert not interim.patient.event  # day-24 decision is not visible on day 23
assert interim.pending_episode_indices == (0,)

adjudicated = observe_dose_schedule_patient(
    [episode],
    as_of=24,
    horizon=30,
    administration_times=[2, 10, 11, 24],
    dose_indices=[0, 0, 1, 1],
)
assert adjudicated.patient.event and adjudicated.patient.time == 10
assert adjudicated.patient.administration_times.tolist() == [2]
assert adjudicated.delivered_administration_times.tolist() == [2, 10, 11, 24]

prior = DoseSchedulePrior([-2.0, np.log(2.0), np.log(10.0)], [0.3, 0.0, 0.0], dose_count=1)
fit = fit_dose_schedule(
    [adjudicated.patient],
    prior,
    [[0.0], [0.0, 2.0]],
    horizon=30,
    draws=8,
    warmup=0,
    chains=2,
    rng=np.random.default_rng(412),
)
```

At day 23, the returned episode status masks both the future adjudication time
and its decision. The onset is visible because it has occurred, but it remains
pending. At day 24 the caller's adjudication is known, and the likelihood
record uses day 10 as the event time. An administration exactly at day 10 is excluded
from the event likelihood to match the event-first calendar tie convention;
all administrations delivered through the as-of time remain in the separate
actual-history arrays.

`final_ready` is advisory. A likelihood-ready interim patient record can be
used even when final adjudication is not ready. With no known qualifying event,
final readiness requires full horizon follow-up and resolution of every
episode whose onset is by the horizon. A known event can be final earlier if
no pending earlier onset could move it back. Episodes after the horizon remain
visible when observed but do not delay risk-horizon finality.

The helper does not generate low-grade episodes, infer whether a dose
reduction was caused by the episode, or implement within-patient dose
modification. It provides reproducible adjudication from supplied onset,
resolution, reduction and as-of times. The paper does not specify a low-grade
incidence or resolution-time distribution for simulating these episodes.
