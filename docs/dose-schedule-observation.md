# Dose Schedule Finder delayed-toxicity observations

The helper `observe_dose_schedule_patient` turns caller-adjudicated toxicity
episodes and actual administration records into an as-of `DoseSchedulePatient`
for the existing likelihood and posterior fit. All episode, administration,
follow-up, horizon and adjudication times use one common relative time origin
per patient.

The paper's example treats a grade-2 toxicity that is not therapeutically
resolved within two weeks of onset, or one that requires dose reduction, as a
dose-limiting toxicity at its initial onset. The helper does not decide
persistence, dose reduction, or when adjudication occurs. Supply
`qualifies=True` only once the caller's study rule determines that the episode
qualifies. Set both `adjudication_time` and `qualifies` to `None` while the
episode remains unresolved.

```python
import numpy as np
from mdanderson_stats import (
    DoseSchedulePrior,
    DoseScheduleToxicityEpisode,
    fit_dose_schedule,
    observe_dose_schedule_patient,
)

# At day 24, the caller has determined that dose reduction is required.
episode = DoseScheduleToxicityEpisode(onset_time=10, adjudication_time=24, qualifies=True)
interim = observe_dose_schedule_patient(
    [episode],
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

prior = DoseSchedulePrior(
    [-2.0, np.log(2.0), np.log(10.0)], [0.3, 0.0, 0.0], dose_count=1
)
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

The helper does not generate low-grade episodes, infer the clinical
adjudication, or implement within-patient dose modification. It provides a
reproducible information boundary for callers who already have onset,
adjudication, and actual administration histories.
