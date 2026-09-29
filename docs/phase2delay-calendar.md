# Phase2Delay: scheduled trial simulation

The calendar workflow connects the [delayed-outcome monitor](phase2delay.md)
to enrollment, evolving follow-up and repeated decisions. It also generates
trial tapes and summarizes operating characteristics across serial simulations.
The model follows [Cai, Liu and Yuan (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4435968/).
The analysis schedule and final-follow-up policy are explicit Python protocol
choices; this is not a reconstruction of the hidden Shiny controller.

## Replay a trial

Arrival times are absolute times from the trial origin. Latent event times are
delays from each patient's arrival; positive infinity represents no event
within the assessment window. Future arrivals and unobserved events never
enter the current posterior. Arrivals and due events at a scheduled analysis
are processed before the decision. Completion and event comparisons use their
absolute calendar timestamps, so ordinary rounding of non-binary times does
not turn a completed window back into a pending one. Positive durations that
collapse to their arrival time are rejected as unrepresentable.

```python
import numpy as np
from mdanderson_stats import replay_phase2_delay_calendar

model = dict(
    endpoint="response", threshold=0.3, cutoff=0.95,
    prior_alpha=0.1, prior_beta=0.2,
    intervals=2, hazard_c=0.01, lambda0=0.1,
    burn_in=100, hazard_draws=400,
)
trial = replay_phase2_delay_calendar(
    arrival_times=[0, 1, 2, 3, 4, 5],
    latent_event_times=[0.5, np.inf, 2, np.inf, 0.25, np.inf],
    analysis_times=[1, 3, 5, 7], final_analysis="complete_window",
    window=3, minimum_completed=5, random_state=141,
    **model,
)
assert trial.looks[0].status == "gate_not_reached"
assert trial.looks[0].observed_events == 1
assert trial.looks[0].completed == 0
assert trial.planned_terminal_time == 8
```

The paper begins monitoring after five patients have completed their **full
assessment periods**. An early response reveals a binary outcome but does not
satisfy that elapsed-follow-up gate. `minimum_completed` controls this rule.
A skipped look has status `gate_not_reached` and undefined posterior summaries;
it is not evidence that the treatment meets the efficacy target.

`final_analysis="at_enrollment_cap"` inserts a final analysis at the last
planned arrival. `"complete_window"` inserts it one assessment window later.
The schedule is truncated at that terminal time and the terminal analysis
appears once. Earlier futility or safety stopping prevents future enrollment.
`stop_time` records that decision, while `latest_enrolled_completion_time`
records when the patients already enrolled would all complete their windows.
The latter does not trigger extra decisions after stopping.

For a trial that does not stop, `completed_at_stop` means the number of full
windows completed at the terminal decision. `completed_at_latest_followup`
describes full follow-up of enrolled patients. A final `continue` decision
means that no stopping boundary was crossed; it is not a separate efficacy
claim.

## Generate trials and estimate operating characteristics

```python
from mdanderson_stats import (
    simulate_phase2_delay_calendar,
    simulate_phase2_delay_calendar_oc,
)

scenario = dict(
    event_probability=0.3, late_fraction=0.7,
    accrual_rate=2, max_subjects=8,
    analysis_times=[3, 5, 7, 9], final_analysis="complete_window",
    window=3, minimum_completed=5, **model,
)
one = simulate_phase2_delay_calendar(**scenario, random_state=2026)
repeated = simulate_phase2_delay_calendar(**scenario, random_state=one.seed)
np.testing.assert_array_equal(one.arrival_times, repeated.arrival_times)
np.testing.assert_array_equal(one.latent_event_times, repeated.latent_event_times)
summary = simulate_phase2_delay_calendar_oc(**scenario, trials=4, random_state=141)
assert len(summary.trial_seeds) == 4
assert summary.futility_stops + summary.safety_stops + summary.continue_trials == 4
```

The first simulated patient arrives at time zero; subsequent exponential
interarrival times implement the specified Poisson accrual rate. Weibull event
times satisfy `P(event by T) = event_probability` and the conditional fraction
of events in `(T/2, T]` equals `late_fraction`. Both probabilities must be
strictly between zero and one. Out-of-window events are stored as infinity.
The paper studies late fractions of 0.7 and 0.9.

Each returned trial seed reproduces its data tape and scheduled analyses when
passed to `simulate_phase2_delay_calendar`. Per-look seeds reproduce the
monitoring draws. The operating-characteristic result includes stopping rate,
enrollment, full-window completion at decision, and trial duration, with
Monte Carlo standard errors. Duration ends at the stopping decision or planned
terminal analysis, not at post-stop follow-up completion. A single trial cannot
estimate a Monte Carlo standard error. The four-trial example illustrates the
API; it is too small for a design evaluation.

The wrapper processes trials and analyses sequentially and retains compact
summaries instead of posterior traces for every look. Bounds on subjects,
looks, sampler work and retained summaries are checked before simulation.
`max_work` bounds conservative total sampling work across the requested batch.

## Assumptions and independent checks

The paper describes continuous monitoring. Posterior probabilities can change
as pending patients accrue follow-up even between arrivals and events. This
API therefore requires an explicit discrete analysis schedule and does not
claim exact continuous-monitoring equivalence. Schedule choice can affect the
operating characteristics.

Priors, `lambda0`, and sampler controls remain explicit. The paper's statement
about setting the prior response probability does not uniquely identify a
prior-predictive calibration for the Gamma-martingale model. The examples use
illustrative numerical settings, not recovered Shiny defaults. Increase and
assess sampling effort for substantive design work; estimated Monte Carlo
error alone does not establish convergence. When converting time units, scale
all times and the window together and divide both the accrual rate and
`lambda0` by the conversion factor.

[The independent base-R reference](../tools/reference_phase2delay_calendar.R)
checks eight Weibull calibrations, 48 inverse-CDF values and a sevenfold time
conversion. Calendar checks use explicit patient histories to verify the
full-window gate, simultaneous events/arrivals, terminal handling and stopping
before future enrollment. The underlying monitor retains its separate
independent hazard-integration references.

Native calibration, unpublished continuous-monitoring details, upload formats
and reports remain open; catalog entry 141 remains partial.
