# TITE-BOIN12 operating-characteristic simulation audit

This simulation wraps the package's AL and BDA calendar replayers. It does not
claim numerical equivalence to the native application or reproduce native
simulation defaults. Each trial draws potential joint binary outcomes by dose
from caller-supplied four-cell probabilities in the order
`(no toxicity/efficacy, no toxicity/no efficacy, toxicity/efficacy,
toxicity/no efficacy)`. Only outcomes at the assigned dose are revealed to the
calendar replay.

For event-positive endpoints, event delay is sampled independently and
uniformly on that endpoint's assessment window, conditional on the binary event.
No-event endpoints have delay `+inf`. This timing choice is explicit because
joint binary truth alone does not specify event-time distributions. Fixed
accrual uses a gap of `1/accrual_rate` before every patient, including the
first; exponential accrual samples every gap independently with that mean.
Endpoint assessment and event ties are processed by the calendar replayer.

The optional Gumbel constructor delegates to the existing U-BOIN Gumbel model
and reorders its `(efficacy, toxicity)` cells to the TITE-BOIN12 four-cell
convention. The association parameter is required explicitly; it is not
inferred from the marginal probabilities.

The simulator uses separate deterministic streams for arrivals,
outcome/timing, and BDA sampling. Results retain only trial-level dose counts,
selection, stop reason, accrual-stop/final times, and the three seeds; histories
and posterior draws are discarded after each serial trial. Means and MCSEs for
dose counts and times use independent trials as the sampling units. Selection
probabilities use binomial MCSEs. Work and retained/scratch storage are bounded
before constructing random streams. In exponential mode, an extraordinarily
large sampled gap can still make calendar arithmetic unrepresentable; the
calendar replay raises an error rather than silently changing the clock.

The reported final time is the latest event or assessment ascertainment among
enrolled patients, bounded below by accrual-stop time. Thus an observed event
can complete follow-up before its endpoint window ends. A safety termination
returns no selected OBD, while the calendar replay may retain its separate
complete-data selection diagnostics.

## Integrated validation

The final seven focused simulation checks pass with warnings treated as errors,
including early safety-stop aggregation and undefined single-trial MCSEs.
This final run took 1.91 seconds, peaked at 148.27 MiB and reported zero swaps.
The worker's earlier simulation/calendar run also passed 13 focused checks.
Validation is serial under single-thread numerical-library settings; no broad
local suite or new CI job is introduced for this workflow.
