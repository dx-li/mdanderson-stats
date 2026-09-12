# PLBARPO control monitoring

`plbarpo_control_counts` aggregates dated binary records into a read-only
`(K, 2)` array whose columns are successes and failures, suitable for passing
directly as concurrent `control_counts`. It uses explicit
half-open enrollment windows `[open, close)`. A record is counted only when its
enrollment time is at or before `as_of` and its outcome observation time is at
or before `as_of`; an infinite observation time remains pending. Window
boundaries are a Python library convention because the PLBARPO guide does not
specify boundary behavior. Starts must be finite and a final close may be
positive infinity. The implementation loops over windows and records through
boolean masks and bounds input to 100,000 records and 100 windows.

`plbarpo_control_monitor` updates treatment beta posteriors and compares each
treatment independently with a control posterior using direct beta ordering
probabilities. In `entire` mode one control count pair is shared by every
treatment. In `concurrent` mode each treatment receives its own control count
pair, so `control_counts` has shape `(K, 2)`. Treatment outputs exclude the
control arm and all returned arrays are read-only.

Futility uses `P(treatment <= control) > pfut`; early and final efficacy use
`P(treatment > control) >= peff` and `>= pfinal`. Confidence cutoffs are
optional and inclusive in `[0, 1]`. The final efficacy probability is the same
posterior ordering calculation as early efficacy, with an independent cutoff.
