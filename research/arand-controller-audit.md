# Adaptive Randomization controller: source gaps

The local version 5.2 guide (`research/raw/ARAND/UsersGuide.pdf`) was inspected
again on September 28, 2026. Its existence and the recovered native binary do
not establish a complete controller contract. The existing posterior kernels
are described in [arand.md](../docs/arand.md).

The guide specifies these rules:

- Page 6, section 3.1: maximum enrollment and/or duration, and minimum
  enrollment before trial stopping.
- Page 7, section 3.2: allocation proportional to the best-arm probability
  raised to a nonnegative tuning power. During initial equal randomization,
  best-arm probabilities are not computed until the specified count is reached.
- Pages 7–8, section 3.3: strict lower best-arm cutoff for reversible loser
  suspension; strict upper cutoffs for early and final winner selection.
  There is no default final winner if the cutoff is not met.
- Page 8, section 3.3(iv): when maximizing only, a strict lower cutoff on
  the posterior probability of exceeding a fixed parameter threshold makes
  futility permanent.
- Page 8, section 3.4: final follow-up contributes to completed-trial duration,
  but not early-stopped-trial duration.
- Pages 9–10: exponential arrival gaps, per-arm early/final winner and drop
  probabilities, and treated-patient means and central simulation intervals.

The guide does not give the allocation-floor transformation, explicitly settle
ranking after permanent arm removal, order simultaneous triggers, or resolve
a hard duration cap arriving before minimum enrollment. Reversible loser
reactivation suggests continued comparison with suspended arms, but that is an
inference rather than a separately verified native algorithm.

The calendar controller is implemented in
[`arand_calendar.py`](../src/mdanderson_stats/arand_calendar.py) as deterministic
replay over supplied arrival, assignment, and potential-outcome tapes. Native
rules are applied with strict cutoffs and a minimum-enrollment gate. Binary
outcomes become visible at their supplied assessment delays; exponential event
delays are relative to assignment and contribute event/censoring exposure only
as of each look. Completed-trial duration includes the configured final
follow-up, while early-stopped duration ends at the stop time.

Unspecified behavior remains an explicit `ArandControllerPolicy`: allocation
floor transform, comparison scope after permanent removal, trigger order,
duration-versus-minimum precedence, multiple-winner tie handling, empty-active
behavior, and same-time arrival/look order. Temporary suspensions remain
comparable so they can reverse. For final selection, suspended arms remain
eligible, while permanently futile arms are excluded; this follows the guide's
final-winner rule and is kept separate from active allocation eligibility.
Tied arrivals follow tape order, and an arrival at the duration boundary is
blocked under `duration_wins`. These are documented Python conventions, not
claims about native defaults. The simulator and operating-characteristic
workflow are not part of this replay implementation.
