# TOP with two endpoints

`TOPMultiEndpointDesign` adds co-primary efficacy and efficacy/toxicity monitoring
to [binary TOP](top-binary.md). It follows the published Supplementary Methods
and decision-combination Tables 3 and 6. The [source audit](../research/top-endpoints-audit.md)
records independent numerical references and the inspected supplement.

## Joint prior and marginal monitoring

Supply `null_joint_probabilities` in the order `(1,1), (1,0), (0,1), (0,0)`.
The first coordinate is efficacy; the second is either another efficacy
endpoint or toxicity. The four probabilities must be positive and sum to one.
Multiplication by `prior_concentration`, default one, gives a Dirichlet prior.
Its aggregated endpoint probabilities define the null thresholds and the two
marginal Beta priors.

Each endpoint has its own observed event count and effective sample size:
`ESS = enrolled - pending + sum(pending weights)`. Its posterior approximation
adds observed events to the prior alpha and `ESS-events` to the prior beta.
The two margins can have different pending patients and assessment windows.
This is the published fractional-information approximation, not exact inference
from the pending-outcome likelihood. Different endpoint ESS values do not
define a joint fractional Dirichlet posterior. Changing the joint prior while
preserving its margins leaves these monitoring probabilities unchanged.

The returned `acceptable_probability` is the upper posterior tail above the
efficacy threshold, or the lower posterior tail below the toxicity threshold.
An endpoint is unacceptable when this probability is strictly below
`C*(n/N)**gamma`. Equivalently, its adverse probability exceeds the paper's
`1-C*(n/N)**gamma`. Equality is acceptable. C and gamma are supplied design
parameters; the constructor does not calibrate their operating characteristics.

## Combining endpoint decisions

With `mode="coprimary"`, one acceptable endpoint permits continuation; both
must be futile to stop. With `mode="efficacy_toxicity"`, either futility or
excess toxicity stops the trial; both endpoints must be acceptable to continue.
Remaining unresolved combinations suspend enrollment.
Combined actions are `continue`, `success`, `suspend`, `stop_futility`,
`stop_toxicity`, or `stop_futility_toxicity`; the last two apply only to the
efficacy/toxicity mode. `success` requires a final look.

Suspension is decided separately for each endpoint before those rules are
combined. `suspension="table"` uses `pending >= ceil(n*n/N)` at interim looks;
`"strict"` uses `pending > n*n/N`. Efficacy suspension is waived when the
observed event count already meets its complete-data go threshold. Toxicity
suspension is waived when that count already establishes excess toxicity.
At the final look, an endpoint with any pending outcome is suspended. Applying
the published combination rule can therefore give co-primary success despite
pending outcomes on the other endpoint, or stop an efficacy/toxicity trial
despite unresolved outcomes on its other endpoint.

```python
import numpy as np
from mdanderson_stats import TOPMultiEndpointDesign

design = TOPMultiEndpointDesign(
    45, [.15, .30, .15, .40], .94, .50,
    mode="coprimary", looks=[15, 30, 45], windows=[2, 4],
)
table = design.boundaries()
np.testing.assert_array_equal(table.complete_event_threshold, [[7, 5], [16, 12], [26, 19]])
np.testing.assert_allclose(table.effective_size_crossing[0, 0, 5], 10.64809344313821)
assert design.evaluate(15, [5, 4], [4, 4], [.5, .5]).decision == "continue"
assert design.evaluate(15, [5, 4], [4, 4], [2, 2]).decision == "stop_futility"
```

## Follow-up and timing weights

`evaluate` accepts enrolled counts plus event, pending and summed-weight arrays
whose final axis contains the two endpoints. Leading axes allow batches of
independent interim states. `evaluate_followup` instead accepts arrays shaped
`(..., patients, 2)`: outcomes are 0, 1, or NaN for pending, and follow-up is
measured in the same units as the corresponding `windows` entry. Callers supply
the binary endpoint status; this function does not convert survival records
into landmark outcomes. Only pending follow-up contributes to the ESS.

Uniform conditional event timing is the default. `timing_probabilities` can
specify three probabilities for the successive thirds of each assessment
window, either a shared triple or a `(2,3)` array. `timing_weight` evaluates
the resulting piecewise-linear conditional CDF. The probabilities are
elicited timing assumptions, not a fitted distribution. Equal thirds recover
ordinary follow-up/window weights.

Boundary tables retain complete-data event thresholds, suspension counts, and
unrounded ESS crossings for each scheduled look and endpoint. For efficacy,
the complete threshold is the smallest acceptable event count; for toxicity,
it is the smallest unacceptable DLT count. A NaN crossing means no root in
the feasible ESS range `[events,enrolled]`, or an event count exceeding enrollment;
use `evaluate` for the actual action. A complete threshold of `enrolled+1`
means no feasible count meets that endpoint's boundary.

Designs support up to 200 patients; summary and patient-level evaluations have
a two-million-cell limit checked before allocating broadcast results.

[Calendar replay and simulation](top-endpoints-simulation.md) apply these rules
to patient arrivals and delayed, jointly generated endpoint outcomes. They report
success probabilities, Monte Carlo uncertainty, enrollment and duration.

[Finite-grid calibration](top-endpoints-calibration.md) searches explicit joint
null scenarios and one alternative with common random numbers and an independent
holdout. It does not claim error control outside the supplied scenarios. Entry
134 remains partial: native optimizer grids, reports, and app-version parity
remain unimplemented.
