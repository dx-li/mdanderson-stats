# TOP delayed binary-response monitoring

`TOPBinaryDesign` implements the binary efficacy portion of
[TOP](https://biostatistics.mdanderson.org/shinyapps/TOP2/), catalog entry 134.
The public app reports version 1.0.6.0, updated 01/05/2026. The primary reference
is Lin, Coleman and Yuan, *JNCI* 112:38–45 (2020),
[doi:10.1093/jnci/djz049](https://doi.org/10.1093/jnci/djz049), especially Tables 2–4.
[Provenance](top-binary-sources.json) records the public documents used.

## Posterior and decisions

At a scheduled look, let n be enrolled patients, r observed responses, m pending
patients, and w the sum of their conditional time-to-response CDF weights.
The effective sample size is `n-m+w`. Known responses and known nonresponses
receive full information credit. Under uniform conditional response timing,
each pending patient's weight is follow-up divided by assessment-window length.

With a Beta(a,b) prior, TOP approximates the posterior as
`Beta(a+r, b+effective_sample_size-r)`. This is a fractional-information
approximation, not the exact pending-outcome likelihood. The default prior has
one effective observation centered on the null response rate: `(p0, 1-p0)`.
An explicit positive prior pair can be supplied.

The futility rule is posterior `P(p>p0) < C*(n/N)**gamma`, evaluated using the
direct upper tail. C and gamma are supplied tuning parameters, not automatically
calibrated by this constructor. The cutoff depends on **enrolled n**, not effective
sample size. Equality continues. Returned actions are `continue`, `stop_futility`,
`suspend`, or `success` at a completed final analysis.

Interim suspension is applied when responses do not already meet the complete-data
go threshold and pending counts reach the suspension threshold. A final analysis
always waits for all pending outcomes, including when enough responses already
meet its go threshold. The default `suspension="table"` uses
`pending >= ceil(n*n/N)` at interim looks. `suspension="strict"` follows the
article's prose instead: `pending > n*n/N`. These differ when `n*n/N` is an integer.

## Interfaces

`evaluate(patients, responses, pending, pending_weight)` broadcasts arrays of
summary counts and summed weights. Patient counts must be scheduled looks, and
response plus pending counts cannot exceed enrollment. Custom conditional timing
weights can be supplied explicitly; their appropriateness is a modeling choice.

`evaluate_followup(nonpending, responses, pending_followup, window)` calculates
uniform weights using the package's existing time-to-event ESS kernel. The last
axis contains pending patients. Follow-up and the window must share units;
normalization occurs before summation. Known outcomes include early responses
as well as completed nonresponses.

`boundaries()` returns scheduled enrollment, complete-data response minima,
suspension pending-count minima, and effective-size futility crossings. Rows
are looks and columns are response counts. Crossing values are unrounded:
strictly greater ESS stops for futility, unless suspension applies first.
Negative infinity means always futile over the feasible ESS range, positive
infinity means no crossing up to enrolled n, and NaN denotes an impossible count.
Use `evaluate` for decisions instead of rounding these crossings.

`complete_data_design()` returns the equivalent BOP2 design, whose existing
operating-characteristic methods apply when responses are completely observed.
Those operating characteristics do not validate performance with delayed responses.
Defaults use interim looks starting at 10 subjects and every 5 thereafter,
including the final sample size; supply explicit looks to match a protocol.
Up to 200 subjects are supported.

```python
import numpy as np
from mdanderson_stats import TOPBinaryDesign

design = TOPBinaryDesign(40, 0.2, 0.86, 0.95, looks=[10, 20, 30, 40])
table = design.boundaries()
np.testing.assert_array_equal(table.complete_go_min, [2, 4, 8, 12])
result = design.evaluate_followup(11, 3, [85, 78, 66, 48, 32, 28, 10, 8, 5], 120)
assert result.effective_sample_size == 14
assert result.decision == "continue"
np.testing.assert_allclose(table.futility_effective_size[1, 3], 15.091039493297)
```

Three focused tests reproduce the published table, verify the follow-up example
and time-unit invariance over 400 orders of magnitude, distinguish suspension
conventions, check final follow-up requirements, compare with complete-data BOP2,
and verify posterior crossings from both sides.

## Source differences and remaining coverage

The published Table 4 labels `(gamma,C)=(.86,.95)`, but its numerical crossings
are reproduced by **C=.86, gamma=.95**. The example above uses the verified order.
At n=20 of N=40, the table suspends with 10 pending while the prose's strict
inequality requires 11. Both conventions are exposed. The earlier arXiv table
has different thresholds; this implementation targets the published table.

**Catalog status is partial.** Calendar-time trial conduct, delayed-response
simulation and calibration, co-primary efficacy and efficacy/toxicity models,
the app's nonuniform timing elicitation, native reports and app version parity
remain pending. Original PDFs and application files are not redistributed.
