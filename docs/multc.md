# Multc Lean and Multc99 Phase IIa

Catalog entries **12 (Multc Lean)** and **3 (Multc99)** are partial. Python
provides marginal response/toxicity monitoring, full and reachable stopping
boundaries, sequential decisions, and exact joint operating characteristics.
This includes the fixed-reference Phase IIa rules described for Multc99;
Multc99's broader multiple-event designs remain pending. The official
[Multc Lean catalog](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/12)
describes that relationship. [Source provenance](multc-sources.json) records the
versions and inspected documents. No original program code or binaries are
redistributed.

## Define a design

```python
from mdanderson_stats import multc_lean_design

# Published tutorial example; tuples contain beta success/failure shapes.
design = multc_lean_design(
    30,
    response_prior=(0.6, 1.4),
    toxicity_prior=(0.5, 1.5),
    historical_response=(30, 70),
    historical_toxicity=(20, 60),
    response_cutoff=0.95,
    toxicity_cutoff=0.95,
)
state = design.monitor(responses=0, toxicities=0, sample_size=6)
print(state.response_probability, state.decision)  # about .964594; stop_response
```

For each endpoint, observing `y` events in `n` patients updates the experimental
prior to `Beta(a+y,b+n-y)`. Historical beta distributions are unchanged and
independent of their corresponding experimental rates. A scalar historical
argument instead supplies a fixed rate. Response and toxicity in the **same
patient** need not be independent.

The [statistical tutorial](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcTutorial.pdf)
defines the response rule as
`Pr(historical_response + response_margin > experimental_response) > response_cutoff`.
The toxicity rule is
`Pr(historical_toxicity + toxicity_margin < experimental_toxicity) > toxicity_cutoff`.
Both comparisons are strict. A cutoff of one disables that endpoint. Fixed
thresholds outside the probability support are handled exactly, including a
zero cutoff with an impossible event.

`max_subjects` is 3–1000, the cohort size must divide it, and `min_subjects`
must be a cohort multiple or smaller than one cohort. Looks are the cohort
multiples at or above the minimum. Experimental beta shapes are in `(0,100]`,
historical shapes in `(0,1000]`, and margins in `(-1,1)`. Negative response and
positive toxicity margins cannot be combined, following the
[user guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcUsersGuide.pdf).

`pretrial_check=True` applies the guide's prior-only rejection screen at zero
enrollment. Its interaction with minimum enrollment is not completely specified
in the guide. Python treats it as a separate screen; use `False` when minimum
enrollment must be guaranteed. This choice also applies to reachability and
operating characteristics.

## Monitoring and boundaries

`monitor(responses, toxicities, sample_size)` broadcasts independent snapshots
at scheduled looks or zero. It does not infer enrollment from the two event
counts, since events can overlap. `monitor_outcomes` instead consumes ordered
binary rows `(response,toxicity)` and evaluates their scheduled cumulative
looks. Its returned history ends at the first stopping decision, ignoring any
later supplied outcomes. Partial final cohorts are not evaluated.

`stopping_bounds()` returns `response_stop_max` and `toxicity_stop_min` at
each look: stop for response at counts at or below the former, or for toxicity
at counts at or above the latter. A response bound of `-1` or toxicity bound
of `n+1` means that endpoint never stops at that look.

`potential_boundaries()` gives compact inclusive count intervals attainable
after surviving earlier looks. An interval with lower endpoint greater than
upper endpoint is empty. Reachability allows all four paired-outcome categories;
a particular scenario with zero-probability categories may have fewer reachable
states. Because continuation restricts each count separately, surviving states
form a rectangle; the implementation propagates its two intervals without
retaining a large grid for every look.

At maximum enrollment, `cap_complete` takes precedence over posterior stopping
reasons. It means the sample cap was reached, not that the treatment passed an
efficacy or safety criterion. Raw endpoint bounds and probabilities remain
available at the cap. This also distinguishes them from the historical display's
final placeholders. The official
[boundary comparison note](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/Multc99vsMultcLean.pdf)
explains the differences between full and potential representations.

## Exact joint trial probabilities

```python
# Category order: both, response only, toxicity only, neither.
oc = design.operating_characteristics([0.12, 0.28, 0.18, 0.42])
print(oc.expected_sample_size, oc.expected_responses, oc.expected_toxicities)
print(oc.sample_size_probability)  # index n, including possible n=0

# Explicit independence assumption; equivalent to the above four probabilities.
independent = design.operating_characteristics_independent(0.4, 0.3)
```

A two-dimensional recursion advances joint response/toxicity count mass one
patient at a time, absorbing it at scheduled looks. Results distinguish response
only, toxicity only, both, and cap completion. They include the sample-size PMF,
mean and SD, and expected response/toxicity counts. Under bounded stopping and
iid patient pairs, Wald's identity gives each expected event count as its true
marginal rate times expected enrollment. Full path enumeration checks this
against direct event-count accumulation. Changing association while retaining
both marginal rates can change the stopping distribution.

## Multc99 Phase IIa mapping

Use scalar historical rates, zero margins and cohort size one. Multc99's
response rule compares `Pr(experimental_response > historical_response)` to
a **lower** efficacy cutoff. Thus its `.05` cutoff corresponds mathematically
to Python's `.95` response cutoff. The toxicity cutoff is unchanged. To monitor
one endpoint, set the other cutoff to one. Python computes CDF and survival
tails directly and does not reproduce native text placeholder conventions or
claim bit-for-bit parity at floating-point cutoff ties.

## Numerical scope and remaining work

Fixed-reference probabilities use direct beta CDF/survival functions. Random
references reuse the package's beta-difference quadrature, with refinement near
a cutoff and a clear error if the decision remains unresolved. Error estimates
include quadrature estimates rather than rigorous interval bounds. Arrays are
read-only; oversized broadcast requests are rejected before numerical copies.
Monitoring accepts at most 10,000 broadcast snapshots. Boundary construction
uses monotone searches with a 20,000-comparison budget. Exact operating
characteristics are limited to 10 million count-state updates and a 4 MB bound
on the two principal working arrays;
a valid design can therefore be too large for this bounded exact calculation.

Validation covers the published N=30 tutorial boundaries, Multc99's N=15
fixed-reference example, independent R integration of shifted beta tails, and
all `4**6` outcome sequences for small continuous/cohort designs. The reference
generator is [tools/reference_multc.R](../tools/reference_multc.R).
The nine focused checks passed in 1.64 seconds (1.73 seconds including test-runner
startup), with 133.2 MiB peak process RSS and no swaps. Ruff and the affected-module
type check also passed. The full repository test suite was not run for this
isolated addition; validation concentrated on the statistical calculations and
bounded allocations.

Duration simulation remains pending: native
[accrual logistics](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcLogistics.pdf)
allow accrual to continue while outcomes are pending when they cannot change the
next decision. Native configuration/report formats, protocol documents, and
the general Multc99 multiple-event workflow are also pending. Parameter
elicitation and distribution inequalities already have separate package APIs:
`solve_distribution_moments`, `solve_distribution_quantiles`, and
`compare_beta_difference`.
