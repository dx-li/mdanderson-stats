# KSBIN1 binomial trial design and operating characteristics

Catalog entry 25 is partial. Fixed multistage design evaluation and all five
single-stage calculation modes and boundary-selection assistance tables are
implemented. A consolidated comparison with a separately chosen single-stage
trial and original reporting/session workflows remain pending. Source: KSBIN1_V1.tar.gz, ksbin190_1.0.

```python
from mdanderson_stats import ksbin1_operating_characteristics

result = ksbin1_operating_characteristics(
    cumulative_trials=[14, 28, 42],
    critical=[0, 1, 3],
    quit=[3, 4],
    probability=[0.2, 0.06],
    alternative="less",
)
print(result.rejection_probability)  # approximately [0.0610041, 0.792488]
print(result.expected_sample_size)  # approximately [21.2209, 25.9792]
```

The probability argument accepts arbitrary arrays. Supplying the null and
alternative probabilities together returns significance and power, respectively.
Leading result dimensions follow that array; the last axis of rejection, quitting
and continuation indexes stages. These stage probabilities are unconditional,
not conditional on surviving earlier stages.

For alternative="less", rejection occurs at or below critical and quitting at
or above quit. For "greater", rejection occurs at or above critical and quitting
at or below quit. Both cutoffs are inclusive; -1 disables a cutoff. Supply critical
for every stage and quit for interim stages only. Every nonrejection at the final
stage quits, even when final rejection is disabled.

The function reuses KStageBinomial's immutable design and cached surviving-path
coefficients. Its public stage_distribution method returns the joint probability
of reaching a stage and observing each cumulative event count (event count is the
last axis). The distribution's total equals the probability of reaching that
stage, so it generally sums to less than one after the first stage. Both APIs use
vectorized binary64 log-weighted probability terms.

## Returned summaries

- rejection and quitting: probability of each decision at each stage.
- continuation: probability of continuing past each stage; final value is zero.
- rejection_probability: total rejection probability.
- expected_sample_size: cumulative stage sizes weighted by termination probabilities.
- expected_given_rejection / expected_given_quitting: conditional expected cumulative
  sample sizes. NaN explicitly means that the conditioning decision has zero
  probability; the corresponding decision probability identifies these cases.
- design, critical, alternative and probability: the evaluated inputs.

Under the null, expected_given_quitting is the source's expected sample size given
a correct decision. Under the alternative, expected_given_rejection is its expected
sample size given a correct decision. The API exposes both conditional expectations
without presuming which supplied probability is the null.

Sizes are increasing cumulative counts, with 1–10 stages and at most 200 total
trials. Overlapping stopping regions and unreachable planned stages raise. The
source binomial-mass helper additionally limits increments to 100; the Python
calculation removes that allocation restriction within the total-200 domain.
Probabilities include 0 and 1, beyond the original input menu's interior range.
No stopping-boundary search or statistical optimization is inferred from these
operating characteristics.

## Validation and remaining coverage

`tools/reference_ksbin1.py` extracts the complete NXTSTG, BINDEN, EXPCOR and EXPOV
routines unchanged into an isolated module with equivalent binary64 constants.
An independent driver applies supplied boundaries, calls the original transition
routine and expected-sample-size routines, and records 18 cases: three designs,
two directions and three probabilities. `tests/fixtures/ksbin1.json` records
archive/source/extracted-code hashes and compiler provenance. This is a reference
for the numerical core, not a test of the original interactive main program.

Independent tests enumerate every Bernoulli sequence in a small three-stage design
to verify rejection, quitting, continuation and expected sample sizes. Tests also
check mass conservation, broadcasting, endpoints, disabled final rejection,
conditional expectations with probability zero and invalid input. Existing KSB1CI
native and exhaustive tests validate the shared probability-code extraction.

Still pending: consolidated single-stage comparison, report file dialogue and
design revision workflow. These are required before this catalog
entry can be marked implemented.

## Single-stage power and adjacent critical regions

```python
from mdanderson_stats import binomial_power

power = binomial_power(trials=50, null_probability=0.2, alternative_probability=0.06, alpha=0.05)
print(power.critical, power.significance, power.power)
print(power.next_critical, power.next_significance, power.next_power)
```

`binomial_power` provides the source solver's forward power calculation and XBIN1
bracketing output. Direction follows the alternative probability. The selected
region is the most permissive nonrandomized one-sided region whose achieved
significance is at most alpha. The next region includes one more event count.
Results retain the inputs and both critical counts, achieved significances and
powers. All inputs broadcast, including mixed directions within an array.

Integer bisection searches the number of included event counts, avoiding rounding
a continuous binomial quantile. Probabilities and alpha must be strictly inside
(0,1), null and alternative probabilities must differ, and trial counts must be
positive integers smaller than 2**53. The source inconsistently rounds or truncates
fractional sample sizes across helpers; Python rejects them. The comparison is
against evaluated binary64 probabilities, without an added tie tolerance.

An empty lower region has critical=-1; an empty upper region has critical=n+1.
Both have significance and power zero. This replaces XBIN1's invalid numerical
sentinels and its negative-count CDF call for empty upper regions. The next region
is still reported. This is an explicit empty test, not a claim that the requested
power has been reached.

`tools/reference_binomial_power.py` compiles a private KSBIN1 source copy and calls
XBIN1 for 36 inputs. As in the ONESAMPLE reference build, five cdf_aux success-status
outputs are initialized at entry; statistical formulas remain unchanged. The
fixture records the source hash, patched auxiliary hash, exact patch and compiler.
It retains native diagnostics when XBIN1 cannot report an empty upper region.
Tests compare valid native brackets, check empty-region behavior independently,
enumerate binomial masses for well-separated attainable sizes, and verify exact
size ties, adjacent regions, broadcasting and invalid inputs.

All five calculation modes are available through the forward power, significance,
alternative-probability, null-probability and sample-size APIs. Interactive design
assistance and the remaining workflows below are still pending.

## Solve significance for a requested power

```python
from mdanderson_stats import binomial_significance

result = binomial_significance(50, 0.2, 0.06, target_power=0.8)
print(result.critical, result.significance, result.power)
print(result.previous_critical, result.previous_significance, result.previous_power)
```

The result selects the least permissive nonrandomized one-sided region whose
power reaches the target. previous_* reports the region with one fewer included
event count; its power is below the target. This solves the significance mode
without treating its step function as continuous. Inputs and direction conventions
match binomial_power; target_power is strictly inside (0,1). Inputs broadcast.
A full rejection region is reported with significance=power=1 when needed, rather
than fabricating a less permissive test that fails to reach the target.

The integer search uses logarithmically many count evaluations and shares its
region evaluator with binomial_power. It compares binary64 tails directly, without
an arbitrary tolerance around the requested target. The result retains trials,
null/alternative probabilities and target_power as well as both candidate regions.

The original significance solve searches its alpha parameter only from SRANGE=1e-8
upward. Python allows smaller attained sizes. At n=100, null=0.3, alternative=0.6,
and target power=0.5, Python rejects at count 60 or above: significance approximately
5.12995e-10 and power 0.543294. The original search floor selects a more permissive
region with size 5.93229e-9 and power 0.696740. Direct binomial mass sums verify the
Python result and show that excluding count 60 drops power below the target.

`tools/reference_binomial_significance.py` reuses the documented private source
build and records 27 calls to solve_binomial_one_sample mode 4. The native results
agree in 26 cases; the remaining case agrees with the original floor-constrained
choice and is independently verified for the expanded Python domain. The fixture
retains source/auxiliary hashes, status repair and compiler provenance. Tests also
cover enumerated attainable power steps, exact ties, full regions, mixed-direction
broadcasting, very small target power, sample sizes up to 1e10 and invalid inputs.

## Solve the alternative probability

```python
from mdanderson_stats import binomial_alternative

result = binomial_alternative(50, 0.2, alpha=0.05, target_power=0.8, alternative="less")
print(result.alternative_probability, result.critical, result.power)
```

This holds the size-controlled test fixed and finds the closest alternative
probability in the requested direction that attains the target power. Direction
is explicit ("less" or "greater"); all numerical inputs broadcast. Null probability,
alpha and target_power are interior probabilities, with target_power>=alpha.
The remaining count rules match binomial_power. An empty critical region raises:
its power is zero for every possible alternative. If the requested power already
equals the attained significance at the null, the returned probability is the
null itself, an explicit boundary solution.

An inverse regularized beta function seeds the search for the selected critical
count. The bracket is then refined using the package's inclusive binomial tails
until its endpoints are adjacent binary64 values (or coincide at the null).
The endpoint on the alternative side is returned and its attained power checked.
The result includes probability_lower and probability_upper for inspection, as
well as the input parameters, critical count, significance and achieved power.
These brackets reflect evaluated floating-point tails, not interval-arithmetic
bounds on numerical error in the special functions. Rounding can make a power
curve locally flat; the bracket locates the crossing of that evaluated curve.

The search has a finite cap of 1076 halvings to cover subnormal probabilities;
ordinary cases finish in far fewer iterations. It avoids a fixed absolute stopping
tolerance that would erase roots near zero. Nonfinite inverse results or failure
to converge/attain the target raise rather than returning an unverified solution.

`tools/reference_binomial_alternative.py` records 27 native mode-2 solves using
the shared documented reference build and auxiliary status repair. The solved
probabilities agree within the original solver's accuracy. Native mode 2 nudges
its solution outward by a relative correction; its reported power is verified at
that actual probability. Separate tests check the tighter Python root, closed-form
zero/all-event formulas, adjacent brackets, broadcasting, null-boundary equality,
empty regions, invalid inputs and a representable probability near 1e-310.

## Solve the null probability

```python
from mdanderson_stats import binomial_null

result = binomial_null(
    50, alternative_probability=0.06, alpha=0.05, target_power=0.8, alternative="less"
)
print(result.null_probability, result.critical, result.significance, result.power)
```

This finds the closest null to the supplied alternative for which a nonrandomized
one-sided test satisfies both constraints. It first chooses the least permissive
critical region attaining the target power at that alternative, then inverts the
region's null tail at alpha. A more restrictive region fails the power constraint;
a more permissive one requires a null farther from the alternative. previous_power
reports the first fact explicitly for the immediately smaller region.

Direction is explicit and numerical inputs broadcast. Input alternative probability
is interior, trials are positive integers, and 0<alpha<=target_power<1. If only the
full rejection region attains the target, the function raises because that region
has significance one for every null. If the requested power and attained size
already coincide at the supplied alternative, the null equals that alternative.

The result includes trials, alternative_probability, alpha, target_power, direction,
null_probability, critical count, significance, achieved power and previous_power.
probability_lower and probability_upper bracket the evaluated null-tail crossing
at adjacent binary64 values; the returned endpoint satisfies significance<=alpha.
The inverse-beta initialization and refinement are shared with binomial_alternative,
including its explicit convergence checks and floating-point interpretation.
As with the other Python solve modes, the original SRANGE/BRANGE probability
search restrictions are removed, permitting representable near-boundary solutions.

`tools/reference_binomial_null.py` records 27 calls to the original mode-1 solver
using the same documented reference build and auxiliary status initialization.
Solved nulls and powers agree within native accuracy; its reported significance is
verified at the native null, accounting for the source's outward correction.
Independent checks cover zero/all-event closed forms, adjacent brackets (moving
the null closer violates alpha), mixed target/sample-size broadcasting, equality
at the null, impossible full-region cases, invalid inputs and subnormal nulls.
The existing alternative/significance/power tests verify the shared search extraction.

## Solve the minimum sample size

```python
from mdanderson_stats import binomial_sample_size

result = binomial_sample_size(0.3, 0.35, alpha=0.05, target_power=0.5)
print(result.trials)  # 236
print(result.critical, result.significance, result.power)
```

Numerical inputs broadcast. The returned BinomialPower contains the smallest
qualifying sample size in the requested inclusive bounds and both candidate
critical regions. Defaults are min_trials=2 (the source solver's lower bound),
max_trials=1e10, and batch_size=256. One trial may be explicitly allowed. Bounds
and batch size must be positive integers, max_trials cannot exceed 1e10, and
alpha<=target_power<1. Other probability rules match binomial_power. An error means
no qualifying design was found within the supplied bounds, not that all conceivable
sample sizes are infeasible.

Nonrandomized power can decrease as sample size increases because the critical
count changes. At null=0.3, alternative=0.35 and alpha=0.05, size 236 reaches power
0.5 while size 237 does not. A direct root finder or binary search on that power
curve can miss the first feasible size.

The search uses the randomized most-powerful test only as an upper bound. If the
selected and adjacent regions have (size, power) pairs (s0,w0) and (s1,w1), its
power is w0 + (alpha-s0)/(s1-s0)*(w1-w0). Randomization includes part of the next
boundary mass. Its optimality follows from the [Neyman–Pearson lemma](https://stat210a.berkeley.edu/fall-2024/reader/hypothesis-testing.html).
Our search argument is that its power cannot decrease with sample size: a test
could always ignore additional observations. Thus a randomized-power crossing
bounds the earliest possible nonrandomized design.

Geometric expansion and binary search locate that crossing. Ordered vectorized
batches then examine integer sizes starting one before the bound, finding the
first actual nonrandomized test reaching target_power. The returned test uses no
randomization. The bound calculation adds 1e-12 upward near floating-point ties;
if boundary mass cannot be resolved it uses the safe upper bound one. This is a
numerical search with the package's evaluated binomial tails, not interval-arithmetic
certification. Degenerate tails or near-equal significance/power requirements can
cause a longer ordered search. Memory is bounded by the input case count times
the configured batch size, rather than the search's upper sample-size limit.

`tools/reference_binomial_sample_size.py` records 36 native mode-3 solves with the
shared reference build and auxiliary status repair. Thirty match the Python
minimum. In three cases the original heuristic returns larger feasible samples
(e.g. 245 instead of 236, and 1403 instead of 1385). Three others report n=2 and
significance 0.51 despite a requested limit of 0.1. The original results remain in
the fixture; Python enforces the requested limit. Every reference case is checked
against all smaller integer sample sizes using the forward evaluator. Independent
binomial mass sums, a power-dip regression, batch-size invariance, inclusive bounds,
no-solution cases and invalid inputs provide additional checks.

A local Python 3.13 / NumPy 2.5.3 run for null=0.3, alternative=0.301, alpha=0.05 and
target_power=0.8 returned n=1,299,497, significance approximately 0.0499948 and power
0.800002 in 0.060 seconds. This is one workload-specific timing, not a universal
runtime guarantee.

## Boundary-selection assistance

```python
from mdanderson_stats import KStageBinomial, ksbin1_boundary_table

# Previously selected boundaries apply; future boundaries can remain disabled.
design = KStageBinomial([14, 28, 42], low=[0, -1], high=[3, -1])
table = ksbin1_boundary_table(
    design,
    stage=2,
    null_probability=0.2,
    alternative_probability=0.06,
    single_stage_critical=4,
    alternative="less",
)
print(table.events, table.significance, table.power, table.power_loss)
```

The reference single-stage trial has the design's final sample size and an
explicit inclusive critical count. The returned single_stage_significance and
single_stage_power give its operating characteristics. Probabilities broadcast;
the last array axis always runs through ascending event counts, including counts
made unreachable by earlier stopping. Stage numbers start at one.

At each candidate count, significance and power sum the joint arrival mass in
that count's rejection region. They are this stage's contributions, so earlier
rejection contributions must be added to obtain cumulative rejection probability.
For "less", rejection includes counts at or below the candidate; for "greater",
it includes counts at or above it.

conditional_reference_power is the chance of rejecting in the reference trial
given the current event count and completion of all remaining observations.
power_contribution multiplies this by the alternative's joint arrival mass.
power_loss accumulates these contributions over the candidate's inclusive
futility region: at or above the count for "less", at or below for "greater".
This is absolute lost probability, not a percentage of reference power. Future
stopping boundaries do not enter this reference-completion calculation. It is
design assistance, not an exact power difference between arbitrary multistage
designs. Selecting overlapping rejection and futility regions is invalid; the
KStageBinomial constructor validates the resulting chosen boundaries.

The Python table corrects a source boundary error: the lower-tail contribution
loop sets the conditional probability to zero when the current count equals the
reference cutoff. It should include the probability of zero additional events.
For example, after one stage of 2 observations in a 4-observation trial with
alternative probability 0.2 and reference cutoff 1, the corrected power_loss at
count 1 is 0.2048; the original gives zero. This also affects the original manual's
first-stage table at count 4 and every cumulative loss entry preceding it.

The source's BINCUM array holds only 100 remaining observations, and some cutoff
values can index outside its initialized range. Python supports all remaining
sizes within the shared total-200 limit and explicitly handles thresholds beyond
the remaining count range, including the final stage with no observations left.

Validation includes 54 exhaustive six-observation path cases spanning both
directions, all three stages, cutoff endpoints and probability endpoints.
Broadcasting, disabled boundaries, the exact-cutoff regression and 199 remaining
observations are also tested. `tools/reference_ksbin1_table.py` compiles unchanged
BINCUM/BINDEN routines and the original main-program contribution/cumulation loop
with bounds checking. Twelve native first-stage tables are preserved in
`tests/fixtures/ksbin1_table.json`: upper-tail tables match directly; lower-tail
tables match after adding the independently calculated omitted path probability.
Source and extracted-driver hashes and compiler settings accompany the fixture.
