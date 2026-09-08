# KSBIN1 fixed-design operating characteristics

Catalog entry 25 is partial. Fixed multistage design evaluation is implemented;
the remaining single-stage inverse solve modes, boundary-selection assistance tables, comparison
with a separately chosen single-stage trial, and original reporting/session
workflows remain pending. Source: KSBIN1_V1.tar.gz, ksbin190_1.0.

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

Still pending: solve_binomial_one_sample_mod and ABIN1's four inverse solve modes
(null probability, alternative probability, sample size and significance),
BINCUM-based power-contribution tables, separate single-stage comparison, report
file dialogue and design revision workflow. These are required before this catalog
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

The forward power mode is now available. Solving for null probability, alternative
probability, sample size or significance remains pending; no continuous root search
is substituted for those discrete design tasks.
