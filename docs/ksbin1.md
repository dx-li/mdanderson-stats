# KSBIN1 fixed-design operating characteristics

Catalog entry 25 is partial. Fixed multistage design evaluation is implemented;
the single-stage design solver, boundary-selection assistance tables, comparison
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

Still pending: solve_binomial_one_sample_mod and ABIN1's five solve modes (null
probability, alternative probability, sample size, significance and power),
BINCUM-based power-contribution tables, separate single-stage comparison, report
file dialogue and design revision workflow. These are required before this catalog
entry can be marked implemented.
