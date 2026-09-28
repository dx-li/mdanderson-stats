# BaCIS equivalent sample size

The archived `bacistool` 1.0.0 function `compESS`, called by `OneTrial`, uses
the variance of retained response-probability draws. The caller uses R's sample
variance, including its `n-1` denominator. Source provenance is recorded in
[bacis-sources.json](../docs/bacis-sources.json).

## Interpretation and source distinction

For observed response count `y` and posterior variance `v`, its cubic is

```text
N^3 + 7*N^2 + [16-(y+1)/v]*N + 12-(y+1)*(1-y)/v = 0.
```

This is equivalent to matching `v` to the variance of
`Beta(y+1, N-y+1)`. The count `y` stays fixed while real-valued equivalent
sample size `N` changes. Among roots, the source selects the one whose
`y/N` is closest to the observed response proportion. It is a total equivalent
patient count, not an effective number of Monte Carlo draws and not an extra
number of prior patients.

The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6546564/) describes matching
posterior mean and variance to a beta distribution. The software instead uses
the observed response count and observed rate, not the posterior mean, in this
calculation. These are not generally interchangeable ESS definitions. The Python
implementation explicitly identifies the software's variance-matching rule.

## Verified native defect and ambiguous roots

The source computes companion-matrix eigenvalues, replaces negative roots by
`.0001`, then ranks rate differences. This can select a value that does not
solve the variance equation. With zero responses and variance
`26/(27^2*28)`, the inspected source returns `.0001`. The unique admissible
equivalent count is 25. All candidate response-rate differences are zero, so
the source's root ordering and clamping determine its incorrect result.
Complex roots are not given a valid ordering by the source's negative comparison.
Python reports absent admissible solutions rather than fabricating roots.

Multiple admissible roots can be real statistical ambiguities. For `y=10` and
`v=11/980`, the polynomial factors as `(N-12)*(N^2+19*N-736)`.
The admissible roots are 12 and `(-19+sqrt(3305))/2`, approximately 19.24456.
For 10 responses among 25 observed patients, the larger root has the closer
response rate and is selected by the native function.

`tools/reference_bacis_ess.R` evaluates only the inspected numerical function
from the archived package. Eight small cases cover ordinary counts, borrowing,
all responses, two admissible roots and the zero-response defect. It records
native outputs and the analytically justified admissible expectation separately;
no JAGS session or clinical-trial simulation is run.

Under the explicit patient-count domain `N>=y`, the zero-response variance
function is strictly decreasing for `N>=0`. Thus there is exactly one
admissible solution for `0<v<=1/12`; the boundary `v=1/12` gives `N=0`.
At that boundary the zero-response rate criterion is defined by its limit
zero. The native ambiguity arises from including clamped inadmissible roots,
not from two admissible zero-response solutions.

Reference generation completed successfully for all eight cases using base R.
The integrated public `bacis_equivalent_sample_size` helper agrees with all
eight admissible references: maximum absolute equivalent-count error
`2.93e-10` (the 10,000-patient case), and maximum relative variance residual
`4.71e-14`. Both roots of the ambiguous example are retained; the zero-response
defect returns the admissible count 25. Results and candidate arrays are immutable.

Three focused tests passed, together with Ruff and module type checking. A
bounded public `bacis_fit` to ESS workflow also passed. The root reference and
workflow audit took .003 seconds after imports, peaked at 114.4 MiB resident
memory and reported zero process swaps. Full trial simulation and native random
stream reproduction were not run or claimed.
