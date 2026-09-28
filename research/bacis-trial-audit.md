# BaCIS one-trial workflow

The archived bacistool 1.0.0 `R/bacisOneTrial.R` returns ten rows per subgroup:
response probabilities above the low and high thresholds; positive latent-theta
probability; high-cluster and efficacy indicators; posterior and observed
response rates; response and patient counts; and equivalent sample size.
The function rounds the completed table to three decimal places. Its cluster
and efficacy decisions use strict inequalities on the unrounded probabilities.
Source provenance is recorded in [bacis-sources.json](../docs/bacis-sources.json).

`bacis_one_trial` composes one call to the existing two-stage fit with one ESS
calculation from the same retained response draws. It retains the unrounded
table as well as a separately rounded report, original data, fit and ESS
diagnostics. No additional posterior fit or DIC calculation is hidden in
report generation. The source's commented-out DIC return does not add a row
to its actual returned table.

The wrapper's hierarchical precision-rate default is 2, matching this CRAN
function, while the package's `bacis_fit` defaults to the app's displayed 10.
Sampler defaults remain 2,000 retained draws, 1,000 warmup and two sequential
chains. The native wrapper defaults to 50,000 MCMC iterations with different
adaptation/thinning conventions. Exact classification probabilities and
singleton Beta summaries preserve the mathematical target while differing
from those native finite-chain values. Native numerical random-stream and
file-format equivalence are not claimed.

Two focused tests passed with warnings as errors: an independent singleton
posterior check and a small five-subgroup borrowing workflow. Ruff and module
type checking passed. The public singleton example was additionally checked
against direct binomial sums for both Beta(3,24) upper tails. Setting both
decision cutoffs exactly to their probabilities correctly returned low-cluster
and not-effective indicators. This audit took .005 seconds after import,
peaked at 115.0 MiB and reported zero swaps. The small borrowing run verifies
workflow composition, not applied-model convergence.

## Paper simulation scope

The CRAN package exports one-trial analysis, posterior extraction, plots and
DIC, but no operating-characteristic simulator. The paper describes six
five-subgroup scenarios with 25 patients per subgroup, low/high response
rates .1/.3, zero through five low-response groups, and 5,000 trials per
scenario. It uses classification cutoff .5, efficacy cutoff .92, precision
shape 50 and rate 2. Its outputs include subgroup classification/rejection
rates, global-null familywise error, the proportion assigned to a single
cluster, and per-subgroup ESS.

The paper's stated 10% type-I-error calibration concerns subgroup error;
familywise error is reported separately as 17.1% under the global null. The
threshold-calibration search and exact simulation RNG are not supplied. A
future bounded Python simulation can implement this declared scenario model,
with explicit replication and sampler budgets; it must not present an
invented search algorithm or random stream as a native software contract.
