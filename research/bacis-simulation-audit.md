# BaCIS operating-characteristic simulation

The reference model is archived bacistool 1.0.0, with provenance in
[bacis-sources.json](../docs/bacis-sources.json). Its package exports do not
include an operating-characteristic simulator. The
[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6546564/) describes simulations
under the following six five-subgroup response scenarios:

| Scenario | True response probabilities |
| --- | --- |
| 1 | .1, .3, .3, .3, .3 |
| 2 | .1, .1, .3, .3, .3 |
| 3 | .1, .1, .1, .3, .3 |
| 4 | .1, .1, .1, .1, .3 |
| 5 | .1, .1, .1, .1, .1 |
| 6 | .3, .3, .3, .3, .3 |

Each subgroup has 25 patients. The stated settings use low/high centers
.1/.3, hierarchical precision shape 50 and rate 2, classification cutoff
.5 and efficacy cutoff .92, with 5,000 trials per scenario. The cutoff
calibration algorithm and exact simulation streams are not provided.

## Distinct classification and efficacy events

A subgroup rejects its null when its second-stage posterior probability
`Pr(p_i > phi_low)` is strictly greater than the efficacy cutoff. Membership
in the high-response cluster is a separate event, not an additional gate.
Null groups have true response probability at most `phi_low`; a rejection
rate at an alternative truth is subgroup power. A truth between the low
and high centers is alternative to that null without being at the desired
high-response target.

Familywise error is the probability of at least one false rejection among
null subgroups. In the global-null fifth scenario all groups contribute to
that event. The paper's nominal 10% subgroup error and its reported 17.1%
global-null familywise error are different quantities. The frequency of all
subgroups receiving one common cluster label is another separate summary.

The paper also tabulates equivalent sample sizes. The initial bounded
Python simulator targets decision and classification rates; averages of
subgroup ESS remain separate work. A small integration run is not evidence
of reproducing the paper's five-thousand-trial estimates or of MCMC convergence.

## Unresolved paper classification discrepancy

The paper states that the classification cutoff is fixed at .5 across its
operating-characteristic scenarios. In the pinned package's `model1Str`,
both logit centers and both precisions are fixed, with independent
subgroup-specific latent signs and response logits. A subgroup's posterior
classification probability therefore depends only on its own response count
and sample size. With a fixed cutoff, identical n=25, p=.1 subgroups must
have the same population classification rate regardless of other subgroups.

However, Table 1 reports .159 for the lone low-response group in scenario 1
and .039 for the first group in the global-null scenario. The package's
adaptive cutoff uses an overall observed-response summary and would permit
such dependence. It is a possible explanation, not a verified account of
how the published table was generated. The Python simulator can expose
both archived-software cutoff modes, but neither the table's exact
classification results nor its calibrated efficacy rates are certified
reference targets without resolving that discrepancy.

Enumeration of the integrated fixed-.5 classifier over all 26 possible
response counts confirmed a high-cluster decision at y>=5. Independent
binomial sums then give classification probability .0979936212 for a
true response rate .1 and .9095280814 for .3. These probabilities cannot
change with other subgroups when this fixed classification rule is used.

## Independent singleton oracle

For a subgroup in a singleton cluster, the native second stage uses
`Beta(1+y, 1+n-y)` regardless of its first-stage label. The upper beta tail
at probability `p` is the finite binomial sum
`Pr[Binomial(n+1,p) <= y]`. Thus the efficacy rejection set can be enumerated
without a fitted hierarchy or posterior simulation.

`tools/reference_bacis_single_group_oc.R` uses independent base-R binomial
sums to generate 26 rows for all response counts with n=25. At phi_low=.1
and cutoff=.92, rejection occurs exactly when y>=5. The exact null rejection
probability is `0.097993621195464703`; the alternative probability at .3 is
`0.9095280814458635`. With a single null group, familywise and subgroup
error coincide. The fixture also records both beta tails and the null and
alternative binomial masses. Generation took .080 seconds, with no package
installation or large simulation. Python checks replayed the independent
table decisions in 32 trials at each truth (.1 and .3), confirmed the
fixed-cutoff classifications and checked the public guide. A three-replication
mixed-group run verified subgroup/familywise/single-cluster aggregation from
the retained indicators. The combined check took .224 seconds after imports,
peaked at 114.7 MiB RSS and reported zero process swaps. Three focused tests
also passed; lint, formatting and type checks passed. These deliberately short
mixed-group chains had maximum probability-chain R-hat values 2.10–2.47 and
verify plumbing only, not converged multigroup operating characteristics.

The singleton oracle verifies decision semantics. Multigroup borrowing
continues to rely on the separately validated two-stage model and its
retained convergence diagnostics.

## Optional subgroup equivalent sample sizes

The archived `bacistool` source returns one ESS for each subgroup in a
completed trial. In `R/internal.R`, `OneTrial` computes
`compESS(1 / var(p.sampled), xDat[i], xObs[i])` separately for subgroup `i`;
`bacisOneTrial.R` places this vector in the “Effective sample size” result row.
The input variance is the unbiased variance of the retained posterior response
probability draws. The implementation follows the package's fixed-response-
count variance matching, which chooses among admissible roots using the
observed response rate. This is more specific than the paper's description of
matching beta posterior mean and variance, so the optional simulator summary
does not claim to reproduce the paper's published ESS table exactly.

When requested, Python retains this per-trial, per-subgroup vector and reports
the arithmetic mean over replications for each subgroup, with sample-standard-
deviation MCSE. The archived package and paper report subgroup-specific
values, but the simulation aggregation code was not included in the recovered
package; arithmetic averaging is therefore an explicit Python summary rule.
Undefined fixed-y matches stop the simulation with replication/subgroup
context rather than dropping that replicate.

Focused verification for the optional path passed 10 BaCIS simulation/ESS tests
with warnings treated as errors (4.20 seconds, peak RSS 129,810,432 bytes or
123.80 MiB, zero swaps). An independent base-R polynomial-root fixture matched
20 per-trial subgroup ESS values and five subgroup mean/MCSE summaries; the
largest absolute ESS difference was `9.31e-13`. Ruff check/format and mypy
passed for the changed simulation module and tests. This is bounded numerical
verification, not full paper-table or convergence validation.
