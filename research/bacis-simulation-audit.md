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
installation or large simulation. Python comparisons remain pending.

The singleton oracle verifies decision semantics. Multigroup borrowing
continues to rely on the separately validated two-stage model and its
retained convergence diagnostics.
