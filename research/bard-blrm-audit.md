# BARD paper BF-BLRM contract — September 28, 2026

This prepares an explicitly configured component from the
[BARD paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12240483/), not a claim
that the current app implements the same route. The official guide describes
BF-BOIN as its stage-one design. Cached paper/guide evidence is under ignored
`research/raw/BARD`; provenance is recorded in [bard-sources.json](../docs/bard-sources.json).

## Model and prior

The reviewer visually inspected PDF page 9 after the extracted equation raised
a possible log-dose ambiguity. The printed model is exactly

```text
logit(p_j) = log(alpha) + beta * (d_j / d_star),  alpha > 0, beta > 0
(log(alpha), log(beta)) ~ N((mu_alpha, mu_beta), diag(var_alpha, var_beta))
```

There is no logarithm around the dose ratio. Substituting the customary
log-dose BLRM would change the published model. All prior and dose inputs
should be explicit. Page 18's simulation uses means `(-1.1, 0)`, variances
`(4, 1)`, doses `(10, 20, 50, 100, 200)`, reference dose 50, target interval
`(.16, .33)` and overdose cutoff `.30`; these are example values, not verified
native defaults.

## Decisions and backfill

Pages 9–11 specify `PTT_j = Pr(gamma1 < p_j < gamma2 | data)` and
`POD_j = Pr(p_j >= gamma2 | data)`. Among doses with `POD_j < eta`, select the
maximum PTT and move one level toward it from the current dose, or stay if
already there. If all POD values exceed eta, terminate with no MTD. Equality
leaves a gap: all doses at eta satisfy neither strict safety eligibility nor
the strict all-over-toxic rule. A future implementation must distinguish
`no eligible safe dose` and cannot invent an MTD. Tie handling also needs an
explicit convention unless additional evidence establishes it.

Backfill eligibility uses a dose below the current dose and observed response
at that dose or below. It closes when POD is at least eta or the evaluable
patient count reaches the dose cap. Assignment prioritizes filling the
current escalation cohort, then the highest open backfill dose, following the
paper's BF-BOIN scheduler reference. It does not use BF-BOIN's conflict-pooling
rule: the monotone fitted model incorporates all dose data directly.

Final stage-one MTD selection uses all escalation and backfill observations,
requires at least six treated patients, and maximizes PTT among doses with
POD below eta. Existing BF-BOIN calendar components offer scheduling patterns,
while `bard.py` supplies stage two; neither currently fits this model.
The first proposed component is model evaluation and bounded posterior fitting
with target/overdose summaries. No BF-BLRM implementation is claimed by this
source-audit checkpoint.
