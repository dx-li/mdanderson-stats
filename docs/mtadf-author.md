# MTADF author-reference isotonic policy

`mtadf_author_decision` is a separate implementation of the isotonic policy in
the authors' `targetAgentDF.r` reference. It is kept distinct from
[`mtadf_decision`](mtadf.md), which follows the paper's calibrated-prior and
strict-safety policy.

```python
from mdanderson_stats import mtadf_author_decision

interim = mtadf_author_decision(
    subjects=[3, 6, 3],
    toxicities=[0, 1, 1],
    responses=[0, 4, 1],
    current_dose=1,
)
final = mtadf_author_decision(
    subjects=[3, 6, 3], toxicities=[0, 1, 1], responses=[0, 4, 1], final=True
)
print(interim.dose, final.dose)  # zero-based dose indices
```

The author program fixes a Beta prior with total concentration `0.5` by solving
`BetaCDF(0.3; alpha, 0.5-alpha) = 0.22`. This prior stays fixed even when a
different `toxicity_limit` or `safety_cutoff` is supplied. Posterior overdose
probabilities are increasing-isotonic pooled with equal weights. Doses pass
when the adjusted probability is **at or below** the cutoff; if none pass, the
reference still retains the lowest dose. `admissible_dose_count` reports the
fresh cap from the observed data.

The Python solver uses a tight Brent root (`alpha` approximately
`0.353639544999907`). The author's R `uniroot` uses its own stopping tolerance,
so this does not claim bitwise-identical prior parameters.

The author uses unit-weight unimodal regression on dose response rates and
resolves fitted-maximum ties to the rightmost dose. Interim input must contain a
contiguous observed prefix and the current dose must have observations. The
rule moves one dose toward the rightmost peak, or explores the next dose when
the current dose is the highest tried peak, then caps at the author admissible
prefix. The first cohort creates a one-dose observed prefix. Since the
historical `Iso::ufit` one-point mode-search path is undefined, this Python
implementation extends it by using that single observed rate unchanged; this
edge behavior is not a claim of native parity. Before enrollment, the result
is a `start` action at dose index zero.

Final selection differs from the paper-policy API. It fits all dose levels to
`responses / (subjects + 0.0001)`, including untried levels, chooses the
rightmost fitted maximum, then caps it at the inclusive author admissible
prefix. A final result can therefore select an untried dose. Counts are
complete aggregate outcomes; this function does not model pending assessments.
Inputs allow 1–20 doses and at most 10,000 total subjects.

The author reference uses a lagged safety cap in its OC simulation: movement
after a cohort is capped by the admissible prefix computed before that cohort,
then the prefix is refreshed. A private decision helper exposes this cap for
the separate simulation wrapper; direct calls to `mtadf_author_decision`
always use the fresh cap.

This is an author-reference isotonic policy, not a claim that the live app uses
every line of the retrieved R file unchanged. In particular, the file's
simulation code and actual-trial function use different cap timing. See the
[source audit](../research/mtadf-author-audit.md). The separate global and
local logistic methods remain documented at [MTADF logistic](mtadf-logistic.md).
