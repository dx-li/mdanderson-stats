# MTADF author logistic-prior source audit

The cached author program does not pass prior arguments to `arm::bayesglm`.
In `research/raw/mtadf-author-reference/targetAgentDF.r`, the `logistic()`
path constructs `x1s=(x-mean(x))/(2*sd(x))` and `x2s=x1s^2`, expands the
observed response counts with `collapse(yeff,n)`, and constructs
`x1reg=rep(x1s,n)` and `x2reg=rep(x2s,n)` before fitting
`bayesglm(yreg~x1reg+x2reg, family=binomial(link="logit"))` (around lines
426–490). The per-trial `df.logistic()` path does the same (around lines
516–565). Thus `bayesglm`'s prior-scale rule sees the patient-expanded
columns at each fit, not the unweighted dose grid.

The primary CRAN archive source inspected here is `arm` 1.6-06.01, published
before the author file date; 1.6-07.01 was also inspected as a nearby
comparison. The author program's exact installed `arm` version remains
unknown. Both source versions give `bayesglm` defaults `prior.df=1` and
`scaled=TRUE`; their manual identifies df 1 as Cauchy and gives logit
`prior.scale=2.5`, `prior.scale.for.intercept=10`, and predictor scaling by
range for two distinct values or `2*sd(x)` for more than two. The source
implementation's scalar-default path needs a qualification: it expands the
scalar `prior.scale=2.5` to all model-matrix columns (including the
intercept), while the separate intercept scale is inserted only when a
non-scalar scale vector is supplied. Consequently the actual author call's
intercept scale follows the implementation's expanded scalar path, not the
manual's stated scale-10 special case. No runtime execution was performed.

For each model fit, the scaling rule is applied to the values in the
patient-expanded design matrix. With one distinct observed dose, each slope
column has one value and its base scale remains 2.5. With two distinct doses,
the slope prior scales divide by that column's range. With more than two,
they divide by twice that column's sample standard deviation, weighted by the
patient replication counts. Prefix composition and unequal cohort sizes can
therefore change the effective slope scales. The initial dose-grid fact
`sd(x1s)=0.5` does not establish an effective scale of 2.5 for every prefix.
The source also appends prior rows to the design matrix and centers its
intercept prior row at the observed design-column means (`bayesglm.fit`'s
`initialize.x`); rank-deficient observed prefixes are therefore handled in
the prior-augmented fit rather than by dropping constant slope columns.

This audit records a source contract for the cached references. It does not
claim that a Python approximation reproduces `bayesglm`'s iterative fitting,
rank/pivot details, or the historical application's exact installed package.

## Cached source provenance

Both tarballs were downloaded from the official CRAN archive and inspected
without installing or running the package:

| Archive | SHA-256 | Relevant members |
| --- | --- | --- |
| `arm_1.6-06.01.tar.gz` | `31009fc9ce175358a1f982d9c022498b1f6246307c2edc32ba904e929037302d` | `arm/R/bayesglm.R`, `arm/R/model.matrix.bayes.R`, `arm/man/bayesglm.Rd` |
| `arm_1.6-07.01.tar.gz` | `69aa2793778b1d3e2f6c2d650c5cf0ea55d3e305eb8af5a7a68ed680098a1d7a` | same members, comparison version |

The source cache is under
`research/raw/mtadf-author-reference/arm/`. In `bayesglm.R`, see
`bayesglm()` for omitted-argument defaults and delegation to `bayesglm.fit`,
`.bayesglm.fit.initialize.priors()` for scalar expansion, and
`.bayesglm.fit.initialize.priorScale()` for the design-column scaling rule.
`initialize.x()` appends prior rows and centers the intercept row. The
companion `model.matrix.bayes.R` shows the numeric predictor columns retained
in the formula matrix. The cached author file remains the primary evidence
for response expansion and the actual call.
