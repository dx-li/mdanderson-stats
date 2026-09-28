# Remaining source leads after EasyCellType

This is research triage, not implemented coverage. It supplements the executable
catalog and does not change pending status without a verified public workflow.

## SurvivalContour parametric models

The package already has ACCFLF accelerated-failure-time likelihoods, fitting and
shape searches (`accflf.py`, `accflf_model.py`, `accflf_search.py`). Inspect these
before adding another parametric fitter. They include a generalized-gamma
boundary, but use source-bounded limiting degrees of freedom and conditional
shape covariance; neither is automatically equivalent to the exact stable
generalized-gamma model and joint uncertainty used by `flexsurv`. The original
SurvivalContour `paraContour.R` is saved and pinned in the source audit. Future
coverage must verify each distribution's parameterization and interval method.

The first exact Weibull, log-normal and log-logistic AFT checkpoint is tracked
in [the parametric-survival audit](parametric-survival-audit.md). The verified
native source is `flexsurv` 2.3.2 at
`2aae4c8ac56823d0eac30c1a9ad654ac599b5938`. The generalized-gamma public workflow
is integrated at `7abe64e`; the spline workflow is also implemented and checked
as recorded in the [spline audit](survival-spline-audit.md).
Its small native generalized-gamma implementation and
headers are saved as exact source bytes under ignored `research/raw/flexsurv`:
`src/gengamma.cpp` (`34c799553ec1f74aa587854148bcf95834f607df`) and
`src/gengamma.h` (`e363f768ccf60786998b12972b5f80d44b036e69`), plus their
distribution, recycling and map headers. The unchanged kernel has since been
compiled with installed Rcpp 1.1.1. Four native-kernel fits, two original-coordinate
transformations and distribution/prediction fixtures are recorded in the
[generalized-gamma audit](generalized-gamma-audit.md); the full flexsurv fitting
stack is not installed.

The Prentice model allows positive or negative Q and has the exact log-normal
case at Q=0. For nonzero Q, its CDF uses a gamma variable with shape Q^-2 and
argument exp(Q*w)/Q^2, reversing the tail for negative Q. The native density
explicitly notes cancellation near Q=0. A Python port needs a stable limiting
calculation and joint shape covariance, not a fixed-shape or finite-df
approximation. The original Stacy parameterization is the positive-Q subset;
its source transformation is mu=log(scale)+log(k)/shape,
sigma=1/(shape*sqrt(k)), Q=1/sqrt(k).

The same pinned author revision contains
[`coxStrataContour.R`](https://github.com/YushuShi/survivalContour/blob/d4645f69f23fc1146c07432f576b4c40f85e1bba/R/coxStrataContour.R)
(blob `46cbd57ff949c18a934c3a94461727b191533ff7`). It predicts separately for
each fitted stratum, uses a common global continuous-covariate grid and adjustment
profile, and prepends time zero with survival one to each surface. Its histograms
use the observations in each stratum. A future port needs a shared coefficient
fit with stratum-specific risk sets and baselines, rather than independently
fitting one model per stratum. Use direct R `survfit` predictions to verify that
the helper's flattened-result reconstruction has the intended matrix ordering.

[`FGContour.R`](https://github.com/YushuShi/survivalContour/blob/d4645f69f23fc1146c07432f576b4c40f85e1bba/R/FGContour.R)
(blob `f8abb256eb13cec172cbdb480760834015ebcce1`) calls `riskRegression::predictRisk`
for the fitted cause at sorted distinct cause-event times, with zero prepended
if absent. It plots cumulative incidence, not survival. The existing CUMINC
nonparametric estimators do not substitute for this Fine–Gray regression model.

## SurvivalContour spline source lead

The same flexsurv pin supplies Royston–Parmar natural-cubic survival splines.
Five exact files were retrieved, inspected and Git-blob verified under ignored
`research/raw/flexsurv` while Luna implemented generalized gamma:

| File | Git blob |
| --- | --- |
| R/spline.R | `b46b2060dbc40da6d1390e725cbc3c753a9120c9` |
| R/survsplinek.R | `845141cb4b845d3b3db7fb76bda52f2ef3e3b23d` |
| src/splines.cpp | `8e0b8dd1dfebeb6f780e555a0befcc964aa81e87` |
| R/deriv.R | `b5475ad42cd2793c02d10ec6c2f80831a465fea7` |
| R/deriv2.R | `0fdcc37c0213f8f4c66969b4de5a683694d4c8ad` |

`flexsurvspline` models the log cumulative hazard, log cumulative odds or
negative normal survival quantile as a natural cubic spline in log time plus
covariate effects. With zero internal knots, these reduce to Weibull,
log-logistic and log-normal models respectively. The default basis is `rp`;
`splines2ns` is an optional alternative with a separate dependency. The author
SurvivalContour wrapper dispatches fitted spline objects through the existing
parametric contour helpers. No new contour formula is needed.

For unweighted right-censored data, default internal knots are equally spaced
empirical quantiles of log event times, and boundary knots are the minimum and
maximum log event times. The number `k` counts internal knots, so k=4 has six
baseline coefficients. Explicit `knots` excludes the two boundary values in
the fitting API but the density/basis APIs include them. Covariate effects
enter gamma0 by default; coefficients are effects on the selected survival
link, not log-time AFT slopes. Ancillary covariate effects and interval censoring
are additional native features that need distinct coverage tracking.

The small C++ file supplies basis and first-derivative evaluations with no
headers beyond Rcpp. R density and survival functions plus analytical
first/second derivatives are available in the retrieved sources. A small
native-kernel fitting harness has now executed without installing the
whole package: six model fits, joint information/covariance and surface
references are committed at `ee0b3b5` and described in the
[spline audit](survival-spline-audit.md). Python fitting, full joint covariance,
stable predictions and contours are implemented for all three links.

Two numerical requirements governed the implementation: evaluate the linear
tails of the natural spline without subtracting huge cubics, and verify that
its transformed cumulative hazard has positive derivative over the required
domain. Native density truncates nonpositive derivative values to zero only
at evaluated times; positivity at observed failures alone does not prove a
valid monotone survival curve between them. Several native density/log-probability
paths also exponentiate and then take logs, requiring direct log-domain
evaluation in the Python port.

## SurvivalContour random survival forest

The app cites randomForestSRC 3.2.2, pinned via its CRAN Git tag at
`b4d099e262423362a8872c13c468e6dbe2f9e9da`. The exact source bundle, algorithm
contract and executed small C/R references are recorded in the
[forest audit](random-survival-forest-audit.md). Native leaf Kaplan–Meier
survival and Nelson–Aalen hazard are averaged separately. Numeric Python fitting,
prediction and fitted-model contours are implemented with bounded storage and
sequential trees. Categorical splits, missing-value handling, out-of-bag
diagnostics and importance remain separate work.

## Interval-censored Cox source mismatch to resolve

The original SurvivalContour dispatcher advertises interval-censored Cox via
`mets::phreg(Surv(lower,upper,type="interval2") ~ ...)`. Its
`coxIntContour.R` helper (blob `26d1143fd364bc3679fcafddb260952377ef8e12` at
the author pin above) delegates prediction to `predictPhreg`.

Inspection found a material contract concern. Both
[`mets` 1.3.3](https://github.com/cran/mets/tree/841d9a66ed6dcf7ad35c1df9948a2567f1702fd2)
(`R/phreg.R` blob `bb95b6f5bbd606ac3c008bc0113cb803cb04a7ff`) and
[`mets` 1.3.12](https://github.com/cran/mets/tree/4177a01298fce3e0019d94ff073c1f5b4a558c21)
(blob `93553619a9f76ab963a4c292f8422a1a0c0f3883`) dispatch any three-column
Surv response as `entry=Y[,1]`, `exit=Y[,2]`, `status=Y[,3]` into a counting-
process Cox partial-likelihood backend, without an interval-type branch.
The 1.3.3 source is dated 2023-12-04 and GPL >=2; 1.3.12 is dated July 2026 and
declares Apache 2.0. Exact source bytes are saved separately under ignored
`research/raw/mets-1.3.3` and `research/raw/mets`.

Installed R survival 3.6-4 confirmed that interval2 data for `(1,2]`, right
censoring at 2 and an exact event at 3 produces rows `(1,2,3)`, `(2,1,0)` and
`(3,1,1)` with `type="interval"`. Those are interval status codes and placeholder
second-column values, not valid start/stop/event observations. Merely accepting
this matrix in `phreg` is not evidence of a correct interval-censored model.
The complete native fit and the app's deployed backend have not been executed;
verify the source mismatch before claiming native interval-Cox parity. Do not
port that interpretation as a valid interval likelihood.

A method/reference lead is
[`icenReg` 2.0.16](https://github.com/cran/icenReg/tree/26fadac37c6b54dd0e29c91c2bf07942ae120356),
dated 2024-01-13, LGPL >=2.0 and <3. Its `R/ic_sp.R` (blob
`0a51f8e92a04249ded93bd6c7476bc6b24d28275`) explicitly supports interval2
proportional-hazards/proportional-odds likelihoods, Newton regression updates
and iterative-convex-minorant baseline updates. DESCRIPTION and that wrapper
are saved and blob-verified under ignored `research/raw/icenReg`. The full
native PH core and maximal-intersection builder now run through a small
Rcpp/Eigen harness without installing the package. Five reference fits cover
mixed censoring, weights, no covariates, current status and tied exact/right
observations; see [the interval audit](interval-survival-audit.md). The native
fixed endpoint perturbation must be replaced by exact inclusivity semantics,
and within-support survival identification bounds must not be mislabeled as
confidence intervals. Python implementation is underway in the single Luna
worker; completed coverage is not yet claimed.

## Fine–Gray executable reference lead

For the next competing-risk model, the CRAN mirror
[`cmprsk` 2.2-12](https://github.com/cran/cmprsk/tree/f81411e1e3f57822796bae2a6870657e455362e9)
was pinned at `f81411e1e3f57822796bae2a6870657e455362e9` (2024-05-20).
The original author is Robert Gray and DESCRIPTION declares GPL >=2. Exact
source-only downloads were saved under ignored `research/raw/cmprsk` and
verified against their Git blob hashes:

| File | Git blob |
| --- | --- |
| DESCRIPTION | `5ab1f6ef57c855b343a4e6020d77e74d5f63beed` |
| R/cmprsk.R | `759528f90b7c7706e421518d21edab69b893046e` |
| src/crr.f | `f2f473eb0e881ad0e565c1cc430521bf42e562ec` |

The `crr` wrapper supports fixed covariates, covariates multiplied by supplied
time functions, and separate censoring-distribution groups. It computes
left-limit censoring Kaplan–Meier estimates and passes them to the Fortran
likelihood/score/information routines. Prior competing failures remain in the
subdistribution risk set with censoring-survival-ratio weights; prior target
events and censors do not. Tied target events use a common risk denominator.
This is not ordinary cause-specific Cox regression.

`crrfsv`, `crrf`, `crrvv`, `crrsr` and `crrfit` supply the objective/derivatives,
variance ingredients, score residuals and baseline increments. The R wrapper
forms a sandwich covariance from `crrvv`, rather than substituting inverse
information. `predict.crr` transforms cumulative subdistribution hazard to
incidence with `1-exp(-H)`; Python should use the stable equivalent `-expm1(-H)`.
The unchanged Fortran source has since been compiled using the existing local
compiler. Six native-reference models and incidence predictions, including an
audit of the original wrapper's time-zero censoring defect, are now recorded in
[the Fine–Gray audit](fine-gray-audit.md). The larger formula/UI dependency stack
is not installed. Method coverage is recorded by the executable catalog when
the Python implementation is integrated.

## BLESS model coefficients and baseline survival

The [primary paper](https://doi.org/10.1016/j.chest.2021.03.059), Molina et al.,
CHEST 160(3), 1075–1094 (2021), is indexed as PMID 33852918 and PMC8449006.
Its indexed text identifies a general interaction model and disease-specific
lung, breast and hematologic Cox models. It explicitly locates reference-patient
baseline survival and prediction instructions in e-Appendix 1/e-Tables 1–8.
Those baselines are essential: rounded hazard ratios from the main tables alone
cannot reproduce absolute survival probabilities.

The indexed supplement is named `mmc1.pdf` and described as about 1.2 MB.
The live PMC page returned a browser challenge, the publisher full-text endpoint
returned 403, and the app's instruction/variable PDFs returned reader cache
misses. Europe PMC and the constructed Elsevier supplement endpoint were also
unavailable to the reader. No supplemental coefficients, baseline curves or
native predictions were retrieved, and no clinical model values were guessed.
The [PubMed record](https://pubmed.ncbi.nlm.nih.gov/33852918/) provides a stable
primary-publication identifier for subsequent retrieval.

## WFMM and FLECS90

WFMM's [archived institutional page](https://bioinformatics.mdanderson.org/public-software/archive/wfmm/)
describes the wavelet functional mixed-model MCMC program and an older C++
executable interface. The [Morris–Carroll primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2744105/)
and [official user guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/wfmm_Users_Guide.pdf)
are indexed. These are leads for a method audit, not evidence that its full
posterior sampler or the catalog's newer version 3.1 is covered. Existing
Pinnacle wavelet code is a potential numerical primitive, not a replacement
for the functional mixed model.

The FLECS90 detail page remains unavailable to the reader. Search results point
to a FLECS-to-Fortran-90 translator; that secondary description is insufficient
to implement its grammar. Retrieve the catalog archive or another primary
source before deciding its full parser/translation contract.

## ComPAS platform design

Entry 140 is Tang, Shen and Yuan's *ComPAS: A Bayesian drug combination platform
trial design with adaptive shrinkage*, Statistics in Medicine 38(7):1120–1134,
[DOI 10.1002/sim.8026](https://doi.org/10.1002/sim.8026),
[PMID 30419609](https://pubmed.ncbi.nlm.nih.gov/30419609/). The
[authors' software site](https://www.trialdesign.org/) links this same design.
The primary abstract describes adaptive borrowing through Bayesian model
selection and hierarchical models, with dropping, graduating and adding
combinations. A generic independent beta-binomial platform would not implement
that contract.

On 2026-09-27, both forms of the institutional app URL and the publisher full-text
URL were unavailable through the reader. The author-site launch link exposed
only a JavaScript shell. A targeted GitHub source search for the DOI returned
no matches. The exact likelihood, model-selection priors, posterior algorithm,
operating rules and native references still need retrieval before implementation;
entry 140 remains pending.
