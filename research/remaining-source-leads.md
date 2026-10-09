# Remaining source leads

The October 8 catalog checkpoint is 90 implemented, 42 partial and six pending.
The [source-recovery record](source-recovery-2026-10-08.md) documents recovered
archives, current blockers and validation. Historical reader failures below
are not evidence that direct archive downloads are still unavailable.

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
sequential trees. Optional out-of-bag curves and native-convention concordance
error are now implemented, along with explicit permutation importance and
independent native-kernel references. Categorical splits, missing-value
handling, alternative split rules and anti-split importance remain separate
work; see the [OOB/importance guide](../docs/random-survival-oob.md).

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
confidence intervals. The Python PH fit, predictions and contour workflow now
pass five native reference cases, independent score/constraint checks and unit
rescaling checks; see the audit and `docs/interval-survival.md`. Shared-coefficient
stratified interval fits are now implemented with independent baseline supports
and joint likelihood fitting; see `docs/interval-survival-stratified.md` and
`research/interval-stratified-audit.md`. Ordinary-PH coefficient bootstrapping
now follows the native weighted resample/frequency-weight/covariance contract;
see `research/interval-survival-bootstrap-audit.md`. Stratified/cluster
resampling remains separate. Native `survCIs` explicitly excludes `ic_sp`, so
baseline confidence bands must not be inferred from that coefficient bootstrap.

## Interval-censored competing-risk regression

The author `FGIntContour.R` at the SurvivalContour pin has blob
`5dea1bd0dcc52f31162616e08a70183120851cc0`. It calls `predict.ciregic` for cause 1
on 50 times from zero to `trainModel$tms[2]`. This is a different model from
the ordinary right-censored Fine–Gray regression already implemented.

[`intccr` 3.0.4](https://github.com/cran/intccr/tree/252644c0d347a663ea5d7bef88fa2ab04f114b1e),
dated 2022-05-09, by Giorgos Bakoyannis and Jun Park, is GPL >=2. It fits
two-cause generalized odds-rate transformation models by a constrained B-spline
sieve likelihood. Each cause has its own link parameter; zero gives Fine–Gray
and one proportional odds. Event status 0 denotes right censoring and ignores
the supplied upper endpoint; statuses 1 and 2 select the observed cause.
Both monotonicity and joint probability constraints require verification.
Native default uncertainty with `nboot=0` uses a least-squares method, rather
than a generic inverse Hessian.

Retrieved and blob-verified under ignored `research/raw/intccr`:
`R/ciregic.R` (`e1f2f6e29d84e6f520e0482adfdcc0a48c8f9b90`),
`R/bssmle.R` (`d122c3ac22a446bfb8071b720c9df962de188ebe`), and
`R/dataprep.R` (`371ac55900e8434a264c557e1933b749f3918576`). Three native fits
now supply parameter, covariance and prediction references. Corrected
constraint-Jacobian runs demonstrate sensitivity to the finite starting-boundary
approximation; neither set is treated as a certified maximum-likelihood target.
The Python likelihood, constrained fitting, least-squares covariance,
cause-specific prediction and contour workflow are now implemented. Native
parameter/prediction references, direct probability and derivative checks,
unit-rescaling checks and the four focused tests are recorded in
`research/interval-competing-risk-audit.md`; bootstrap uncertainty remains open.
Repeated-visit conversion now implements the source's first-event rule with
corrected chronological sorting and consistent first-visit event retention.
Native defect fixtures and source-row provenance are recorded in
`research/interval-competing-risk-data-audit.md`.

The [implementation contract](interval-competing-risk-audit.md) records the
additional response, spline, initialization and least-squares covariance
sources, and source-level derivative discrepancies requiring numerical review.
Pure-R optimizer dependencies are available for a source-only reference run;
no package installation is required for that approach.

The same pinned upstream tree also supplies `R/bssmle_se.R`, blob
`c923ec3cd71842c4e4895d34905c635381ade5d6`, now saved in the ignored source
cache. For `nboot > 1`, it resamples rows uniformly with replacement, rebuilds
the sample's empirical spline knots on each refit, extracts the two regression
blocks, drops NA coefficient rows and takes their sample covariance. It does
not catch refit errors: a resample missing either cause appears to abort through
`Surv2`, rather than enter the failed-optimizer count. Native `nboot=1` returns
a coefficient vector as `Sigma`, not a covariance. Intercept-only input is
internally represented by a zero column and two nominal zero slopes. These
source edge cases require explicit Python conventions and a numerical reference
before implementation; no competing-risk bootstrap is claimed here.

## Neural survival models already use a Python backend

The author `pycoxContour.R` (blob `f924c363e959e7e3b9192bb801dbc37c767a2159`)
accepts a fitted model and calls `survivalmodels::predict(...,type="survival")`.
The inspected [`survivalmodels` 0.1.191](https://github.com/cran/survivalmodels/tree/d8a6a4a36368fb57253e0ebf3f0f32e05f05c561)
wrapper `R/helpers_pycox.R` (blob `50601edd5d1aad9bb9e928f0780ff481b48de670`)
delegates to Python `pycox` for Cox-Time, DeepSurv, DeepHitSingle,
LogisticHazard and PCHazard. Its prediction path uses `predict_surv_df`, with
model-specific label transforms and optional interpolation. Training and
preprocessing contracts must be preserved when choosing an integration.

The [`pycox` source](https://github.com/havakv/pycox/tree/3eccdd7fd9844a060f50fdcc315659f33a2d2dc1)
is pinned at `3eccdd7fd9844a060f50fdcc315659f33a2d2dc1`; setup.py declares
version 0.3.0 and BSD licensing. These models already have an open Python
backend, so inspect reuse before writing another training stack. PyTorch,
pycox and torchtuples are absent from the project environment. No dependencies
were installed and no neural training or prediction has been validated.

Further inspection of the pinned `helpers_pycox.R` identifies contracts for
the eventual adapter. Cox-Time fits its time transform on training outcomes;
DeepHitSingle, LogisticHazard and PCHazard fit different discretization label
transforms. Any validation outcomes are transformed with that fitted object,
never used to refit cutpoints. PCHazard retains an additional within-interval
exposure target. The wrapper casts durations to integer before discretization;
a Python interface must decide and document whether to reproduce that loss of
fractional time precision rather than doing it accidentally.

Prediction uses batched `predict_surv_df`; Cox-Time and DeepSurv first compute
baseline hazards. PCHazard uses its own subdivision setting, while the two
other discrete models can use constant-hazard or constant-density interpolation.
The wrapper rounds survival to four decimal places and then calls `fill_na`.
Those presentation/recovery steps are not validated numerical behavior for a
Python backend and should not silently discard precision or conceal invalid
curves. The contour helper receives the model's prediction-time index, instead
of constructing the 50-point time grid used by the interval competing-risk
helper. This remains source inspection only, with no new method coverage claim.

The following Python backend files at that same pin are now saved under ignored
`research/raw/pycox` and verified against their Git blob hashes:

| File | Git blob |
| --- | --- |
| pycox/models/cox_time.py | `e663d678377ad74e1991e1f6408c86cff7cd1675` |
| pycox/models/logistic_hazard.py | `045364dce4a81d1d98a7c8c895304b7957e31cd9` |
| pycox/models/pc_hazard.py | `362685980282d415872dbf5a48fa41380aaad318` |

Cox-Time incorporates time as a network input and computes separate event-time
risk-set denominators; ordinary proportional Cox prediction is not equivalent.
Its cumulative-hazard predictor allocates a time-by-profile matrix, so a bounded
adapter must preflight the complete output as well as batch neural evaluation.
PCHazard has one fewer network output than cutpoints and predicts via positive
softplus hazards. LogisticHazard's default survival path adds `1e-7` to each
conditional survival factor before taking logs; zero hazards can therefore
produce factors above one. A future reuse adapter needs an explicitly verified
probability-preserving prediction path, not blind acceptance of every backend
default. No backend execution has been performed.

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

October 8 update: the official BLESS instruction and variable-definition PDFs
now download successfully and define input units/coding. Neither contains
fitted coefficients or baseline curves. The app shell is also accessible.
The old PDF reader failures below are superseded; the essential supplementary
fit/baseline evidence remains unavailable in the current runtime. Added
publication hosts are saved in the environment draft for later access testing.

Further October 8 recheck: the full PMC article is readable at
`https://pmc.ncbi.nlm.nih.gov/articles/PMC8449006/?pdf=render` (HTML, not PDF).
Its exact appendix link is
`https://pmc.ncbi.nlm.nih.gov/articles/instance/8449006/bin/mmc1.pdf`.
That link returns a download gateway or browser challenge instead of a PDF.
The normal Chromium HTTPS browser did not obtain the attachment. Elsevier
and Europe PMC still returned HTTP 403. This advances the source location,
but does not recover baseline survival or fitted constants.

The indexed supplement is named `mmc1.pdf` and described as about 1.2 MB.
The live PMC page returned a browser challenge, the publisher full-text endpoint
returned 403, and the app's instruction/variable PDFs returned reader cache
misses. Europe PMC and the constructed Elsevier supplement endpoint were also
unavailable to the reader. No supplemental coefficients, baseline curves or
native predictions were retrieved, and no clinical model values were guessed.
The [PubMed record](https://pubmed.ncbi.nlm.nih.gov/33852918/) provides a stable
primary-publication identifier for subsequent retrieval.

A September 29 direct retrieval check resolved the publisher article identifier
to `S0012369221006760`, but its `1-s2.0-S0012369221006760-mmc1.pdf` supplement
endpoint returned HTTP 403. Europe PMC's full-text XML endpoint returned HTTP
500 and the NCBI open-access lookup returned HTTP 404. The PMC reader still
showed a browser challenge. These attempts did not recover the required model
constants or baseline survival. Do not repeat them without new access or a new
source lead.

## Recovered legacy sources: WFMM, FLECS90, Multc99 and SYNERGY

WFMM's [archived institutional page](https://bioinformatics.mdanderson.org/public-software/archive/wfmm/)
describes the wavelet functional mixed-model MCMC program and an older C++
executable interface. The [Morris–Carroll primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2744105/)
and [official user guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/wfmm_Users_Guide.pdf)
are indexed. These are leads for a method audit, not evidence that its full
posterior sampler or the catalog's newer version 3.1 is covered. Existing
Pinnacle wavelet code is a potential numerical primitive, not a replacement
for the functional mixed model.

October 8 update: the catalog detail route still returns HTTP 500, but the
institutional FLECS90_V1.tar.gz archive was recovered successfully. Its
public-domain C source defines the translator grammar. The Python translation,
file/CLI workflows and all four output options are now implemented and
validated against 144 C outputs and GNU Fortran execution; see the
[FLECS90 guide](../docs/flecs90.md) and [audit](flecs90-audit.md). Entry 43 is
implemented. The unsupported secondary description is no longer the source basis.

WFMM's original Linux bundle and example now also download successfully.
Eight synthetic native initializations verify the inverse-gamma mapping, now
implemented in Python; see [the native-prior audit](wfmm-native-prior-audit.md).
The native MOM/profile optimizer, automatic proposals and other transform/file
workflows remain distinct gaps. The pancreatic example's fitted workflow has
not been rerun.

Multc99's original C source now defines and validates the general-outcome,
historical-mixture, calibration, trial and saved-study workflows; entry #3 is
implemented. See [the audit](multc99-audit.md) and retained licensing terms.
The separate Multc Lean MSI is recovered, but its timing backend is native
Windows machine code and the time-law/clock contract remains unverified.

SYNERGY's original S-PLUS/R archive is recovered. Its four parametric models
now provide fits, parameter inference and figures, checked against the author
kernels and nine R fits. See [the audit](synergy-parametric-audit.md). The
separate semiparametric bootstrap interval convention remains unresolved.

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

October 8: direct app-shell and official flowchart retrieval succeeded.
The flowchart names the adaptive borrowing/model choices but does not give
the complete likelihood, priors or posterior algorithm. It does not resolve
the missing statistical contract.

## CNSRISK primary-publication lead

The publisher's [ASCO 2023 CNS proceedings, abstract 2012](https://s3.amazonaws.com/files.oncologymeetings.org/prod/s3fs-public/2023-05/AM23-Central-Nervous-System-Tumors.pdf?OGy51zBP9jxHDPllGcBfS_rgINC3.W.D=)
contain Hasanov, Milton, Lo et al., *External validation and nomogram for risk
factors of CNS metastasis in patients with clinically localized melanoma*,
doi:10.1200/JCO.2023.41.16_suppl.2012. This is a candidate source for entry 161;
the deployed app's exact model version has not been confirmed.

The abstract describes competing-risk prediction with death as a competing
event, validated at two, five and ten years. It reports that the initial model
overpredicted risk and was reduced to primary tumor site, melanoma subtype,
Breslow thickness and mitotic rate. Consequently the older initial model must
not be substituted for the reduced calculator. The abstract supplies neither
exact coefficients nor baseline cumulative incidence, so it does not yet support
implementing absolute-risk predictions. No clinical model values were inferred.
