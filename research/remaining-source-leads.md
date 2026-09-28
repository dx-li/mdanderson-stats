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
`2aae4c8ac56823d0eac30c1a9ad654ac599b5938`. Generalized-gamma and spline families
remain separate work. Its small native generalized-gamma implementation and
headers are saved as exact source bytes under ignored `research/raw/flexsurv`:
`src/gengamma.cpp` (`34c799553ec1f74aa587854148bcf95834f607df`) and
`src/gengamma.h` (`e363f768ccf60786998b12972b5f80d44b036e69`), plus their
distribution, recycling and map headers. Rcpp 1.1.1 is already installed,
so a future small serial reference compilation may avoid installing the
full flexsurv dependency stack. No compilation is claimed at this checkpoint.

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
