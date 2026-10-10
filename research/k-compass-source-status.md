# K-COMPASS source status

Catalog entry 169 remains pending. A bounded independent-source review on
October 4, 2026 identified the peer-reviewed report by Tang and colleagues,
*KIM-1 and Circulating Tumor DNA Are Prognostic Markers for Oligometastatic
Clear-cell Renal Cell Carcinoma: Development of a Multivariable Prognostic Model*,
European Urology 89(4):313–317 (April 2026),
[DOI 10.1016/j.eururo.2026.01.004](https://doi.org/10.1016/j.eururo.2026.01.004).
The [institutional publication record](https://mdanderson.elsevierpure.com/en/publications/kim-1-and-circulating-tumor-dna-are-prognostic-markers-for-oligom/)
was readable; it supplies an abstract, not the fitted prediction equation.

The [ASCO GU 2026 abstract 537](https://ascopubs.org/doi/10.1200/JCO.2026.44.7_suppl.537)
identifies systemic therapy–free survival after metastasis-directed therapy as
the modeled outcome. Elastic-net selection precedes a Weibull refit using
baseline KIM-1, baseline ctDNA residual-disease status, prior systemic-therapy
lines, performance status, metastasis count and time from diagnosis to metastasis.
Its reported biomarker association hazard ratios do not specify the final
six-variable prediction equation.

The inspected records do not supply the complete final coefficients, covariate
coding/transformations, Weibull intercept and shape, or uncertainty calculation.
These are required to reproduce an individual survival prediction. A generic
Weibull fitter or a score built from the abstract's association hazard ratios
would not implement this calculator. No predictor is claimed here.

The DOI reader could not retrieve the full article; the PubMed landing-page
read returned no article content. Neither route was retried through an alternate
transport. The restricted institutional application was not requested, and no
native numeric outputs or author code were recovered in this review.

October 10 update: the public app now serves PID 1178 v1.4.2.0 (April 15,
2026). Its default illustrative profile produces an estimated STFS of 52 months,
95% interval 37–77 and Intermediate risk, with a warning that omitting KIM-1 and
ctDNA reduces model robustness. This is a rounded public-app observation, not
recovery of the exact regression/intercept/shape/covariance. Entry 169 remains
pending; clinical predictions are not reconstructed from fitted screenshot curves.
