# Pending risk-calculator sources

Bounded primary-source review on September 28–29, 2026 did not establish the full
prediction equations for these three catalog entries. They remain pending;
variable lists and rounded hazard ratios are insufficient to reproduce a model.

## K-COMPASS, entry 169

The [official application](https://biostatistics.mdanderson.org/shinyapps/K-COMPASS/)
identifies version 1.4.2.0, updated April 15, 2026, and authors Chad Tang, Alex
Sherry, Peng Yang and Pavlos Msaouel. Its target population is clear-cell renal
cell carcinoma with at most five metastases after metastasis-directed therapy,
without concurrent systemic therapy.

The identified primary publication is Tang et al., *European Urology* 89
(2026), 313–317, [DOI 10.1016/j.eururo.2026.01.004](https://doi.org/10.1016/j.eururo.2026.01.004).
Accessible descriptions identify Weibull regression with four clinical variables
plus baseline KIM-1 and circulating tumor DNA. Exact coefficients, transformations,
and intercept/scale or equivalent baseline survival were not retrieved. The
publication supplement or author model code is the next useful source.

## MDS-DPSS, entry 170

A September 29 static retrieval of the MDS-DPSS route returned a page titled
MDS-HOPE (PID 1183, version 1.0.0, updated November 21, 2025). This route alias
does not establish an independently specified dynamic DPSS model. No exact-title
primary paper, model code or coefficient table was found in the bounded search.
The time-updating rules, predictor coding and risk outputs remain unverified.
Do not substitute WPSS, IPSS-R or another MD Anderson score based on name alone.

## RMC-COMPASS, entry 174

The [official application](https://biostatistics.mdanderson.org/shinyapps/RMC-COMPASS/)
identifies version 1.0.1.0, PID 1189, updated May 11, 2026, and authors Stepan M.
Esagian, Peng Yang, Alexander D. Sherry and Pavlos Msaouel. Its visible inputs
include ECOG, ten baseline metastatic-site indicators, ANC and ALC for NLR,
calcium and albumin, and optional TP53 status.

The page distinguishes a recommended clinical model and an optional TP53 model.
It reports more than 50% missing TP53 values, no external validation and clamping
outside the observed data range. The interactive model-details section was not
available in the static page. Exact coefficients, predictor transformations,
baseline survival and clamping limits were not retrieved. A subsequent bounded
publisher search identified Esagian et al., *Journal of Clinical Oncology*
44(16_suppl), e16535 (2026),
[DOI 10.1200/JCO.2026.44.16_suppl.e16535](https://doi.org/10.1200/JCO.2026.44.16_suppl.e16535),
“Development and internal validation of a clinical prognostic model for
SMARCB1-deficient renal medullary carcinoma (RMC).” The indexed publisher
abstract establishes the matching author group and model-development study,
but does not supply an exact fitted parameter set. Direct article and PDF
retrieval returned HTTP 403 on September 29. This is a primary-source lead,
not enough evidence to implement predictions. Do not assume a corrected-calcium
formula or reuse coefficients from a different renal-cancer model.

## Separate MDS-HOPE recovery and access limits

For entry 171, the publisher-hosted supplement yielded Equation S1 and six-group
cutoffs; see [its audit](mds-hope-source-status.md). The missing cytogenetic
encoding and original standardization constants were not recovered. A saved
browser permission blocked interactive access to the MD Anderson domain on
September 29. No alternate transport was used to bypass that denial; the
implementation uses the previously retrieved publisher supplement and local
source records. Static K-COMPASS and RMC-COMPASS pages retrieved before the
denial still did not expose their exact fitted parameter sets.
