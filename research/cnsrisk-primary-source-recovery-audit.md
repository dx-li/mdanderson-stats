# CNSRISK primary-source recovery audit

Audit date: 2026-10-04. This bounded pass looked for the final fitted model
behind CNSRISK #161, beyond the previously cached 2021 risk-factor abstract.
It found a fuller 2023 ASCO abstract, but not an implementation-complete model.

## Newly established from the primary abstract

Hasanov et al., ASCO 2023 abstract 2012, *External validation and nomogram
for risk factors of CNS metastasis in patients with clinically localized
melanoma*, is available in the official ASCO Central Nervous System Tumors
proceedings PDF (abstract pages 10–11):
https://s3.amazonaws.com/files.oncologymeetings.org/prod/s3fs-public/2023-05/AM23-Central-Nervous-System-Tumors.pdf

The abstract reports AJCC 8th edition stage I–II patients diagnosed during
1998–2014 in the MDA and Melanoma Institute Australia cohorts (4,332 and 9,610
patients). Death is included as a competing event; patients alive without CNS
metastasis at last follow-up are censored. The initial MDA model considered
gender, primary site, melanoma subtype, Breslow thickness, ulceration, mitotic
rate, lymphovascular invasion and perineural invasion. After external
validation, the final reduced model retained primary site, subtype, Breslow
thickness and mitotic rate. It predicts cumulative incidence at 2, 5 and 10
years. Reported O/P calibration ratios were 0.96, 0.97 and 1.03; AUCs were
0.75, 0.71 and 0.69, respectively.

Melanoma Institute Australia's institutional ASCO23 announcement confirms
the authors and presentation but does not add model parameters:
https://melanoma.org.au/news/mia-asco23/

## Remaining model contract

Neither source supplies the fitted coefficient vector, complete categorical
coding/reference levels, Breslow and mitotic transformations or cutpoints,
Fine–Gray intercept/baseline subdistribution hazard, or the exact conversion
from its fitted quantities to the 2-, 5- and 10-year individual cumulative
incidence outputs. The abstract says that a nomogram and risk calculator
would be available, but those statements are not the prediction equations.
The published 2021 hazard ratios likewise do not identify the final
externally validated model.

Therefore the actual CNSRISK calculator still cannot be implemented from the
recovered primary contract. Do not substitute the earlier risk-factor model,
an ordinary Cox model, or a reconstructed baseline based only on cohort-level
incidences. The remaining blocker is exact model coefficients, coding and
baseline/incidence parameterization from the final reduced fit or an
independently available author-provided implementation.
