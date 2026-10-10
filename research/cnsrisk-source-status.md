# CNSRISK source status

Catalog entry 161 remains pending. A September 29, 2026 review found no local
CNSRISK model files. The independent primary-source lead is Hasanov et al.,
[ASCO 2023 abstract 2012](https://doi.org/10.1200/JCO.2023.41.16_suppl.2012),
also available in the meeting's [CNS proceedings](https://s3.amazonaws.com/files.oncologymeetings.org/prod/s3fs-public/2023-05/AM23-Central-Nervous-System-Tumors.pdf).

The abstract describes a validated model for CNS metastasis in initially
localized stage I–II melanoma. It identifies primary tumor site, subtype,
Breslow thickness and mitotic rate as the reduced predictors, with validation
at two, five and ten years and death handled as a competing risk. It does not
provide the fitted coefficients, complete predictor coding and transformations,
or baseline risk function required to calculate individual predictions.

The related [2021 abstract 9580](https://doi.org/10.1200/JCO.2021.39.15_suppl.9580)
reports an earlier risk-factor analysis. Its hazard ratios cannot substitute
for the final validated model or its missing baseline prediction function.
No generic model has been presented as the CNSRISK calculator. Recovery of the
actual fitted model or a complete mathematical specification remains necessary.

The denied application domain was not accessed during this review.

The [October 4 primary-source audit](cnsrisk-primary-source-recovery-audit.md)
records the development/validation cohorts and published calibration and
discrimination summaries. These do not supply the missing individual-risk
equation; the implementation status is unchanged.

October 10 update: the public app (PID 1138, version 1.0.0.0, updated May 31,
2023) returned cumulative risks .02/.06/.10 at 2/5/10 years for scalp,
superficial-spreading melanoma, Breslow 1.43 and mitotic rate 0–1. These rounded
outputs confirm the application is live, but do not recover the exact
competing-risk coefficients/baselines. Entry 161 remains pending.
