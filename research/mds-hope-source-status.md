# MDS-HOPE score source audit

Catalog entry 171 is now **partial**: publisher Supplemental Equation S1 is
recovered and its raw Cox linear predictor is implemented. Primary material is
the publisher supplement `mds-hope-supplement.docx` (SHA-256
`ff9c43b8fa47aece229e999c415b118d514175576a6215abd391a20757a5cdfd`), read with
its extracted text and checked against the independent equation audit.

Eq. S1 uses coefficients .025 age, -.067 neutropenia, +.160 anemia, +.021
thrombocytopenia, +.051 marrow blasts, +.284 cytogenetic score, -.294 SF3B1,
+.347 EZH2, +.345 TP53, +.681 KRAS and +1.094 PTPN11. Source transforms are
neutropenia=-ANC, anemia=-hemoglobin and thrombocytopenia=-platelets/10. Thus
raw-unit contributions are +.067 ANC, -.160 hemoglobin and -.0021 platelets.
TP53 is 0/1/2 for wild type/single-hit/multi-hit; the other four retained genes
are binary indicators.

The supplement does not specify the numeric mapping for the five cytogenetic
categories or the training linear-predictor mean and SD. Callers must supply an
already-encoded numeric cytogenetic score. Six-group cutoffs are implemented
only for already-standardized scores or with explicit caller-supplied center
and SD. No cytogenetic parser, missing-mutation default, cohort re-standardizing,
baseline survival, absolute survival probability, or deployed-app equivalence
is inferred. These remain outside the recovered source contract.
