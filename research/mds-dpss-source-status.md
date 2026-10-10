# MDS-DPSS source status

Catalog entry 170 remains **pending**. On 2026-10-04, the institutional
[Ziyi Li laboratory publication list](https://www.mdanderson.org/research/departments-labs-institutes/labs/ziyi-li-laboratory/publications.html)
listed Yue Lyu and colleagues' *Dynamic Prognostic Scoring System for
Myelodysplastic Syndromes* as submitted, with Ziyi Li and Guillermo
Garcia-Manero as corresponding authors. That entry has no manuscript,
supplement or code link. The linked laboratory resources page did not supply
an MDS-DPSS implementation.

This establishes a primary bibliographic lead, not a prediction algorithm.
The inspected pages do not provide fitted parameters, predictor coding,
time-updating rules, risk-group cutoffs or the calibration needed to reproduce
the calculator. A bounded title/name search did not recover those details.
This is a statement about inspected evidence, not proof that no other source
exists.

MDS-HOPE (catalog 171), older MDS prognostic scores and myelofibrosis DIPSS are
distinct models and cannot substitute for MDS-DPSS. No score or survival
probability has been inferred from them. Completing this entry requires an
independently available author implementation or an implementation-complete
manuscript and fitted model specification.

October 10 update: the public MDS-DPSS interface is reachable and identifies
PID 1183, version 1.0.0, updated November 21, 2025. Its document title says
MDS-HOPE, while its visible UI says MDS-DPSS; this does not establish equivalence
with the separate MDS-HOPE entry. Inputs include five diagnosis-time categories,
blasts, hemoglobin, platelets, neutrophils, age, cytogenetics and ten gene mutation
statuses. DPSS/IPSS-R/IPSS-M panels are present, but the exact dynamic regression,
calibration and outputs were not recovered. Entry 170 remains pending.
