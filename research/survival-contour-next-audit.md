# Next uncovered software: SurvivalContour

Catalog entry 166 remains pending. The
[official app](https://biostatistics.mdanderson.org/shinyapps/survivalContour/)
was read on 2026-09-27: version 1.0.2, updated 2025-05-13. It exposes survival
or cumulative-incidence surfaces over time and one continuous covariate,
adjustment for other covariates, two-/three-dimensional plots and selected
curve summaries. Backends include Cox, parametric/spline models, Fine–Gray
and random survival forests. Its linked author package also supports neural
survival models and interval-censored workflows. These are distinct coverage
requirements, not interchangeable model names.

The app links [YushuShi/survivalContour](https://github.com/YushuShi/survivalContour).
GitHub reads pinned commit `d4645f69f23fc1146c07432f576b4c40f85e1bba`
(2025-07-21). DESCRIPTION reports version 1.0 and GPL >=2. Five source-only
files are saved under ignored `research/raw/survivalContour`; their exact Git
blob hashes were verified after saving:

| File | Git blob |
| --- | --- |
| DESCRIPTION | `88f95b030687159b8fb6980a50b9cb3daadd8dc1` |
| R/coxContour.R | `56dfb1dff64cc056d843ebef20de837c7a04f653` |
| R/cox3DContour.R | `afbc7ce204b93f4d0602be81b6f1bc1b3d362ddd` |
| R/paraContour.R | `39e141a400d440097946571bf15eba3c6061b6cc` |
| R/contourPart.R | `e050f0751a59856cef631978ff128472b766cde3` |

## Source contracts and proposed first port

The Cox helper builds a grid of 30 values evenly spaced between the empirical
2.5th and 97.5th percentiles. Other numeric columns default to their means;
categorical columns default to their most frequent level. An explicit profile
can replace those defaults. It calls `survival::survfit` on an already fitted
Cox model and passes a covariate-by-time probability matrix to the contour
renderer. Its 3D variant also passes lower/upper prediction limits. The
parametric helper uses sorted unique event times, prepending zero when needed,
and requests survival predictions with intervals.

There is an existing bounded `survan_cox` fit and stable risk-set likelihood in
this Python package. Audit these before adding another fitter. Its ties are
Breslow, while an unconstrained R `coxph` call defaults to Efron; independent
references must specify the matching convention. `survan_baseline` implements
Kalbfleisch–Prentice steps and must not be silently substituted for the
exponential cumulative-hazard baseline used by the Cox prediction workflow.

A useful first port can provide bounded Cox prediction surfaces, explicit
adjustment profiles, returned grid data and optional contour plotting through
the existing plotting extra. Numerical prediction intervals need a dedicated
source/reference audit before being claimed. Stratification, interval censoring,
parametric/spline models, Fine–Gray, forests, neural models and native plot
layouts remain separate work. No implementation, installation or numerical
job was performed for this scouting record.

Additional pending-entry scouting: MDS-DPSS and the FLECS90 detail page returned
reader errors. The BLESS landing page loaded but its linked instruction and
variable-definition PDFs returned cache misses. No model coefficients or
clinical calculator equivalence were inferred from those pages.
