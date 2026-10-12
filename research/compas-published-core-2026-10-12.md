# ComPAS explicitly configured published core

The parent recovered PNOC022 v3.0 (January26,2023), pp83–95, from the official
EU trial record. Its newly authored factual extraction is preserved in
compas-pnoc022-specification-2026-10-12.json, SHA256
b7a5cef6c4a0a8114aa1abd7ba00ca1f96c98dafbca297a123401b0bd1cfe894.
The executor inspected the full extraction, reconstructed from Library text;
the original259-page protocol PDF is not redistributed. Its parent-verified
SHA256 isfa69432cb8e68abec0b206a3e0f753c7cb2b0e40827651a99fc18411e5c37e80.
The original method citation is Rui Tang, Jing Shen and Ying Yuan,
DOI10.1002/sim.8026. The source is a protocol reproduction of the hierarchy,
not independent access to that article's full text or proprietary app.

Closed declared scope: fixed design-constant variance calibration; literal
reported PNOC variance tuples; five-model binomial/logit-normal hierarchical
inference with Normal(0,variance1e6) group means; explicit borrowing groups and
model prior weights; normalized evidence and MAP selection; strict selected-model
.1/.95 decisions with explicit equality policy; numerical uncertainty holds;
immutable saved-study/reopen and self-contained JSON reports.

Numerical integration retains all variance-dependent Gaussian constants and
binomial coefficients. The group mean is integrated over the real line; no
improper-flat-prior substitution is made. Conditional logit integration uses
stable mode-centered density ratios, adaptive quadrature and a concavity-based
12-prior-SD truncation bound. Explicit transition splits prevent narrow tail
boundaries being missed under the diffuse mean prior. Reported quadrature errors
are estimates, not certified bounds or clinical credible intervals. Failures
raise; unresolved model or cutoff comparisons return numerical_hold.

Independent two-arm references analytically eliminate the group mean and use
mean/contrast coordinates, distinct from production conditional integration.
Fifteen reference cases agree within5e-13 in evidence/tails. A second independent
implementation uses the marginal multivariate-normal logit prior and48/80-node
Laplace-centered Gaussian tensor quadrature: five three-arm evidence references
agree, with refinement differences below7e-12. Reference outcomes are synthetic,
not claimed original-app outputs. Source calibration, no-data analytic cases,
complementarity, factorization, arm permutation, adding an unobserved arm,
input ownership and failure/uncertainty paths are tested.

Catalog140 is partial, not implemented. Remaining original/default or platform
contracts: model prior weights; intended borrowing-group interpretation;
replacement ordering; data retention after arm closure; simultaneous/sequential
recomputation; platform-wide maximum enrollment; final selection of continuing
arms; exact minimization randomization; optional Shotgun-style cross-cohort
borrowing. No500-run table is represented as a deterministic test. Explicit
PNOC variances and .95 equation cutoff preserve the documented contradictions;
no inferred corrections are silently substituted.

License notices remain unchanged. Mathematical validation, source compatibility,
full application completion and clinical validity remain distinct claims.
Final-code full-suite and installed-artifact acceptance is recorded separately.

Final acceptance:73,227 full tests passed in986.11 seconds, both extras and
warnings-as-errors, with all767 production Python files unchanged. A fresh
wheel/extras environment passed755 focused tests and28 plotting/TIFF tests;
9,347 broader installed-wheel checks passed. Both ComPAS and BLESS examples ran
outside the checkout. Both independent reference fixtures regenerated
byte-identically. Ruff, formatting, mypy, wheel/sdist build and the771-file
package-byte audit passed;29 licensing files remain and raw assets are excluded.
See compas-published-validation-2026-10-12.json. Artifact hashes are retained in
the final preservation commit message. No full ComPAS platform promotion is made.
