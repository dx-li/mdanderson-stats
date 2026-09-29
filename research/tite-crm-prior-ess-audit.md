# TITE-CRM prior-information ESS audit

## Scope and source contract

This implementation covers the scalar-start branch of `dfcrm::onetite` and the
TITE branch of BayesESS `essCRM`. It does not claim an exact native RNG stream,
full native application, two-stage initial design, or cohort-delay support.

The working model is `p(d, beta) = d**exp(beta)` with a normal prior on beta.
The scalar-start source starts at dose 1. At each arrival it updates using
elapsed follow-up, uses the linear TITE weight `min(followup / obswin, 1)` for
non-DLTs, gives observed DLTs weight one, recommends the closest skeleton
probability to target, and limits an upward move to one dose. The final MTD
recommendation uses the complete generated outcome vector and unit weights.

BayesESS hard-codes Poisson accrual in its TITE call, even though its surrounding
arguments suggest other accrual choices. Its ESS path pairs eventual outcomes
with `min(arrival / obswin, 1)`. This is not elapsed follow-up and can include a
DLT before it is observed. The API therefore requires the caller to choose
`criterion="native_arrival"` (compatibility diagnostic reproducing that weight)
or `criterion="followup"` and supply `assessment_delay`. The latter assesses
at `last_arrival + assessment_delay`, marks only DLTs whose event time has
occurred by then as observed, and uses elapsed follow-up for non-DLT weights.
Fixed accrual is supported as a Python option; it is not mislabeled as the
BayesESS native experiment.

BayesESS `getDiff` is the second derivative of each weighted log-likelihood
contribution at beta zero. It can be positive or negative. The native
information-gap calculation adds the average signed second derivative to
`1 / beta_sd**2`; no sign clipping is performed. The exact expected curvature
of a size-m uniform subset is `m / M` times the complete path curvature. This
Rao-Blackwell calculation removes subset simulation noise while preserving
trial-path variability.

Posterior moments default to a full-real normal-prior posterior. The explicit
`posterior_moments="native_truncated_numerator"` mode retains the dfcrm
convention of using the full-real denominator and moment numerators restricted
to `[-10, 10]`. The result records `starting_dose`, criterion, moment
convention, numerical refinement discrepancy, and evaluations. The discrepancy
is an integration diagnostic, not a rigorous error bound. Unresolved
quadrature raises an error.

## Numerical details and limits

Posterior likelihood integration uses positive Gauss-Hermite quadrature refined
through orders 32, 64, 128, 256, 512, and 1024. Refinement checks posterior
mean, variance, and log normalizer; zero quadrature weights are omitted before
log conversion. Native truncated numerators use deterministic split
Gauss-Legendre integration. At `w == 1`, the no-DLT curvature reuses the
complete-CRM cancellation-safe kernel. Sign crossings are detected by direct
sign comparisons, avoiding overflow from multiplying extreme precision gaps.

The simulation preflights dose, patient, replication, tape-cell, and
conservative quadrature-work caps before random generation. Inputs are explicit
tapes or a random generator/seed. A complete replay provides outcome and
event-delay uniforms and, for Poisson accrual, arrival uniforms; fixed accrual
has deterministic spacing. Output arrays are immutable. The implementation
supports at most 20 doses, 200 patients, 1,000 replications, and 200,000
replication-by-patient cells. The default work budget is 50,000,000 units and
the hard caller-selectable ceiling is 1,000,000,000 units.

## Independent validation

`tools/reference_tite_crm_prior_ess.R` is a portable source-hash-guarded
reference generator. It sources the pinned local upstream files without
copying them, extracts the nested BayesESS `getDiff` expression, and runs
unchanged `onetite` with explicit bound random tapes. Run it from the repository
root as:

```sh
Rscript tools/reference_tite_crm_prior_ess.R . tests/fixtures
```

The generator verifies the checked-in source blobs before execution:

- dfcrm `18891ccb969e3e4f87e4489a04b48df227bb9273`, blob
  `c955ae921232695c6201515bcc1a7720e4c9703f`;
- BayesESS `4bbf4df3789912b967774e8ff5c3a2d6d5646cdd`, blob
  `813a734264059ff7e47eebf51eda02351818a8ce`.

Committed CSV fixtures cover fixed and irregular Poisson arrival paths, each
interim beta and dose, final MTD/beta, source-native and corrected as-of
curvature, the late pending DLT, and a positive weighted non-DLT curvature
case. The source paths produce dose sequence `[1,2,1,1,1,1,1,1]`, final beta
mean `-0.983991693211495`, and MTD 1 in both accrual cases. The BayesESS
arrival-weight curvature is `-9.81135830513328`; the corrected follow-up
curvature is `-7.01424553663204` (fixed) and `-7.08908052429197` (Poisson).
The last DLT is pending at assessment time in both corrected ledgers, with
follow-up weight 0.3 rather than the BayesESS arrival proxy 1.

Independent adaptive-quadrature checks in the root audit script
`research/raw/check_tite_moments.py` covered 12 cases: partial-weight,
all-DLT, and all-non-DLT histories at prior SD `sqrt(1.34)` and 4, under both
moment conventions. Against direct-product likelihood and scalar adaptive
quadrature, maximum absolute discrepancies were `1.3004353e-10` (full) and
`7.4550144e-11` (native truncated); runtime was 0.172 seconds with 121.31 MiB
peak process RSS and no swap. After integration, the portable R generator
regenerated all five fixtures byte-identically in a separate ignored directory
(0.541 seconds, 84.30 MiB peak child RSS, zero swaps). The
reproducible project test command is:

```sh
PYTHONPATH=/private/tmp/iboin-completion-luna/src \
  .venv/bin/python -m pytest tests/test_tite_crm_prior_ess.py -q
```

The focused test file covers complete-CRM reduction, late-DLT as-of handling,
source replay fixtures, Poisson uniform transformation, signed and near-one
curvature, and preflight validation. Root ran the final focused file with all
7 tests passing in 1.36 seconds (1.446 seconds including the driver), 130.48 MiB
peak RSS, and no swap. No full test suite or large simulation was run for this
tranche.
