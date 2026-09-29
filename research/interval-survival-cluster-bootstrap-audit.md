# Interval-PH cluster coefficient bootstrap

## Source contract

The cluster method follows the `ic_sp` / PH branch of pinned `icenReg` 2.0.16,
commit `26fadac37c6b54dd0e29c91c2bf07942ae120356`. The source helper is
`clusterBootstrap.R`, blob
`3b0667806deaab7650d4feb0d69580eaf481bcc8`. `make_sample_inds` splits row
indices by subject ID, samples the same number of subject groups with
replacement, then concatenates every row from each selected group. Unequal
cluster sizes therefore make the replicate row count vary. Repeated group
selections retain repeated observations.

`IR_cluster_vcov` refits the model once per resample and computes the sample
covariance of the returned coefficients (`cov`, denominator B-1). Any fit
error aborts the native procedure; it does not redraw failed resamples. The
native method also accepts parametric interval fits, but this implementation
is deliberately restricted to the existing Python semiparametric PH fit.
There is no Python parametric interval fitter here. The fitted likelihood and
point fit retain the package's independent-row model; only the resampling
covariance reflects cluster sampling.

## Python contract and limits

`bootstrap_interval_survival_cluster_coefficients` requires one nonempty
integer or string ID per observation, with a homogeneous ID type. Its sorted
unique labels define columns in the zero-based `(B, G)` cluster-draw tape.
Each replicate draws G IDs with replacement. Repeated selected clusters are
represented by integer frequency weights on their member rows, which keeps
the likelihood and endpoint support equivalent to concatenating those rows
while limiting the live refit input to the original n rows. `resampled_row_counts`
retains the expanded sample size for each replicate, and
`maximum_possible_resampled_rows` records the preflight upper bound.

NumPy's random stream is not claimed to match R. A tape provides exact resample
replay. Source behavior is raise-on-fit-failure; `on_fit_failure="record"` is
an explicit Python extension that retains a failed row without redrawing.
Coefficient covariance and standard errors use successful fits only and are
undefined with fewer than two successes. This scope does not claim
case-weighted cluster sampling, stratified cluster sampling, or baseline
confidence bands.

The preflight bounds cluster-draw and row-frequency work, the explicit tape,
retained coefficient/status summaries, the largest fit scratch, and the
iteration-weighted work for the original plus every replicate before any RNG
is created or advanced.

## Reproducing the source reference

`tools/reference_interval_survival_core.R` contains the shared, hash-guarded
loader for the unchanged icenReg optimizer helper used by the ordinary and
cluster bootstrap reference scripts. Run
`Rscript tools/reference_interval_survival_cluster_bootstrap.R` from the
repository root after restoring the ignored `research/raw/icenReg` cache. The
cluster reference sources the pinned `clusterBootstrap.R` by its verified Git
blob hash, uses the repository's mixed interval fixture with four rows per
synthetic cluster, and records the source helper's fixed-seed selected-group
tape before fitting each repeated-row sample. It writes input, tape, slope,
covariance, and fit-summary CSVs under `tests/fixtures/`. These fixtures check
source fitting and covariance construction with a replayable tape; they do not
claim R/Python RNG stream parity.
