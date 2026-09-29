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
integer or string ID per observation, with a homogeneous ID type. Python-sorted
unique labels define columns in the zero-based `(B, G)` cluster-draw tape. This
is a deterministic Python convention, not a claim of identical factor ordering
to every R locale.
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
blob hash, uses the repository's 96-row mixed interval fixture with unequal
cluster sizes `(18, 22, 26, 30)` and labels `(20, 10, 40, 30)`, and records the
fixed zero-based tape `[[0, 1, 1, 3], [2, 0, 3, 0]]` in Python-sorted label
order `(10, 20, 30, 40)`. A narrow `sample` adapter feeds those choices to the
unchanged `make_sample_inds`; the source helper performs row expansion and
native refits use those expanded rows. The resulting row counts are 84 and
100. The generator writes input, tape, slope, covariance, and fit-summary CSVs
under `tests/fixtures/`. These fixtures check source fitting and covariance
construction with a replayable tape; they do not claim R/Python RNG stream
parity.

## Integrated numerical evidence

The unchanged native resampling helper expands the two fixed unequal-group
tapes to 84 and 100 rows. Its native PH refits and sample covariance agree
with Python's compressed frequency-weight refits: maximum coefficient
difference 2.795e-7 and covariance-entry difference 2.905e-8. The comparison
is retained in the focused expanded-row test. The helper supplies row
selection; the shared native optimizer adapter supplies refits and R `cov`
supplies covariance. This does not execute the full R object/formula wrapper.

The extracted shared loader also regenerates all five original PH cases;
all eight existing PH/ordinary-bootstrap CSVs remain byte-identical. Native
generation took 6.977 seconds with peak child RSS 309.80 MiB and zero swaps,
including the single-process C++ builder. No dependency was installed.

All 31 affected cluster-bootstrap and forest checks pass together in 3.50
seconds at 145.64 MiB peak RSS and zero swaps. The public cluster example
completes all eight refits, retaining expanded counts from 74 through 106.
Both new guide examples and the native coefficient comparison take 0.937
seconds at 121.78 MiB RSS, zero swaps. Numerical threads are capped at one.
