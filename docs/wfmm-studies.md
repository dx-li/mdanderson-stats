# WFMM saved studies and PCA transforms

`WFMMStudy`, `read_wfmm_study` and `run_wfmm_study` provide a versioned Python
workflow: save/reopen observed curves and controls, transform and select
coefficients, initialize, sample, reconstruct curves and variance functions,
predict population curves, and export JSON and CSV summaries. This is a Python
JSON contract, not native MAT/binary-file compatibility. See the executable
[synthetic example](../examples/wfmm_study.py).

```python
import numpy as np
from mdanderson_stats import WFMMStudy, read_wfmm_study, run_wfmm_study

y = np.array(
    [
        [1, 4, 2, 3],
        [2, 2, 3, 1],
        [4, 3, 1, 5],
        [5, 6, 4, 2],
        [6, 3, 7, 4],
        [3, 1, 2, 6],
        [2, 5, 5, 3],
        [4, 2, 6, 2],
    ],
    dtype=float,
)
study = WFMMStudy(
    y,
    np.ones((8, 1)),
    transform="PCw",
    alpha_pc=0.9,
    draws=32,
    warmup=16,
    seed=212,
    prediction_design=[[1.0]],
)
saved_input = study.to_json()
result = run_wfmm_study(read_wfmm_study(saved_input))
saved_report = result.report_json()
if result.fit is not None:
    curve_csv = result.curve_csv()
else:
    print(result.initialization.holds)
```

The short example verifies composition, not posterior precision. Stable
initialization retains its convergence/proposal holds: a held study returns its
initialization and diagnostic report without running MCMC. It does not silently
switch to the native profile, increase iteration budgets, or override holds.
Moment, optimizer and empirical-Bayes iteration limits are saved controls.

Studies support identity, periodic wavelets, explicit square custom inverse
pairs, PC, PCw and wPC; design matrices and random/residual group labels are
saved with them. The wavelet default is explicit one-level Haar, differing from
the native guide's db4 default. PCA and custom transforms retain the existing
coefficient-independence model convention; nonorthogonal or redundant coordinates
do not represent independent observations in the original data space.

The seed initializes NumPy PCG64 through two SeedSequence children, one for
fitting and one for prediction. Requesting predictions therefore does not alter
posterior fitting draws. Reopening preserves the controls and reproduces runs
under the same numerical implementation and dependencies; no cross-version or
native RNG identity is promised. Reports include the input JSON SHA-256, NumPy
version, seed, retained indices, energy fractions, acceptance/proposal counts,
coefficient and variance R-hat/MCSE, curve summaries and variance functions.
Nonfinite diagnostics are JSON null with an explicit explanation. Classical
split R-hat and batch-mean MCSE remain estimates; reports explicitly state that
precision is not certified and clinical validity is not established.

Saved prediction controls describe population fixed-effect curves with optional
new residual noise and explicit residual strata. The returned `fit` also supports
the full existing [random-level prediction API](wfmm-prediction.md); random-effect
draws are retained for studies with a random design. JSON reports omit the full
posterior draw arrays, which remain available on the result. Curve CSV indices
are zero-based and its simultaneous bounds use the existing posterior-summary
definition, not separate pointwise credible limits.

## PCA and hybrid bases

`wfmm_pca_basis(curves, kind="PC", alpha=.99)` returns a `WFMMPCTransform` with
its calibrated `basis`. Use that basis with `wfmm_transform`, `wfmm_inverse`,
`wfmm_summarize`, covariance summaries, selection and compression. A basis now
distinguishes `time_count` from `coefficient_count`.

- PC uses uncentered right singular vectors. Reconstruction projects onto the
  retained subspace; it does not impute discarded directions.
- wPC first applies the supplied periodic wavelet basis, then computes PCs in
  coefficient space. Its calibrated analysis/synthesis matrices operate directly
  on curves in the original time coordinates.
- PCw concatenates PC scores with wavelet scores of the residual after PC
  projection. Its synthesis adds the two reconstructions. It has more coefficient
  coordinates than time points and is a redundant representation. Independent
  coefficient priors/variances are a declared working model, not an exact
  independent-coordinate change of variables for the observed data likelihood.

Hybrids require an explicit `wavelet_basis`. No input centering is performed.
The fit rescales the entire matrix internally for numerical stability and records
that scale and the relative singular values. At alpha below one, retention uses
the native strict cumulative-energy crossing rule. At alpha one, Python retains
the numerical row space; it does not copy the original zero-column failure.
Zero-energy matrices and thresholds splitting numerically tied singular values
are rejected. Signs are pinned by positive largest-magnitude loadings, so native
scores can differ in sign while describing the same scientific subspace.

Manual positive integer `partition_sizes` must sum to the retained PC count.
Alternatively, `npartitions` splits evenly with the remainder in the last group.
Automatic grouping follows recovered half-unit bins of log-base **singular
values**, despite the guide's eigenvalue wording, merging singletons forward
and the final singleton backward. The default base is ten. Sparse grouping
avoids native endpoint/allocation failures; exact floating-point boundary
parity is not asserted. PCw appends wavelet partitions after its PC partitions.

## Pass filtering followed by compression

`wfmm_pass_filter(coefficients, basis, highpass=0, lowpass=0)` returns a selection
in original packed order. Original executable behavior reverses the guide's
descriptions: positive highpass h removes the approximation and h coarsest detail
bands; lowpass l removes l finest detail bands. Zero highpass removes nothing.
Invalid or empty selections raise an error. For two levels, highpass=1 retains
only the finest detail, while lowpass=1 retains the approximation and coarse detail.

Apply energy compression to the filtered space by passing the selection's
`retained_indices` as `eligible_indices` to `wfmm_compress_coefficients` together
with the original full coefficients and basis. Returned indices and vote counts
retain original identities. Its energy fractions then use the eligible energy
denominator; they do not include energy removed by pass filtering. At alpha=1,
the bypass retains only the eligible coefficients. Saved studies compose these
operations in that order. Zero restoration describes the selected subspace only.

Native boundary extensions, other wavelet-family parity, multidimensional
transforms, native file/report interoperability, optional native hyperparameter
updates and complete native sampling compatibility remain open. WFMM70 remains
partial. The [transform/workflow audit](../research/wfmm-transform-study-checkpoint-2026-10-12.md)
records independent references and final validation.
