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

## Explicit extended boundary transforms

Use `transform="extended_wavelet"`, `filter_length=2..20` (even), and
`boundary="periodic"`, `"zero"`, or `"reflection"` in `WFMMStudy` to select the
native extended-mode one-dimensional db1–db10 convention. Odd lengths are
supported. Reflection repeats endpoint samples; periodic wraps indices; zero
pads outside the signal. Each level retains `floor((n+L-1)/2)` approximation and
detail values. Reconstruction overlap-adds and crops to the original lengths.
`wfmm_extended_wavelet_basis` exposes the same matrix pair independently.

This dense helper supports up to 512 time points with bounded matrix storage and
work. The legacy `transform="wavelet"` periodic-decimation convention remains
unchanged. Redundant coefficients use an explicit independent-coefficient
working model. Source comparisons establish numerical convention agreement,
not bitwise filter or RNG identity; details and tolerances are in
[the boundary checkpoint](../research/wfmm-boundary-checkpoint-2026-10-12.md).

## Contrasts, fitted random functions and future curves

Saved studies accept `effect_contrast` (output effects by fitted fixed effects)
and `time_contrast` (original time points by output regions), using
`L @ curve @ LT`. Fixed summaries apply both. Random-function and prediction
summaries apply the time contrast; prediction rows already specify their fixed
linear combination through `prediction_design`. CSV uses `region_index` when a
time contrast is present. Reports retain the contrast matrices.

`confidence`, increasing `quantiles` in (0,1), and nonnegative `effect_sizes`
control curve reports. Quantiles also control covariance-function summaries.
`include_covariance=True` requests full covariance and plug-in correlation
matrices on the **original time grid**, subject to existing memory/work bounds.
`summarize_random_effects=True` reports retained fitted random functions for up
to 128 fitted levels. Fixed summaries support up to 128 output contrasts, and
prediction summaries up to 128 rows. Studies fail early on invalid designs or
stratum labels. Older complete version-1 studies reopen with historical defaults
for omitted additive reporting controls.

`prediction_existing_random_design` has the same random-level columns as the
fitted design. `prediction_new_random_design` has one column per new random
level; rows sharing a column share that random effect. Its
`prediction_new_random_strata` maps columns to fitted random-variance groups.
Either design can be supplied, or both. With neither, predictions are population
fixed-effect curves. `prediction_residual=True` adds independent row-specific
residual noise using `prediction_strata`. Prediction uses its separate saved RNG
stream. See [the example](../examples/wfmm_reporting.py).

The manual documents fitted functions and covariance output, not a native
predicted-curve file interface. These explicit prediction targets follow the
published model. Python reports keep real-valued standardized deviations,
empirical-tail SimBaS and linear quantiles; they do not copy the native
[integer-truncation defect](../research/wfmm-postprocess-audit-2026-10-12.md).

## Named wavelet families

`wfmm_named_wavelet_basis(..., wavelet="sym4")` supports `db1`–`db10`,
`sym2`–`sym8` and `coif1`–`coif5` with the same extended boundary conventions.
The manual aliases `haar`, `d4`–`d20`, `s4`–`s16` and `c6`–`c30` are accepted.
For saved studies, select `transform="extended_wavelet"` and
`wavelet_name="sym4"` (or another supported name), leaving `filter_length` at its
study default. Conflicting named/length controls are rejected.

The Symlet/Coiflet tap table contains numerical impulse responses, not original
executable code. Native filter precision is preserved: coif5 reconstruction error
reaches approximately `7e-9` in the two-level source comparisons and `2.1e-8` in
short-signal/deep-level mathematical checks. Tests bound these cases at `5e-8`;
machine-precision orthogonality is not claimed. db transforms keep their existing
filter construction. See [the family checkpoint](../research/wfmm-family-checkpoint-2026-10-12.md).

## Images and volumes

`wfmm_spatial_wavelet_basis(shape, layout="pyramid")` supplies extended 2D/3D
separable wavelets. `layout="rectangular"` applies independent complete 1D
transforms along each axis of a 2D image. Pass `levels`, `wavelets` and
`boundaries` as scalar settings or per-axis tuples. Pyramid transforms require
common settings. Arrays use column stacking (`ravel(order="F")`); the first
spatial index varies fastest. Each contiguous subband has its own shrinkage
partition. Dense matrices are bounded to 512 input samples and two million
estimated live cells; every decomposed axis must have at least two samples.

Saved studies use `transform="spatial_wavelet"`, `spatial_shape=(rows, columns)`
(or three dimensions), and `spatial_layout`. Common controls are `levels`,
`wavelet_name` (default Haar), and `boundary`; optional `axis_levels`,
`axis_wavelets`, `axis_boundaries` replace their scalar defaults. Conflicting
scalar/axis settings are rejected. Prediction, covariance and reporting use the
original flattened grid; reshape with `order="F"`. See
[the spatial example](../examples/wfmm_spatial.py). Pyramid studies support highpass/lowpass by whole spatial level, including all
orientations. Rectangular pass flags remain unsupported; compression is available.

Native analysis-only references establish coefficient ordering for tested
extended transforms. Native initialization/report failures remain documented;
native inverse-file interoperability is not established. Synthesis is an
independent composition of validated 1D inverses with reconstruction checks.
Redundant coefficients define a working model, not proof of independent errors.
Existing initializer holds remain enforced, including for degenerate boundary
coefficients. Clinical validity and exact native RNG identity are not claimed.

## Retained draws and thinning

`WFMMStudy(..., draws=1000, warmup=500, thin=5)` performs 5,500 transitions per
chain and retains 1,000 draws, every fifth transition after warmup. The sampler
API has the same convention. `thin=1` retains historical behavior; older saved
studies default to one. Thinning is bounded to 1–100 and all transitions count
toward the work budget. It controls storage and does not establish better Monte
Carlo efficiency. Conditional random-effect draws are still generated on skipped
post-warmup transitions, preserving the unthinned RNG stream and exact subsequence
replay. Reports distinguish retained draws, thinning, and completed iterations;
held studies report zero retained draws and no completed iteration count.

Pyramid spatial `highpass=h>0` discards the approximation and h coarsest detail
levels; `lowpass=l` discards l finest detail levels. Every orientation receives
the same level decision. These recovered source conventions reverse the guide's
short descriptions. Empty selections are rejected. Filtering precedes energy
compression, and reporting restores omitted coefficients as zero. Native
rectangular pass attempts failed in the audited settings, so those flags remain
explicitly unsupported. Native fitting/file compatibility remains a separate
claim from validated analysis selection.

## Explicit optional shrinkage hyperpriors

`WFMMShrinkageHyperprior(a, b, shape, scale, update_mask=True)` enables shared
partition-level inclusion probabilities π and slab variances τ. Parameters
broadcast to fixed effects × sorted retained partitions; a vector addresses
partitions. The independent priors are Beta(a,b) and inverse-gamma(shape,scale),
with density proportional to τ^(-shape-1) exp(-scale/τ). This is an explicit
community model, not a claim of reproducing native hyperprior conversion.

`fit_wfmm_coefficients(..., shrinkage_hyperprior=...)` requires initial π and τ
constant within each partition. Each complete coefficient sweep is followed by
Beta(a+included,b+excluded) and inverse-gamma(shape+included/2,
scale+sum(included coefficients squared)/2) updates. Included exact zeros count
as included. False mask entries retain supplied initial values and consume no
hyperparameter draws. Disabled updates preserve the existing random stream.
Draws outside representable open support raise rather than being clipped.

Saved studies accept the same optional object. Active parameters start at the
Beta prior mean and inverse-gamma prior mode; empirical-Bayes initialization
and its validity holds still run. Masked parameters retain empirical-Bayes
values, which must satisfy the sampler's within-partition constancy requirement.
Reports include π/τ means, requested quantiles, R-hat, MCSE, labels and masks.
Older saved studies default to no updates. Fits retain read-only
`inclusion_probabilities` and `slab_variances` arrays shaped chains × retained
draws × fixed effects × partitions. Thinning preserves exact subsequences,
including hyperparameter draws.

`wfmm_shrinkage_conditional` exposes the independently testable conditional
parameters. `wfmm_mode_shrinkage_hyperprior` optionally calibrates explicit modes
and finite variances using bounded root solving. Its inverse-gamma shapes exceed
two. This mode-centering convention is independent of the diagnosed native
conversion: see `research/wfmm-hyperprior-audit-2026-10-12.md`. Proper explicit
priors with smaller shapes remain allowed but may have infinite moments.
Neither these checks nor posterior diagnostics establish clinical validity.
