# Saved WFMM contrasts, random functions and prediction targets

Base checkpoint: 54b1a05ee82c8ed17292b422208ba8baeffcfb7b. WFMM remains partial;
catalog counts remain 103 implemented, 31 partial, 4 pending.

The saved Python workflow now exposes effect/time contrasts, confidence,
pointwise quantiles and effect thresholds. It can report retained fitted random
functions and full original-grid covariance/plug-in correlation, with explicit
bounded requests. Existing and new random-level prediction designs compose with
fixed effects and optional residual noise. Shared design columns mean shared
new random effects. The prediction stream is separate from fitting. Initializer
holds still stop before sampling; successful execution does not certify Monte
Carlo precision or clinical validity.

Independent tests calculate fixed/regional means, sample SDs, quantiles and
effect probabilities directly from retained posterior draws. Existing-level
predictions match retained random draws; repeated new-level rows share summaries;
saved runs replay, and adding prediction leaves fitting draws unchanged.
Invalid dimensions, nonfinite thresholds, labels and controls fail early.
Regression coverage reopens complete historical version-1 studies lacking all
new fields, supplying their original defaults. Random summaries remain opt-in
and covariance remains on the original grid even when curves use time regions.

## Verified advertised scope

The recovered official wfmm_Users_Guide.pdf (PostProcessSpecs and Output sections)
documents L and LT, fixed/random function summaries, variance/correlation output,
quantiles, effect probabilities and simultaneous bands. It documents no native
predicted-curve file interface. The earlier audit correctly identifies prediction
as the extension described by Morris and Carroll, Section 5.1. Therefore a
hypothetical native prediction-file contract is not a completion gate. Original
MAT/binary fitted-output compatibility remains a separate claim from a complete
Python study workflow. No visual GUI or exact native RNG identity is required.

The same guide advertises db1–db10, sym2–sym8, coif1–coif5, periodic/zero/reflection/
interval boundaries, extended_mode 0/1, and 2D/3D transforms (including rectangular
2D settings). Remaining wavelet families, interval/reduced boundary modes and
multidimensional transforms still require implementation and independent/source
validation. Native hyperparameter updates and broader sampler contracts also
remain explicit gaps; no application promotion is justified by this checkpoint.

Native integer-truncated bands/SimBaS remain isolated research evidence, never
the scientific default. PCA transfer provenance and the provisional test-only
Haar sign correction remain preserved in the transform-study checkpoint.
See the adjacent validation manifest for final acceptance evidence. Notices,
packaging/licensing restrictions and PR restrictions are unchanged.

## Acceptance

76,760 full tests passed in 368.02s with 777 production files and 1,602
test/fixture files unchanged. Focused WFMM: 628 passed. Fresh installed wheel:
656 passed including 28 plot/TIFF checks; four examples and dependency checks
passed. Lint, formatting, types, build, 781 package-file equality, 29 retained
notices and original-binary exclusion checks passed. Artifacts predate final
evidence documentation. See [validation](wfmm-reporting-validation-2026-10-12.json).

Five native report-edge cases regenerated identically, including a probability-
vector-dependent quantile and incorrect zero-SD SimBaS. A native conditional U
variance discrepancy remains unresolved and explicitly outside accepted parity;
see [the edge audit](wfmm-report-edges-audit-2026-10-12.md).
