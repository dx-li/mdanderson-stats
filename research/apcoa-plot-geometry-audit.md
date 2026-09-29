# aPCoA plot geometry source audit

## Recovered contracts

The cached aPCoA 1.3 `R/aPCoA.R` source calls `dataEllipse(..., levels=0.95,
plot.points=FALSE)` separately for each group and panel. It also calculates
95-percent ellipse extents for its panel limits using
`sqrt(2*qf(.95, 2, n_group-1))`. The package imports `car` but does not pin its
version. The exact plotting helper was recovered from contemporaneous
`car` 3.0-12, commit
[`b6f80103057cfc3dff208cc2a68525c72d61a0ca`](https://github.com/cran/car/commit/b6f80103057cfc3dff208cc2a68525c72d61a0ca): unit weights in `cov.wt`, ordinary sample covariance, 51 segments/52 closed
vertices, and a pivoted Cholesky factor with the pivot columns restored. The
Python helper reproduces this ellipse geometry with a scaled 2-by-2 pivoted
factorization. Rank-one covariance is returned as a collapsed line. Rank-zero
covariance raises; it is not jittered. This identifies the contemporaneous
source behavior, not an unpinned dependency guarantee for the app.

For centers, `R/aPCoA.R` passes square group submatrices directly to
`cluster::pam(..., 1)` without `diss=TRUE`. Consequently, PAM considers matrix
rows as observations with Euclidean distances between row profiles. The
original panel uses `D[group,group]`; the adjusted panel uses `E[group,group]`.
The native dependency recovered for this check is `cluster` 2.1.6. Its k=1
BUILD step computes gains from `s - distance`, with `s=1.1*max(distance)+1`,
and updates on `<=`, so the last exact candidate tie wins. The implementation
follows this bounded k=1 path rather than general PAM. Reconstructing original
within-group distances from a centered Gram matrix and using scaled matrices
can change choices under floating-point near-ties; no exact tie-parity claim is
made for such roundoff cases.

The plot overlay options are off by default to preserve existing Python plot
behavior. The geometry helper returns immutable ellipse vertex arrays and
sample indices for medoid anchors. Original-distance reconstruction clips only
small negative squared values within a scale-aware roundoff bound and reports
the clipped-cell count; materially negative values raise. Medoid work is
preflighted at twice the sum of group-size cubes (50 million bound), and a
within-group profile is limited to 4,000,000 cells. Pairwise row-profile
distances are computed with bounded two-dimensional scratch, not a cubic
broadcast tensor.

## Numerical references

`tools/reference_apcoa_overlays.R` reconstructs the aPCoA matrix/projection
equations in base R, then uses `car` 3.0-12 ellipse geometry, `cov.wt`, `qf`,
and `cluster::pam`. It exports
four group/panel summaries, all 208 ellipse vertices, sample coordinates and
ten constant/path medoid tie cases. The focused Python regression compares
these coordinates, anchors and polygons, including rank-one and rank-zero
covariance behavior, tight translated clusters, tiny Gram scales, and extreme
coordinate scaling. Eight focused aPCoA core/overlay tests pass; maximum
absolute coordinate and ellipse-vertex differences are `6e-15` and `9.1e-15`,
respectively; all medoid indices match
and no squared-distance cells required clipping. The Agg overlay preview
rendered. The final focused test run took 1.85 seconds, with peak RSS
133,136,384 bytes and zero swaps. Targeted Ruff, formatting and mypy checks
pass. No broad suite or dependency installation was run.
