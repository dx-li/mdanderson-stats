# Randomized discrete and rounded CDF diagnostic audit

Johnson (2004), *A Bayesian χ² Test for Goodness-of-Fit*, develops the
posterior CDF diagnostic in equations (2)–(3). The cached primary-text extract
`research/raw/BCSTTE/johnson.txt:191-235` assigns each observation to the
equal-probability bin containing its posterior CDF value and uses the Pearson
statistic with `K−1` reference degrees of freedom. Its discrete-data extension
at `:281-310` describes random allocation when an observation's CDF mass
crosses bin boundaries. It also provides a separate fixed-bin definition in
equations (5)–(6), where the expected count uses posterior-draw probabilities
summed over each observation and each bin. Both references retain the paper's
asymptotic regularity conditions.

The cached BCSTTE user guide extract `research/raw/BCSTTE/guide.txt:54-78`
advertises `--discrete` for integer-rounded survival times and interprets an
integer `t` as the interval `(t−1/2, t+1/2)`. The cache does not establish that
the native executable uses a particular RNG stream or how it seeds the
randomization.

## Python contract

`bayesian_chi_square_discrete_cdf` accepts same-shaped matrices of posterior
left and right CDF bounds, with posterior draws by row and observed values by
column. It requires finite numeric bounds satisfying `0 <= left < right <= 1`.
For a discrete atom these are `F(y−|theta)` and `F(y|theta)`; for a rounded
measurement they are the CDF at the interval endpoints under the model used to
fit the posterior. Every row must be the same joint posterior parameter draw
for all observations.

For each posterior row and observation, the function samples an independent
uniform location in the supplied mass interval, then delegates bin counting,
Pearson statistics and asymptotic summaries to the existing continuous-CDF
routine. Equal-probability bins use its convention `(a[k−1], a[k]]`; a CDF
value exactly on an internal edge is assigned to the bin ending at that edge.
The lower endpoint is open for the randomized discrete transform; an inward
floating-point adjustment prevents interpolation from rounding back onto the
left endpoint. When the bounds are adjacent floating-point values, the upper
bound is used because no representable interior point exists; the existing
upper-inclusive bin rule determines its assignment. A positive interval whose
bounds collapse to the same floating value is rejected rather than treated as
an observed event.

This API consumes posterior bounds only. It neither fits discrete/rounded
likelihoods nor converts censored observations into CDF intervals. A valid
caller must fit using the observed atom probability or the rounded-interval
probability represented by the supplied CDF bounds. The explicit uniform
randomization and seed/Generator API are community Python conventions; native
randomization parity, report formatting and a native rounded-data fitter are
not claimed. Input storage is bounded to one million draw-observation cells,
and output count storage to two million draw-bin cells, both checked before
array materialization or RNG consumption.

The executable geometric-model example in `docs/bayesian-chi-square.md`
provides a small correctly conditioned posterior and computes atom bounds from
the geometric CDF. This is a runnable demonstration, not a native fixture.
