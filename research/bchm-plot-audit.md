# BCHM plotting workflow audit

The cached CRAN BCHM 1.00 source exports exactly three plot methods:
`BCHMplot_cluster`, `BCHMplot_post_value`, and `BCHMplot_post_dist`.
The official cached help slides demonstrate the latter two with color and axis
overrides and call the cluster plot. The existing Python fit already retains
the representative partition, observed rates, and target-specific posterior
draws needed to complete these outputs; no sampler or model changes are needed.

`plot_bchm_cluster` maps observed response rates against one-based subgroup ID
and colors them by the representative partition. `plot_bchm_posterior` plots
the posterior means and optionally observed rates. Its interval endpoints
follow the exact cached CRAN `boa.hpd` rule: with `n` pooled retained draws,
`m=max(1, ceil((1-HPD)*n))`; sort values, form paired endpoints from the first
and last `m` order statistics, and choose the first pair with minimum width.
This interval construction is applied to all retained Python chains pooled
together; native plotting uses the first JAGS chain. The result means and
observed rates use the three-decimal R tie rule used by BCHM results. The
observed rate includes the native `1e-9` pre-rounding offset.

`plot_bchm_density` uses the R `stats::bw.nrd0` bandwidth, including fallbacks
for constant samples, and its default Gaussian support from the sample minimum
minus three bandwidths to the maximum plus three bandwidths. Direct Gaussian
kernel summation replaces R's 512-point FFT interpolation. This keeps the
bandwidth and support but allows small differences from native FFT evaluation.
The Python helpers return Matplotlib axes and import plotting support lazily.
Density kernel work is preflighted and evaluated in bounded chunks.

Source hashes:

- Cached `BCHM_src/R/plot.R`: `83912ca3d49d7c07cdf1063d578d92271afbb2f358154ebd92faad0a5eb6fc69`.
- Cached `BCHM_src/R/utils.R`: `a1ccb9b2f6181564ba7570c758977965c173baf6d43950c09ef14842cc6439a5`.
- Cached official help PDF: `70558b3a1fb83af84cd0314da2da6b775121d0a50642794a1b647b6545eda07e`.

Independent R-generated fixtures check 18 HPD cases, R bandwidths and direct
Gaussian-kernel ordinates for six posterior-sample shapes, including ties,
two draws, zeros, ones, and constant values. FFT ordinates are retained in the
fixture to quantify—but not force away—the expected interpolation difference.
Across 3,072 ordinate checks, maximum absolute FFT-versus-direct difference
was `0.00080059613964` and RMS difference was `0.00014747799121`; direct
Gaussian values agreed with the independent reference to `1e-12` absolute and
relative tolerance.
