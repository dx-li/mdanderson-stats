# Interaction Index pooled-error source audit

Reviewed 2026-10-04. Sources were the cached `research/raw/CIInteractionIndex/
paper-bioc.xml` and `ReadmeCI.pdf`, plus the primary full-text article available
on the public PMC article page. The readme maps native `CI.known.effect` to
Section 2 and `CI.delta` to the fixed-ray calculation in Section 3.

## Recovered Section 2 contract

Lee and Kong (2009), §2, equations (8)–(9), explicitly distinguish uncertainty
in a measured combination effect from uncertainty in the separately fitted
single-agent curves. Equation (8)'s raw-index response contribution and its
log-index equivalent are

```text
Var(tau_hat)_y = [sum_i((1 / beta1_i) * d_i / Dhat_y_i)]²
                 * Var(y) / [y * (1-y)]²
Var(log(tau_hat))_y ≈ Var(tau_hat)_y / tau_hat²
```

For replicated combination responses, the text says estimate `Var(y)` from the
sample variance at that combination dose. When no replicate-based estimate is
available, the text says replace
`Var(logit(y)) ≈ Var(y) / [y(1-y)]²` by the average squared residual from all
involved single-drug median-effect regressions, under constant transformed
error variance for single-agent and combination effects. Equation (9) uses
Student-t degrees of freedom `n - 2k`, with `n = sum(n_i)` for the single-agent
fits. The implementation's concrete pooling convention is
`sum((n_i-2) * residual_variance_i) / sum(n_i-2)`; the pooled residual mean
square is then the variance on the logit-effect scale. The primary prose says
“average squared residuals” but does not spell out a denominator. Residual-df
weighting is therefore the implementation's explicit pooling convention,
consistent with OLS residual mean squares and Eq. (9)'s `n-2k` degrees of
freedom; it is not claimed as recovered native implementation detail.

The implementation adds only this no-replicate observed-combination fallback.
It keeps each coefficient covariance as fitted, converts the pooled
transformed-response variance to the response-gradient contribution directly,
and retains the existing Section 2 degrees-of-freedom calculation. It does
not pool coefficients or reweight their covariance matrices.

## Ray distinction

The paper's §3.2, equations (13)–(15), uses a fitted combination curve on the
fixed ray and propagates the covariance for each individual drug regression
and the combination regression. The source does not instruct one to replace
those separate covariances by a single pooled MSE. The new function is
therefore limited to the observed-combination case; fixed-ray inference stays
on `interaction_index_ray` unchanged.

## Provenance and limits

- Cached software readme: `research/raw/CIInteractionIndex/ReadmeCI.pdf`,
  SHA-256 `747dd671f7024f7a47b6f997520d3a53e6e13e753e9cd612355d5ab983bb19bb`.
- Cached article text: `research/raw/CIInteractionIndex/paper-bioc.xml`,
  SHA-256 `9de592a2da6b2d01985870f914412fbcb1549835c0e5dcdd2a60abda39b2b3e6`.
- Primary full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC2796809/>.
- A normalized transcription of the retrieved equation passage, with source
  URL and SHA-256 provenance, is preserved in
  `research/interaction-index-pooled-error-source-excerpt.txt` (SHA-256
  `23934104187fc2b4dd5cf7308b36eae6c81ab9384d70fc25260d7fcca4c3e3c2`).

No raw observation or design information is inferred by the new API. Callers
provide already-fitted independent single-agent curves, the combination dose,
and its supplied mean effect. The pooled fallback assumes homoscedastic normal
errors on the logit-effect scale as in the source. Replicate-level combination
convenience and native file/report behavior are outside this addition.
