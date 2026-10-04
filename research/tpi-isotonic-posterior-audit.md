# TPI isotonic-transformed posterior interval audit

## Source scope

The cached file `research/raw/TPI/paper-bioc.xml` is the mTPI article, not the
original TPI article. Its metadata identifies the title “A modified toxicity
probability interval method for dose-finding trials” and DOI
`10.1177/1740774510382799`. In the Discussion passage at source offset 29582,
the authors describe drawing each dose probability independently from its
posterior beta distribution, applying isotonic transformation to each joint
sample, and obtaining posterior intervals numerically from those transformed
samples. The text treats MTD selection/inference as distinct from trial-design
decisions.

This procedure is implemented for `TPIDesign` as an application of that
general inferential proposal to the original TPI posterior. It is not claimed
to be an interval feature specified by the original TPI paper or by native TPI
software. Original TPI source metadata is recorded in `docs/tpi-sources.json`;
the paper's directly retrieved full text was unavailable in the cached source
record, while the author presentation is cached. The TPI guide identifies
posterior isotonic intervals as missing coverage. The primary mTPI evidence is
also cross-referenced in `research/mtpi-isotonic-posterior-audit.md`.

## Posterior and transformation

For dose `j`, the posterior shape parameters come from
`TPIDesign._prior_for_dose(None)` and observed counts:

```text
alpha_j = a_j + y_j
beta_j  = b_j + n_j - y_j
```

For each Monte Carlo replicate, draw independent `Beta(alpha_j,beta_j)` values
across all supplied doses and apply increasing weighted isotonic regression to
that vector. The intervals are empirical marginal summaries of the
transformed-draw distribution. This is not a posterior conditioned on
monotonicity. Common TPI priors retain their posterior; dose-specific prior
pairs use the existing explicit conjugate Python extension. If `n_j=0`, the
configured prior contributes that dose's draw. Including untried doses is an
explicit policy because the source does not resolve that choice.

## Explicit Python choices and bounds

The source does not prescribe weights, simulation count, confidence level or
empirical quantile rule. The implementation uses equal positive weights by
default, an explicit NumPy `Generator`, a caller-specified draw count and
confidence, and NumPy's linear empirical quantiles. It shares the bounded draw
and isotonic kernel with the existing mTPI calculation; the mTPI function's
prior model, default RNG path, and output remain unchanged. Dose vectors and
weights are preflighted before numerical allocation. The existing two-million
matrix-cell and fifty-million work limits are retained; invalid work requests
must leave the caller's random-generator state unchanged.

This feature does not define TPI tuning: although the cached original-TPI
source says its SD multipliers require calibration, no unique scenario design,
objective function, constraints, or optimizer is specified. No native
spreadsheet, archive, calibration, or RNG parity is claimed.

## Validation contract

1. A fixed-seed TPI case with three distinct Beta prior pairs (including one
   untried dose) is replayed by direct beta draws and isotonic projection, then
   compared for transformed samples and all summaries.
2. A common Beta(.005,.005) TPI case is compared with the same shared draw path
   through `MTPIDesign`, ensuring no change to the existing mTPI behavior.
3. A dose/prior shape mismatch and an insufficient work cap are rejected
   before consuming the supplied RNG.
4. Existing mTPI isotonic posterior regression tests continue to cover one-dose
   beta quantiles, monotone transformed draws, the two-uniform projection
   identity, and unmodified RNG behavior.
