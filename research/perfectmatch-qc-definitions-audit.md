# PerfectMatch `.pdn` QC statistic definitions

This note records what can be recovered from the cached PerfectMatch 2.3
manual and the Zhang, Miles and Aldape paper. It does not infer native file
formats or fill in undocumented aggregation rules.

## Manual output definitions

PerfectMatch Manual §4, cached PDF page 8, shows a per-array `.pdn` output row
with `err_T`, `corr`, `P_size`, `crossPM`, and `avg_Affynity` (PDF text lines
173–188). The accompanying descriptions say:

- `err_T` is goodness of fit between the model and observed data for a probeset.
  The manual gives no residual formula, scale, or aggregation rule.
- `cross_PM` is an estimated ratio of nonspecific binding signal to total probe
  signal. It does not identify the numerator and denominator precisely, say
  whether totals use observed or fitted signals, or define aggregation over the
  probeset's probes.
- `avg_affynity` is the average gene-specific binding affinity of probes in a
  probeset. The manual does not define the averaging operation or specify
  whether it uses all probes or the subset retained for model fitting.
- `P_size` is the number of probes used in model fitting. The existing
  `PDNNExpression.probes_used` result already reports this count per probeset.

## Paper equations and their limits

The paper's Methods Eq. (1), p. 3, defines the specific-binding contribution to
each probe's fitted signal as `N_j / (1 + exp(E_ij))`. This supplies a
per-probe affinity-like factor, but neither Eq. (1) nor later equations define
the `.pdn` `avg_Affynity` aggregation or the probes included in that average.

Eq. (4), p. 4, defines one mean squared error of natural-log intensities over
all probes on an array. It is
not a per-probeset statistic and does not establish the formula, scaling, or
outlier handling for the manual's `err_T`. Eq. (5), p. 4, gives the
per-probeset expression estimator and identifies probe exclusions for that
estimator, but it does not define the `crossPM` summary or connect its exclusions
to `avg_Affynity`.

Thus the paper provides useful per-probe model quantities, while the three
named `.pdn` summaries still lack enough detail for source-faithful calculation.
Implementing plausible residual or signal ratios would add Python conventions,
not recover the native definitions. The manual also says the example
`PDNN.log` `ScalingFactor` and `Absent genes` values are incorrect for that
release (Manual §4, PDF text lines 190–195); they are not treated as usable
definitions here.
