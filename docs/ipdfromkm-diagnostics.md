# IPDfromKM reconstruction diagnostics

Use `ipd_reconstruction_diagnostics` to reproduce the precision and
Kolmogorov–Smirnov fields in the pinned IPDfromKM report without changing the
existing `reconstruct_ipd` result:

```python
from mdanderson_stats import ipd_reconstruction_diagnostics, reconstruct_ipd

observed = [1.0, 0.9, 0.8, 0.65, 0.5, 0.4, 0.2]
fit = reconstruct_ipd([0, 1, 2, 3, 4, 5, 6], observed, patients=100)
diagnostics = ipd_reconstruction_diagnostics(observed, fit.survival)
print(diagnostics.rmse, diagnostics.ks_statistic, diagnostics.ks_pvalue)
```

Pass corresponding observed and fitted survival probabilities in the same
coordinate order. Observed values are kept unrounded; fitted values are rounded
to three decimals, then fitted-minus-observed differences are rounded to three
decimals. Rows with a NaN in either member are removed as a pair, matching the
source report's row-wise `na.omit`; the result records retained input indices
and the number omitted. Infinities, finite values outside `[0, 1]`, mismatched
vector lengths, and fewer than two retained pairs raise an error. A call
accepts up to 100,000 input points.

The source RMSE divides squared rounded differences by `n - 1`, while mean
absolute error divides by `n`. Its field labeled “max absolute error” is
actually the signed `max(difference)`. That value is returned as
`legacy_signed_max_error`; `max_absolute_error` supplies the literal absolute
maximum as well.

The two-sided KS statistic and nominal p-value compare rounded fitted values
against the original observed values, as the R report does. Under R 4.4.1's
default, the p-value uses an exact tied-label permutation calculation when
`n_x * n_y < 10000`, and the asymptotic two-sided limiting KS distribution
otherwise. The result records which branch was used. These are paired curve
coordinates and the fitted values come from the same reconstruction, so the KS
p-value is a source-compatible report field, not a valid independent-sample
inferential p-value. Further source and validation details are in the
[diagnostics audit](../research/ipdfromkm-diagnostics-audit.md).
