# CondiS survival-curve comparison

The CondiS vignette compares a Kaplan–Meier curve using the original event and
censoring indicators with a second curve that treats every CondiS-imputed time
as an event. This package exposes the same comparison as reusable data and an
optional plot. The imputed curve is descriptive; treating imputed observations
as events does not make its usual censoring-based uncertainty valid.

Supply risk-table times explicitly. Counts use the pre-event convention: a
subject is at risk at time `t` when their observed or imputed time is at least
`t`. Censor marks are placed on the original curve after applying its
right-continuous survival step. No confidence intervals, default risk-table
grid, or native Shiny output are inferred.

```python
from mdanderson_stats import condis_impute, condis_survival_comparison

fit = condis_impute([1, 2, 4, 6], [1, 0, 1, 0])
comparison = condis_survival_comparison(fit, risk_times=[0, 2, 4, 6])
print(comparison.observed_at_risk)
print(comparison.imputed_at_risk)
```

When the optional `plot` extra is installed, `plot_condis_survival_comparison`
draws the two curves, censor marks, and the same explicit risk table. Matplotlib
is imported only when that plotting function is called.

```python
from mdanderson_stats import plot_condis_survival_comparison

curve_axes, risk_axis = plot_condis_survival_comparison(comparison)
curve_axes.figure.savefig("condis-survival-comparison.png", dpi=120)
```
