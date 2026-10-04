# Median-effect and interaction-index plots

The optional plotting helpers display existing calculations; they do not fit
models or recompute interaction indices. Install the plotting extra with
`pip install 'mdanderson-stats[plot]'` if Matplotlib is not already available.

```python
import numpy as np
from mdanderson_stats import fit_median_effect
from mdanderson_stats.interaction_index_plot import plot_median_effect

dose = np.array([1, 2, 4, 8, 16], dtype=float)
effect = np.array([0.12, 0.22, 0.39, 0.61, 0.78])
fit = fit_median_effect(dose, effect)
ax = plot_median_effect(fit, dose, effect, label="Drug A")
ax.legend()
```

`plot_median_effect` displays the supplied observations at `log(dose)` and
`logit(effect)` and overlays the fitted straight line. The fit is passed
explicitly so plotting never silently refits or changes the analysis.

For an interaction-index result, `plot_interaction_index(effect, result)` draws
the pointwise interval as a sorted line and shaded region. Use
`kind="points"` to retain the original order and display individual error
bars, or `log_index=True` when exponentiating the index could overflow or
underflow. The horizontal additivity reference is 1 on the raw scale and 0 on
the log scale. Fixed-ray intervals are pointwise delta-method intervals, not
simultaneous bands.

```python
from mdanderson_stats import MedianEffectFit, interaction_index_ray
from mdanderson_stats.interaction_index_plot import plot_interaction_index

curves = [
    MedianEffectFit(0.0, -1.0, np.eye(2) * 0.001, 12, 0.01),
    MedianEffectFit(0.4, -0.9, np.eye(2) * 0.001, 12, 0.01),
]
combination = MedianEffectFit(0.2, -1.0, np.eye(2) * 0.001, 12, 0.01)
effects = np.array([0.2, 0.4, 0.6, 0.8])
ray = interaction_index_ray(curves, combination, [1, 2], effects)
ax = plot_interaction_index(effects, ray, label="1:2 composition")
ax.legend()
```

Inputs are limited to 10,000 plotted points and must be finite; median-effect
observations require positive doses and effects strictly between zero and one.
Line plots require distinct effect coordinates. Pass an existing Matplotlib
`Axes` with `ax=` to overlay multiple fits or results. Matplotlib imports are
lazy, so numerical users do not need the plotting extra merely to import the
package.
