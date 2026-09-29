# BCHM plots

The three plot helpers mirror the outputs in the BCHM application. Matplotlib
is optional; install the package's `plot` extra to use them. Each call returns
an axes object and leaves display and file saving to the caller.

```python
import matplotlib.pyplot as plt

from mdanderson_stats import bchm_fit
from mdanderson_stats.bchm_plot import (
    plot_bchm_cluster,
    plot_bchm_density,
    plot_bchm_posterior,
)

# Small illustrative sampling settings for a local plot, not convergence defaults.
fit = bchm_fit(
    [1, 2, 7],
    [10, 12, 15],
    burn_in=50,
    iterations=100,
    warmup=32,
    draws=64,
    chains=2,
    seed=158,
)

plot_bchm_cluster(fit)
plot_bchm_posterior(fit, hpd=0.8, observed_mean=True)
plot_bchm_density(fit, xlim=(0, 1), ylim=None)
plt.close("all")
```

Cluster colors use the representative partition chosen by the clustering
stage. Posterior points display the R-rounded subgroup posterior means, with
the observed rates marked in magenta when requested. HPD interval construction
matches the CRAN helper; the three posterior summaries use the pooled retained
Python chains. Density curves use the R bandwidth and support with direct
Gaussian-kernel evaluation. Each plot accepts an optional Matplotlib axes so it
can be combined with other graphics.
