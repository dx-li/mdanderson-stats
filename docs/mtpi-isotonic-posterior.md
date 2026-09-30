# mTPI isotonic posterior intervals

The mTPI paper distinguishes dose selection from posterior inference. It
suggests drawing each dose toxicity probability independently from its beta
posterior, applying an isotonic transformation to every vector of draws, then
forming numerical posterior intervals from the transformed samples. This
module provides that inferential calculation; it does not change the existing
MTD-selection rule.

```python
import numpy as np
from mdanderson_stats import MTPIDesign, mtpi_isotonic_posterior_intervals

result = mtpi_isotonic_posterior_intervals(
    MTPIDesign(target=0.30, lower=0.25, upper=0.35),
    patients=[6, 6, 0, 3],
    toxicities=[0, 2, 0, 2],
    draws=20_000,
    confidence=0.95,
    rng=np.random.default_rng(141),
    retain_draws=True,
)
print(result.lower, result.median, result.upper)
```

At dose `j`, the untransformed posterior is
`Beta(prior_alpha + y[j], prior_beta + n[j] - y[j])`, using the independent
common beta prior supplied to `MTPIDesign` (uniform by default).
The full supplied dose grid is transformed; an untried dose has `n=0` and
therefore contributes a draw from that prior. For each draw,
increasing weighted isotonic regression pools any order violations. Equal
weights are the default; a positive finite `weights` vector is an explicit
alternative policy. The paper does not state whether the interval operation
should include untried doses; this API explicitly transforms the complete
supplied grid, with untried doses contributing their prior draws.

The returned bounds are marginal equal-tailed intervals, not simultaneous
coverage bands across the dose grid. They are calculated with
`numpy.quantile(method="linear")`; `median` and `mean` summarize the transformed
draws. These posterior means are not the same as isotonic regression applied
once to the vector of beta posterior means. The source does not choose the
isotonic weights, simulation count, or quantile convention, so these are
explicit Python choices. They do not claim native spreadsheet interval parity.

`retain_draws=True` returns the transformed sample matrix for joint summaries;
otherwise it is discarded after the marginal summaries are computed. The
implementation caps the dose-by-draw matrix at two million cells and bounds
draw, isotonic, and empirical-quantile work before consuming the supplied
generator. Monte Carlo intervals can vary with the seed and draw count; increase
draws and assess simulation precision for consequential use.

See the [focused source and validation audit](../research/mtpi-isotonic-posterior-audit.md).
