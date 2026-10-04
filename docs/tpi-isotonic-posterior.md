# TPI isotonic-transformed posterior intervals

This calculation applies the inferential procedure described in the mTPI
paper to `TPIDesign`'s original TPI beta posterior. It is a separate analysis
from TPI's posterior-SD interval-mass decisions and from its one-shot isotonic
MTD point estimate. It does not alter dose conduct or claim native TPI
software parity.

```python
import numpy as np
from mdanderson_stats import TPIDesign, tpi_isotonic_posterior_intervals

design = TPIDesign(
    target=0.30,
    prior=[(1.0, 19.0), (3.0, 17.0), (6.0, 14.0), (2.0, 8.0)],
)
result = tpi_isotonic_posterior_intervals(
    design,
    patients=[8, 6, 0, 3],
    toxicities=[1, 2, 0, 2],
    draws=20_000,
    confidence=0.95,
    rng=np.random.default_rng(141),
    retain_draws=False,
)
print(result.lower, result.median, result.upper)
```

For dose `j`, each draw is sampled independently from
`Beta(a_j + y_j, b_j + n_j - y_j)`, using the prior pair configured for that
dose. Every joint vector is then transformed by increasing weighted isotonic
regression. The returned marginal intervals describe the distribution of
these **isotonic-transformed draws**; they are not a posterior conditioned on
the toxicity probabilities being monotone. Equal isotonic weights are the
default; positive finite dose weights are an explicit alternative.

The full supplied dose grid is included. An untried dose contributes draws
from its configured prior. This is an explicit Python convention because the
source does not state whether interval summaries should include untried doses.
For `TPIDesign(prior=(a,b))`, all doses use the common prior. For a matrix of
per-dose shape pairs, patients and toxicities must be one-dimensional vectors
with exactly one count per configured dose. Dose-specific Beta priors are a
conjugate Python generalization documented in
[`tpi-informative-priors.md`](tpi-informative-priors.md), not a recovered
original-TPI prior specification.

`lower`, `median`, `upper`, and `mean` summarize the transformed draws.
Equal-tailed bounds use NumPy's linear empirical quantile convention. Draw
count, confidence level, weights and RNG are explicit Python choices. The
operation shares the bounded beta-draw/isotonic implementation with
`mtpi_isotonic_posterior_intervals`; the existing MTPI API and random stream
remain unchanged. Matrix/work bounds are checked before the RNG is consumed.
Set `retain_draws=True` only when downstream joint summaries need the
transformed sample matrix.

This is not tuning/calibration for TPI's lower and upper SD multipliers. The
cached original-TPI source says those multipliers require calibration but does
not define a unique scenario set, loss or optimizer. Native archive and
spreadsheet/report workflows also remain separate work.

See the [source and validation audit](../research/tpi-isotonic-posterior-audit.md).
