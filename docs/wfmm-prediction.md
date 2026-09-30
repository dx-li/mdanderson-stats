# WFMM posterior prediction

WFMM fit draws can produce posterior predictions for explicitly supplied future
fixed and random-effect designs. The prediction function returns draws in the
transformed coefficient domain; pass them to `wfmm_summarize` to reconstruct
curves and calculate pointwise or simultaneous summaries.

```python
import numpy as np
from mdanderson_stats import (
    WFMMPrior,
    fit_wfmm_coefficients,
    wfmm_basis,
    wfmm_predict_coefficients,
    wfmm_summarize,
    wfmm_transform,
)

basis = wfmm_basis(8, transform="identity")
curves = np.arange(32.0).reshape(4, 8) / 10
coefficients = wfmm_transform(curves, basis).coefficients
x = np.ones((4, 1))
z = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
fit = fit_wfmm_coefficients(
    coefficients,
    x,
    z,
    prior=WFMMPrior(0.5, 1.0),
    random_variance=0.5,
    residual_variance=0.25,
    estimate_variances=False,
    sample_random_effects=True,
    draws=32,
    warmup=16,
    chains=2,
    rng=np.random.default_rng(41),
)
future = wfmm_predict_coefficients(
    fit,
    np.ones((2, 1)),
    existing_random_design=[[1, 0], [0, 1]],
)
summary = wfmm_summarize(future, basis)
print(summary.mean.shape)  # (2 future rows, 8 time points)
```

`fixed_design` has shape `(new_rows, fixed_effects)`. For population fixed-effect
means, omit both random designs: no random effects are added even if the fit
contains retained draws for its observed levels. Existing-level prediction uses
`existing_random_design` with exactly the fitted random-level columns and
requires `sample_random_effects=True` in the fit. Its values weight the fitted
conditional random-effect draws.

New levels are separate columns in `new_random_design`; rows that load the same
column share that new random effect. `new_random_strata` gives one fitted
random-variance group index per column. New random effects are drawn
independently across levels conditional on each posterior variance draw. A
Generator is required for this target. Existing and new random designs can be
provided together, so a future set may combine known and new levels.

By default predictions are latent means. Set `include_residual=True` and pass
one `residual_strata` index per new row to add independent row-specific
residual draws. A Generator is required for residual draws as well. The fit's
coefficient-specific residual variances are used; omitting residuals does not
include observation noise. Do not pass residual strata when residual draws are
not requested.

The result is a read-only array with shape `(chain, draw, new_row, coefficient)`.
`wfmm_summarize(result, basis)` applies the basis inverse and returns curve
means, uncertainty summaries and intervals. This prediction interface follows
the implemented independent-level coefficient model. It is a Python extension;
native prediction output formats and file workflow are not claimed. Output,
working-cell and multiplication work limits are checked before prediction
allocations and random draws.

For two future curves sharing a new random-effect level, with independent
observation noise, reuse the fitted variance draws and provide component maps:

```python
replicates = wfmm_predict_coefficients(
    fit,
    np.ones((2, 1)),
    new_random_design=[[1], [1]],  # both rows share one new level
    new_random_strata=[0],
    include_residual=True,
    residual_strata=[0, 0],
    rng=np.random.default_rng(42),
)
replicate_summary = wfmm_summarize(replicates, basis)
```
