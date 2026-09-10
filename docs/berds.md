# BERDS variable selection

MD Anderson catalog entry 35 implements **backward elimination via repeated data
splitting**, developed by Thall, Russell and Simon. `berds` provides the complete
statistical workflow: repeated elimination/validation splits, regression deletion
paths, trimmed validation-loss curves, truncated threshold selection, and a final
full-data regression. `backward_elimination` also accepts a specified threshold.
An intercept is always fitted, as in the distributed command-line program.

```python
import numpy as np
from mdanderson_stats import berds

rng = np.random.default_rng(63)
x = rng.normal(size=(200, 12))
y = 1 + x[:, :3] @ np.array([1.0, -0.7, 0.3]) + rng.normal(size=200)
selection = berds(y, x, seed=33)
assert selection.model.selected == (0, 1, 2, 5)
predicted = selection.model.intercept + x @ selection.model.coefficients
assert predicted.shape == y.shape

# Preserve actual splits to replay the same analysis independently of the RNG.
replay = berds(y, x, validation_indices=selection.validation_indices)
assert np.array_equal(replay.scores, selection.scores)
```

`response` is a length-n vector; `predictors` is an n-by-p matrix without an
intercept column. Selected predictor indices are **zero-based**. The returned
coefficient vector retains all original columns, with zeros for deleted variables.
The fit includes residual sum of squares, MSE, R-squared, Mallows' Cp and the full
model MSE. Output arrays are read-only. Predictions use the returned intercept and
coefficients; no preprocessing objects are required.

## Selection rules

Defaults follow the source: 20 repetitions, elimination fraction 0.5, trim 0.1,
and truncation quantile 0.9. Each validation set has
`int((1 - elimination_fraction) * n)` observations; its complement estimates the
regression. Every split requires at least three observations in each set and
positive residual degrees of freedom for the full regression. All fits must have
full column rank and positive, numerically resolved residual variance.

At each deletion, the predictor with the largest two-sided Student-t p-value is
removed. Ties remove the lowest original column index. The final full-data fit
stops when every retained p-value is at most the selected alpha. Validation
responses never enter coefficient estimation.

Candidates are the distinct deletion p-values across splits. The domain starts
at the specified quantile of split minima and ends at the complementary quantile
of split maxima, using linear interpolation between order statistics. As in the
source, when the smallest split maximum does not exceed the largest split
minimum, the upper endpoint is extended to one. An empty candidate domain raises
an error. Trimming retains sorted loss positions
`int(trim * repetitions):int(repetitions - trim * repetitions)`; this preserves
the original asymmetric rounding when the trim count is fractional. At least two
split losses must remain. The largest minimizing alpha wins exact score ties.

`alphas` and `scores` contain the candidates **inside** the domain. `split_alphas`
stores deletion p-values in full-model-to-empty-model order; `split_scores` has
p+1 columns, from zero deletions through intercept-only. These arrays support
plotting and inspection without recreating the original text-file interface.
Explicit `validation_indices` must have shape `(repetitions, validation_size)`;
each row contains distinct observation indices. They are mutually exclusive with
`seed`. NumPy's generator provides reproducibility, not the native RANLIB seed
sequence.

## Corrections and compatibility

Default `scoring="refit"` evaluates each reduced model using its refitted
elimination-set coefficients. Its threshold lookup follows the same stopping rule
as the final regression, including when successive deletion p-values are not
monotone. `scoring="native"` deliberately reproduces two source behaviors:

- `berds.c` removes a variable and scores the reduced model before refreshing the
  remaining coefficients and intercept. This uses coefficients from the preceding,
  larger model, and can change the chosen threshold.
- Its curve lookup scans deletion thresholds in reverse and assigns equality to
  the reduced model. This differs from the final regression's strict deletion
  rule and can differ further for nonmonotone paths.

Both modes report conventional MSE as SSE/(n − retained predictors − 1).
`native_mean_squared_error` separately exposes the original `back.c` value,
SSE/(n − retained predictors), which omits the intercept in the denominator.
Mallows' Cp uses the full-model residual variance and counts the intercept.
The source's no-intercept internal branch is not exposed by its command-line
program; Python supports its advertised intercept-fitting workflow.

Predictor centering/scaling and SVD fitting replace the native cross-product
sweep operations. This avoids forming normal equations and makes rank checks and
coefficient inference less sensitive to units. The implementation fits each
nested model once per split, then uses NumPy operations to combine score curves.
It supports up to 200 predictors and two million input predictor entries, with
additional limits on split and score-aggregation work. It raises on rank loss,
unresolved residual variance, or unrepresentable numerical outputs.

## Validation and provenance

The synthetic fixture `tests/fixtures/berds-native.json` contains 40 observations,
four predictors and six explicit validation splits. It was generated with
`default_rng(42)`: standard-normal X,
`y = 0.8 + X @ [0.5, -0.8, 0, 0.1] + 0.7 * standard_normal(40)`, then six
permutations whose first 20 indices define validation sets. Its native scores and domain
were calculated by compiling the original `back.c`, `berds.c`, `utils.c`,
`dcdflib.c` and `ipmpar.c`. Only the `ranf` provider was replaced with supplied
ranks to ensure identical splits; statistical C routines were unchanged.
All retained native score-curve values match Python to relative tolerance 1e-12;
the threshold values match to 1e-10. Independently, R 4.4.1 `lm` was used to refit
every reduced model on every split; those prediction losses agree to 1e-12.
The selected corrected model's coefficients and residual-df MSE also agree with R.
Focused checks additionally cover replay, predictor units from 1e-100 to 1e100,
response scaling, collinearity and exact-fit rejection. These are numerical
checks, not a claim about the operating characteristics of post-selection tests.

Source: [MD Anderson BERDS description](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/35)
and [original BERDS archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BERDS/BERDS_V1.tar.gz).
Archive and consulted-file hashes are in `berds-sources.json`. The original
archive has commercial-use restrictions; no original source, documentation or
sample data is redistributed. This implementation independently expresses the
statistical calculations in Python. The fixture is newly generated synthetic data.
