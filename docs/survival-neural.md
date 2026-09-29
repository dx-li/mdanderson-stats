# Neural survival models

`fit_survival_neural` fits one of the five neural families called by the
SurvivalContour author wrapper: `deepsurv`, `coxtime`, `deephit`, `loghaz`, or
`pchazard`. The implementation is a bounded NumPy multilayer perceptron with
full-batch Adam. It has no torch/pycox dependency. `predict_survival_neural`
returns point survival probabilities and `survival_neural_contour` returns the
existing contour data shape without confidence intervals.

```python
from mdanderson_stats.survival_neural import (
    fit_survival_neural,
    predict_survival_neural,
    survival_neural_contour,
)

time = [1, 2, 2, 3, 4, 5, 6, 7]
event = [1, 1, 0, 1, 1, 0, 1, 0]
x = [[-1.0], [-0.4], [0.2], [0.5], [0.8], [1.0], [1.3], [1.5]]
fit = fit_survival_neural(
    time, event, x, family="deepsurv", hidden_layers=(8,), max_epochs=80,
    patience=12, random_state=17,
)
survival = predict_survival_neural(fit, [[0.0], [1.0]], [1, 3, 5])
contour = survival_neural_contour(
    time, event, x, 0, family="deepsurv", n_grid=8, times=[1, 3, 5],
    hidden_layers=(8,), max_epochs=80, patience=12, random_state=17,
)
```

The result retains copied, read-only weights, feature scaling, fitted time
cuts, baseline log-hazard increments where applicable, loss history, best
epoch, epoch count, early-stop status, and seed. `patience` monitors training
loss: `stopped_early=True` means the patience threshold was reached, while
`False` means the epoch cap was reached. Neither status proves convergence.
Patience is evaluated on training loss only; there is no implied held-out
validation sample or validation-based early stopping. Inputs are numeric and
categorical variables must be encoded by the caller. The input matrix is
scaled using training rows only and the same transform is retained for
prediction. Durations remain floating point, extending the R wrapper, which
casts them to integer before preprocessing.

For `loghaz` and `deephit`, `n_cuts` is the number of training-only
Kaplan–Meier-survival quantile cuts; explicit strictly increasing `cuts` can
replace those cuts. Event labels round upward to a cut, censoring labels
downward, and exact cut hits stay on the cut. For `pchazard`, `n_cuts` means
the number of output intervals; its fitted cut vector has one extra endpoint.
PCHazard uses fractional exposure within the last interval. Events at or
before its initial cut are rejected instead of silently excluded from its
loss; censor records at that cut are excluded from its mean loss because they
have zero exposure. Durations beyond the final cut are treated as censored at
that cut even when the input says event, matching the backend transform.
Curves stay flat beyond the last trained cut. Its network outputs
integrated interval hazards, and prediction interpolates continuously within
each interval.

DeepSurv and CoxTime use exact full-risk-set Breslow objectives, including
subjects tied at an event time. CoxTime reevaluates every at-risk subject at
each event time. DeepHit uses the appended zero-logit tail cell and
source-compatible ranking normalization; its `alpha` controls
`alpha * likelihood + (1-alpha) * ranking`. Cox baselines are stored as log
increments for numerical stability. These are mathematical implementations,
not reproductions of the app's torch architecture, optimizer/RNG sequence,
preprocessing integer casts, or presentation rounding. Python defaults
(`hidden_layers=(32,32)`, `max_epochs=300`, Adam step size `.003`, patience
`30`) are explicit implementation choices; the R examples use 64-by-64,
dropout, batch normalization, shuffled mini-batches and an epoch limit of
1000. No uncertainty intervals are returned because the source contour wrapper
requests point predictions only.

Hard bounds cap fitting at 10,000 observations, 200 covariates, 200 requested
discrete cuts (201 for PCHazard endpoints), 20,000 epochs, 100 million
estimated work units and 128 MiB of estimated fit scratch. Prediction surfaces
are limited to two million cells, with separate network-work and scratch
limits. These limits protect memory and runtime; they do not promise that a
large fit is practical. The [source audit](../research/survival-neural-audit.md)
records provenance and model deviations.
