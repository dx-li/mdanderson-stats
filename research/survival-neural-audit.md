# SurvivalContour neural methods audit

## Provenance

The author project is pinned at [`YushuShi/survivalContour`
`d4645f69f23fc1146c07432f576b4c40f85e1bba`](https://github.com/YushuShi/survivalContour/tree/d4645f69f23fc1146c07432f576b4c40f85e1bba).
Its `R/survivalContour.R` dispatches `coxtime`, `deepsurv`, `deephit`,
`loghaz` and `pchazard`; `R/pycoxContour.R` requests model survival
predictions. Training examples show ReLU, two 64-unit layers, dropout `.2`,
batch normalization, shuffled batches of 256, early stopping, and up to 1,000
epochs. Those are examples, not a guarantee of exact model configuration for
every family.

The inspected `survivalmodels` wrapper is pinned at
[`d8a6a4a36368fb57253e0ebf3f0f32e05f05c561`](https://github.com/cran/survivalmodels/tree/d8a6a4a36368fb57253e0ebf3f0f32e05f05c561)
(package version 0.1.191). It casts durations to integer before preprocessing;
its `R/helpers_pycox.R` prepares the selected network, fits model-specific
duration transforms on training outcomes, and sends validation rows through
those fitted transforms. This Python implementation intentionally preserves
finite fractional times.

Backend mathematics was checked against pinned PyCox 0.3.0 at
[`3eccdd7fd9844a060f50fdcc315659f33a2d2dc1`](https://github.com/havakv/pycox/tree/3eccdd7fd9844a060f50fdcc315659f33a2d2dc1),
including `models/cox_time.py`, `models/logistic_hazard.py`,
`models/pc_hazard.py`, `models/loss.py`, `models/data.py`,
`preprocessing/label_transforms.py` and `preprocessing/discretization.py`.
The project keeps the inspected source files under ignored
`research/raw/pycox` where available. The neural implementation and
independent 85-digit likelihood/gradient references live in the Python source
and `tests/fixtures/survival-neural-*.json`.

## Implemented objectives and choices

DeepSurv uses the full Breslow risk set `T >= t`, grouping tied event times.
This is a conventional exact risk-set objective, whereas the pinned PyCox
DeepSurv source uses a descending cumulative-risk approximation. CoxTime
conditions the network on each distinct event time and evaluates every member
of that event-time risk set, including ties. Its Breslow baseline is evaluated
at each event time with that same time-varying score. These full-risk-set
objectives are explicit Python mathematical choices and do not reproduce the
app's mini-batch/case-control or optimizer streams. They divide the partial
likelihood by the number of observations; pinned PyCox divides its Cox loss by
the number of events. This positive scaling leaves the unpenalized minimizer
unchanged but changes finite-step optimizer and stopping scales.

LogisticHazard uses conditional Bernoulli likelihood through each row's
rounded event or censor cut. DeepHitSingle adds the fixed zero-logit tail cell,
uses event probability mass or right-tail mass for censoring, and implements
PyCox's `alpha*NLL + (1-alpha)*ranking` objective. Comparable-pair ranking
uses the `n²` normalization including ineligible zero pairs. PCHazard uses
softplus interval increments and fractional exposure within the last interval;
the returned curve interpolates within each interval. A censor at its initial
cut has zero exposure. An event at or before that cut is rejected rather than
being silently omitted from the native PyCox loss mask. Censor records exactly
at the PCHazard initial cut are excluded from its mean loss, matching the
backend's zero-exposure mask. Outcomes beyond the last cut become censored at
that cut even when the input event flag is one; curves remain flat beyond the
last trained cut.

Discrete cuts use the training sample's Kaplan–Meier survival-scale quantile
construction and are retained in the fit. The wrapper converts durations to
integers before those transforms; this Python API preserves finite fractional
durations. Features use training-only max-absolute unit scaling followed by
training mean/standard-deviation scaling. CoxTime appends standardized
`log1p(time)` at each event time; both its time center and scale are retained.
Tied events use Breslow groups and risk sets `time >= event_time`. Input bounds
and public fit choices are Python conventions.
The implementation uses a small full-batch Adam MLP and exposes the epoch
limit, patience and diagnostics. It does not claim native network, dropout,
batch-normalization, shuffling, RNG, contour-layout or parity. Survival
predictions are point estimates; the source contour helper requests survival
predictions, not uncertainty intervals. Native application calibration and
report parity remain open.

## Independent checks

`tools/reference_survival_neural.py` computes the five mathematical
likelihoods and their derivatives using only the Python standard library and
high-precision `Decimal`; it does not import the package, NumPy or SciPy.
Fixtures include tied Cox events, fitted baseline log increments and 48
independent Cox/CoxTime profile-time predictions. Discrete fixtures exercise
initial-cut censoring, exact-cut censoring, truncation after the last cut,
ranking and PCHazard fractional exposure. Focused package tests compare loss,
logit/weight gradients, baseline increments, predictions, all-family seeded
fitting and contour shape. These validate the stated mathematical objectives,
not native app parity.

## Integration validation — September 29, 2026

Root independently compared all five Decimal loss references (exact after
conversion to double precision), 56 derivatives (maximum absolute error
1.12e-16), six log-baseline increments (2.23e-16), and 48 predictions
(1.12e-16). Common Cox score shifts of +1000 and -1000 preserve predictions.
A DeepSurv query with 2,000 profiles and 10,000 event times evaluates one
requested time per profile without allocating the 20-million-cell dense
profile-by-baseline matrix. Extreme PCHazard logits of -1000 retain finite
likelihoods and the expected gradients. This independent process took
1.532 seconds, peaked at 123.22 MiB RSS and reported zero swaps.

The integrated ten focused tests pass with warnings treated as errors and
one BLAS thread: 1.94 seconds total process time, 145.84 MiB peak RSS and zero
swaps. Targeted Ruff format/check and mypy checks pass. An additional extreme
DeepHit ranking example produces finite loss and gradients but overflows the
Adam squared-gradient accumulator; fitting now rejects that update explicitly
instead of retaining infinite optimizer state. The dedicated probe confirms
this rejection. Public exports include both result types and all three API
functions. A separate read-only review found no material integration blocker.

These checks validate the documented likelihoods, derivatives and bounded
implementation. They do not establish equality to stochastic native training
runs, clinical performance, or statistical convergence of every fitted network.
