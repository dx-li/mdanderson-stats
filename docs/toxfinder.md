# ToxFinder: two-agent toxicity modelling

Catalog entry [14](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/14)
is **partially implemented**. The package provides the six-parameter toxicity
surface, grouped likelihood, Bayesian posterior fitting, first-stage decisions
and target-contour calculations. Priors are explicit;
there is no universal default prior. Source details and the distinction between
the paper and the later software guides are recorded in
[toxfinder-sources.json](toxfinder-sources.json).

## Model and data

Divide each physical dose by its corresponding single-agent acceptable dose.
For standardized doses `(x1, x2)` and parameter order
`(alpha1, beta1, alpha2, beta2, alpha3, beta3)`, the toxicity odds are

```text
q = alpha1*x1**beta1 + alpha2*x2**beta2
    + alpha3*(x1**beta1*x2**beta2)**beta3
p = q / (1 + q)
```

`toxfinder_probabilities` and `toxfinder_log_probabilities` return cells in
`(no toxicity, toxicity)` order. Doses have shape `(N, 2)`; parameter arrays may
have leading batch dimensions, giving outputs `(..., N, 2)`. All doses are
nonnegative. Prediction permits zero alpha coefficients, as in the official
scenario generator; beta coefficients must be positive. At the origin toxicity
is exactly zero, and on either axis the other agent and interaction terms vanish.

`toxfinder_log_likelihood` accepts grouped toxicity and subject counts, omitting
the parameter-independent binomial coefficients. Historical single-agent data
can be included as dose rows with the other agent at zero. Do not include the
same observations both in an already-updated prior and in the likelihood.

## Explicit priors and fitting

Each parameter has an independent gamma prior specified by its **mean and
variance**, converted to shape `mean**2 / variance` and scale `variance / mean`.
A zero variance fixes a strictly positive parameter at its mean; this is an
explicit Python extension. Fitting requires strictly positive means, including
fixed coordinates.

```python
import numpy as np
from mdanderson_stats import ToxFinderPrior, fit_toxfinder, toxfinder_standardize

# Paper Table1 single-agent priors plus its interaction prior.
prior = ToxFinderPrior(
    mean=[.4286, 7.6494, .4286, 7.8019, 1, .05],
    variance=[.1054, 5.7145, .0791, 3.9933, 3, 3],
)
doses = toxfinder_standardize([[300, 150], [480, 240]], [1200, 600])
fit = fit_toxfinder(
    doses, toxicities=[0, 1], subjects=[2, 2], prior=prior,
    rng=np.random.default_rng(2026), draws=500, warmup=250, chains=2,
)
posterior_mean_toxicity = fit.probabilities[..., 1].mean(axis=(0, 1))
```

The case-study guide instead uses interaction beta3 mean **1**, variance **0.9**.
That is a materially different prior. The paper's beta3 prior has shape about
`0.0008333`, with considerable mass below the smallest positive floating-point
number. `fit.log_parameters` therefore holds the authoritative draws. Use them
with `log_parameters=True` for subsequent predictions. Exponentiating through
`fit.parameters` can underflow and is not a suitable route for reusing those
draws in calculations.

Posterior fitting uses componentwise slice sampling in log coordinates, including
the transformation Jacobian. This differs from the original proposal mechanism;
the target density is the published model. With no observed subjects, independent
log-gamma draws replace unnecessary posterior sampling. The returned chain
summary describes **log parameters**, and retained arrays are read-only. Small
draw counts are useful for smoke checks; inspect mixing and Monte Carlo error
before interpreting estimates. Agreement for one reduced model does not establish
mixing for every six-parameter posterior.

## First-stage decisions and target contours

`toxfinder_stage1(dose_levels, parameters, treated_doses, toxicities, *, target,
log_parameters=False, start_index=0)` accepts ordered dose pairs on a straight
line and chronological treatment history. A row can represent a patient or a
cohort; its toxicity count identifies whether toxicity occurred. Supply fitted
parameter draws separately, using the complete corresponding data for the fit.
The helper does not fit the posterior or decide when to switch stages.

For empty history, it selects the specified starting row. Otherwise it minimizes
distance of posterior mean toxicity from the target while disallowing upward
skips of untried discrete levels. Following the first toxicity, it adds midpoints
above that dose and allows continuous doses below it. Means at the higher
midpoints are evaluated directly. The lower continuum uses interpolation of
node means, following the paper's computational procedure. Thus the returned
`estimated_toxicity` at an interpolated dose can differ from a fresh model
evaluation at that dose. The result exposes candidates, their means and
eligibility for inspection. The line may have a nonzero offset.

```python
from mdanderson_stats import toxfinder_stage1, toxfinder_contour

# A simple single-agent surface embedded in the two-agent model: p=x1/(1+x1).
theta = [1, 1, 0, 1, 0, 1]
levels = [[.1, .1], [.3, .3], [.6, .6], [1, 1]]
decision = toxfinder_stage1(
    levels, theta, levels[:3], [0, 0, 1], target=.2,
)
# decision.dose is [.256, .256]; interpolated toxicity is .2.

# The additive surface q=x1+x2 has target .5 on x1+x2=1.
contour = toxfinder_contour([1, 1, 1, 1, 0, 1], [0, .25, 1], target=.5)
# contour.doses: [[0, 1], [.25, .75], [1, 0]]
```

`toxfinder_contour` finds the second-agent dose for each requested first-agent
dose within explicit `x2_bounds`. It uses direct posterior means, not stage-one
interpolation. Unbracketed points have a false `attainable` flag and a missing
second dose. Endpoint crossings are included; for a flat target segment the
lower endpoint is returned. A root with excessive toxicity residual raises an
error rather than reporting convergence from dose tolerance alone. This supplies
contour geometry; it does not implement native second-stage selection.

## Remaining coverage

Automated physician-prior elicitation, the native second-stage information
criterion, full trial simulations, and TMML/CSV/GUI parity remain open.

The second-stage information criterion requires further source clarification.
As written, equation12 in the paper is a six-dimensional outer product, with
rank at most one; equation13's single-patient log determinant cannot give a finite
ranking. This is our mathematical inference from the displayed formula. The
software guide explicitly substitutes second derivatives, but does not provide
enough detail to reproduce its calculation. The Python implementation does not
silently add a ridge or claim an alternative criterion reproduces ToxFinder.

## Numerical evidence and resource bounds

[reference_toxfinder.R](../tools/reference_toxfinder.R) uses base R to evaluate
the four official case-study surfaces and zero-alpha scenarios, integrate a
one-dimensional nonconjugate posterior, and calculate analytic log-gamma moments.
These independent references supplement focused checks of zero doses, log tails,
posterior sampling and input limits. No original executable has been run.

Prediction is limited to 100 dose pairs and 200,000 parameter-dose combinations.
Fitting uses at most 100 grouped rows, 10,000 subjects and four chains, with
bounded retained draws and evaluation work. Gamma shapes below `1e-8` or
unrepresentable shape/scale conversions are rejected. Stage one accepts 2–50
original levels and at most 100 history rows; contour searches bound both point
counts and total root work. Limits are checked before large
working arrays are formed. Numerical checks are run sequentially with numerical
thread counts set to one; no large simulation or full-suite run is needed for
this addition.
