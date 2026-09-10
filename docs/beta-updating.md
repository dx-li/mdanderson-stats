# Bayesian beta-binomial updating

Catalog **101 (BU1BB)** and **102 (BU2BB)** are implemented as reusable Python
statistical APIs. The source specifications are the official
[BU1BB help](https://biostatistics.mdanderson.org/shinyapps/BU1BB/BU1BB.pdf) and
[BU2BB help](https://biostatistics.mdanderson.org/shinyapps/BU2BB/BU2BB.pdf), by
Yanhong Zhou and J. Jack Lee. [Source hashes](beta-updating-sources.json) pin the
retrieved documents. These are independent mathematical implementations; no
Shiny server source or R random-number sequence is claimed to have been ported.

BU1BB provides sequential updating, prior/data sensitivity, credible intervals
and simulated cohort histories. BU2BB compares independent treatment response
rates before and after observing data, including their ordering probability.
The Python equivalents expose distributions and complete histories for reuse
and plotting. Browser sliders, animation timing and button state are presentation
controls, not additional statistical methods, and are not reproduced.

## Updating and distribution summaries

```python
import numpy as np
from mdanderson_stats import BetaBinomialPosterior, compare_beta_binomial

prior = BetaBinomialPosterior(alpha=1, beta=1)
posterior = prior.update(successes=1, failures=2)
assert posterior.mean == 0.4  # Beta(2, 3)
updated = posterior.update(successes=7, failures=4)

# Broadcast priors/data to compare assumptions or treatment arms.
arms = BetaBinomialPosterior([1, 0.5], [1, 0.5]).update([1, 6], [3, 4])
x = np.linspace(0, 1, 501)
densities = BetaBinomialPosterior(arms.alpha[:, None], arms.beta[:, None]).pdf(x)
comparison = compare_beta_binomial(
    BetaBinomialPosterior(2, 4), BetaBinomialPosterior(6.5, 4.5)
)
print(comparison.treatment_greater)  # approximately 0.86272321
```

For a proper Beta(a,b) prior, successes s and failures f give Beta(a+s,b+f).
The result supplies `mean`, `variance`, `pdf`, `logpdf`, `cdf`, `sf` and `quantile`.
The upper tail is evaluated directly. Shapes broadcast and are stored in owned,
read-only arrays; updates return a new distribution. Counts are nonnegative
integers below 2**53. Improper zero-shape priors are rejected. Beta densities may
be infinite at the endpoints when a shape is below one.

## Credible sets

```python
region = posterior.credible_set(0.95)
print(region.intervals[0])
# [0.04379098, 0.77230669]
equal_tail = posterior.credible_set(0.95, method="equal-tail")
u_shaped = BetaBinomialPosterior(0.5, 0.5).credible_set(0.95)
# [0, 0.46077045] union [0.53922955, 1]
```

BU1BB's manual identifies its reported interval as highest density. A genuinely
highest-density region for a U-shaped beta density is disconnected. The Python
API therefore returns **sets**, with shape `broadcast_shape + (2, 2)` for two
possible components and their endpoints. The unused second component is NaN.
For a uniform density the answer is nonunique; the central interval is chosen.
Monotone densities use an endpoint interval. Unimodal and U-shaped cases use
vectorized probability-space bisection to match boundary log densities.

`complements` stores each endpoint's directly computed distance from one.
Use this when endpoints round to one: for Beta(1000, 0.01), a 50% region has a
lower endpoint indistinguishable from one in float64, but its complement remains
positive. Mass checks use the direct endpoint/complement pair. If even those
representations cannot resolve the mass, or boundary densities fail to converge,
the calculation raises `ArithmeticError`. Probability mass must be in (0,1).
This avoids silently reporting a zero-width region as an accurate result for
extremely singular or concentrated shapes.

## Cohort histories and simulation

```python
from mdanderson_stats import beta_binomial_sequence, simulate_beta_binomial

history = beta_binomial_sequence([1, 2, 0], [2, 1, 3])
# history.posterior contains the initial prior followed by each posterior.
regions = history.posterior.credible_set()

rng = np.random.default_rng(123)
trial = simulate_beta_binomial(0.3, cohort_size=3, cohorts=10, rng=rng)
continued = simulate_beta_binomial(
    0.3, cohort_size=3, cohorts=10,
    alpha=trial.posterior.alpha[:, -1], beta=trial.posterior.beta[:, -1], rng=rng,
)
```

The final axis of supplied counts represents cohorts; leading dimensions and
priors broadcast. Returned cohort counts, cumulative counts, observed rates and
all posterior shapes preserve the full history. A zero-size observed cohort is
allowed; its cumulative observed rate is NaN until any observations exist.
Simulation uses a fixed true response probability and independent binomial
cohorts, with optional `trials` for many experiments in one NumPy draw. Its prior
may vary by trial. Continuing with final shapes and the same generator preserves
the posterior and RNG state; cumulative counts in the new result describe the
new block. It does not reseed or animate. NumPy seeds do not reproduce R draws.

## Two-arm comparison and accuracy

`compare_beta_binomial(control, treatment)` integrates
`F_control(x) * f_treatment(x)` to obtain P(treatment > control). Independent
continuous distributions have zero tie probability. Both directions are
integrated directly and returned as `treatment_greater` and `control_greater`.

Logit coordinates remove beta endpoint singularities. Integration bounds discard
at most two explicitly budgeted beta tail masses. Logarithmic small-x identities
handle inverse quantiles clipped at float64's minimum normal value. Each result
includes an absolute quadrature error estimate plus the tail allowance; this is
not a rigorous bound on all floating-point error. The default absolute tolerance
is 1e-9 (accepted range 1e-12 to 1e-3). It does not promise relative accuracy for
probabilities much smaller than that tolerance. Convergence failure, invalid
probabilities or inconsistent complementary integrals raise `ArithmeticError`.
Shape updates, summaries, histories and credible-set searches are vectorized;
adaptive two-arm quadrature runs separately for each broadcast comparison.

## Focused validation

The 21 tests in `test_beta_binomial.py` and `test_beta_comparison.py` check the
manual examples, associative updates, exact rational binomial-tail identities,
analytic densities/quantiles, credible probability mass and endpoint density
conditions, monotone/uniform/arcsine cases, direct endpoint complements,
independent simulation moments, continuation, invalid input and array ownership.
Two-arm probabilities are independently checked by integrating integer-beta CDF
polynomials as exact rational beta moments, by the uniform-comparison mean
identity, and by reflection and concentration checks. Singular shapes down to
0.01 are included. These focused checks replace another unrelated full-suite run
for this addition.
