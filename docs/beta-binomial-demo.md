# Beta Binomial Distribution Demo

Catalog **96** is implemented through the existing beta-binomial updating and
simulation engine plus an optional history plot. The source is the
[MD Anderson demo](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/96)
and its [version 1.0 guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BBDD/BetaBinomialDistributionDemoUsersGuide.pdf).
The Windows executable and guide are not redistributed.

## Step-by-step learning

```python
from mdanderson_stats import beta_binomial_sequence

# Guide pages 10–11: start with Beta(1,1), then observe three cohorts.
path = beta_binomial_sequence([10, 8, 12], [10, 12, 8])
print(path.posterior.alpha)  # [1, 11, 19, 31], including the original prior.
print(path.posterior.beta)  # [1, 11, 23, 31].
print(path.posterior.mean)
print(path.posterior.credible_set(0.95).intervals)
```

The two count inputs are **successes and failures**, not successes and total
patients. Each update adds successes to the first beta shape and failures to the
second. A single update can also use `BetaBinomialPosterior(a,b).update(s,f)`;
call `.update()` again on its result to continue. This reproduces the demo's use
of the previous posterior as the next prior.

The guide page 4 prints `(a+x)/(a+b+n-x)` for the posterior mean. This is
inconsistent with its correctly stated `Beta(a+x,b+n-x)` posterior. The correct
mean is `(a+x)/(a+b+n)`, which the package computes. Here `n=s+f`.

The demo's `alpha` is the excluded credible probability, so use
`credible_set(1-alpha)`. The Python `alpha` argument to the updating functions
instead denotes the **first beta shape**. Highest-density credible sets are the
default, with equal-tail sets also available. For a U-shaped beta density the
highest-density region can have two disconnected intervals; the result retains
both. Uniform priors use a central interval because the highest-density interval
is not unique. Credible mass must be strictly between zero and one, and beta
shapes must be positive. Endpoint singularities are legitimate densities, not
probabilities greater than one.

## Trial simulation and continuation

```python
import numpy as np
from mdanderson_stats import simulate_beta_binomial

rng = np.random.default_rng(123)
first = simulate_beta_binomial(
    0.5,
    cohort_size=20,
    cohorts=3,
    rng=rng,
)
continued = simulate_beta_binomial(
    0.5,
    cohort_size=20,
    cohorts=2,
    alpha=first.posterior.alpha[:, -1],
    beta=first.posterior.beta[:, -1],
    rng=rng,
)
print(first.successes, first.cumulative_successes, first.posterior.mean)
print(continued.posterior.mean[:, -1])
```

The true success probability is fixed across binomial cohorts, matching the
simulation tab. Set `trials` to simulate many independent trial paths in one
NumPy call. Posterior arrays include the original prior on their final axis;
cohort counts do not. Continue with the same generator and the last posterior
shapes. The continued object's cumulative counts describe only the newly
simulated segment; its posterior incorporates the earlier observations through
those supplied shapes. NumPy seeds are reproducible within this implementation,
but do not reproduce the native C# random-number stream.

## Prior and posterior display

The optional `plot` dependency provides Matplotlib rendering:

```python
from mdanderson_stats import plot_beta_binomial_sequence

# Display the third update from the worked example above.
ax = plot_beta_binomial_sequence(path, cohort=3)
ax.figure.savefig("beta-binomial-history.png", dpi=150)
```

`cohort=0` displays the original prior. Omit `cohort` to display the last update.
Earlier distributions are gray, the immediate prior blue, and the current
posterior red. Shading shows the current highest-density set; the title shows
cohort counts, cumulative counts, posterior shapes, mean, and credible endpoints.
For simulation output, select a trial explicitly, for example
`plot_beta_binomial_sequence(first, trial_index=(0,))`. More generally,
`trial_index` selects all leading axes of a broadcast sequence.

The function returns Matplotlib axes without displaying, saving, clearing, or
changing the backend. Use `ax.figure.savefig()` for export or Matplotlib's normal
interactive display. Repeated calls with different cohort indices replace the
desktop demo's staged viewing; Python calls control when to pause, continue, or
restart simulation. Windows menus and wall-clock animation controls are not
reproduced. Density curves use a uniform grid supplemented with beta quantiles
for narrow peaks. Infinite endpoint densities are omitted from the drawing and
annotated. Credible sets and summaries are computed independently of that grid.

Validation covers the guide's three-cohort posterior sequence, exact integer-beta
identities, highest-density mass and geometry, extreme endpoint complements,
fixed-probability simulation moments, and continuation. The worked-example and
U-shaped credible-set plots were rendered and inspected.
