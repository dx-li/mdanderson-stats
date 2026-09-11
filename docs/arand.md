# Adaptive Randomization: posterior core

Catalog entry **62 remains partial**. The binary and time-to-event posterior
models, best-arm probabilities and exponential tuning are implemented for one
through ten independent arms. Native calendar simulation, allocation floors,
reversible loser suspension, permanent futility, stopping order and full
operating-characteristic reporting still require implementation and verification.
These functions are statistical building blocks, not a trial controller.

Sources: [MD Anderson entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/62),
[version 5.2 guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/ARAND/ARandUsersGuide.pdf),
and [version 5.2.2 installer](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/ARAND/AdaptiveRandomization_V5.2.2.zip).
The MSI contains managed design/UI assemblies and a native C++ simulation kernel
inside `ARand.Calcs`; interpreting only the managed UI is insufficient to
establish controller behavior. Original files are kept outside the distribution.

## Binary outcomes

```python
import numpy as np
from mdanderson_stats import arand_binary_posterior

posterior = arand_binary_posterior(
    successes=[3, 6, 8],
    failures=[7, 4, 2],
    prior=np.full((3, 2), 0.5),
    tuning=0.5,
    threshold=0.3,
)
print(posterior.best.probability)
print(posterior.allocation_probability)
print(posterior.exceedance_probability)
```

Each prior row is `(alpha, beta)`. Observed successes and failures increment the
respective parameters. Pending outcomes contribute neither count. By default,
`best.probability[i]` is the probability arm i's response probability exceeds
**every** other arm's. Set `maximize=False` to prefer the lowest probability,
for example when the outcome is toxicity.

The optional `threshold` returns `Pr(theta_i > threshold)` per arm. Its direction
always means greater-than, even with `maximize=False`. This is a probability
calculation; it does not declare or persist futility. The original guide applies
its futility rule only when maximizing.

## Exponential time-to-event outcomes

```python
from mdanderson_stats import arand_survival_posterior

posterior = arand_survival_posterior(
    events=[4, 2, 5],
    exposure=[20, 25, 30],
    prior=[[2, 5], [2, 5], [2, 5]],
    parameter="mean",
    threshold=4,
)
print(posterior.parameters)  # [[6, 25], [4, 30], [7, 35]]
print(posterior.best.probability)
```

The exponential likelihood has an inverse-gamma prior on its mean or median
time parameter. Rows are `(shape, scale)`, with density proportional to
`theta**(-shape-1) * exp(-scale/theta)`; scale is not a gamma rate on theta.
An observed event increments shape. Total observed time at risk, including
right-censored follow-up, increments scale for the mean-time parameterization.
For `parameter="median"`, exposure is multiplied by `log(2)` before updating
scale. The prior scale and threshold must already refer to the median time.
Changing an entire mean-time analysis to median time therefore also requires
multiplying prior scales and the threshold by `log(2)`.

`maximize=False` prefers the smallest mean/median time. No sample mean, posterior
mean or posterior variance is required for the comparison, so proper
inverse-gamma posteriors whose moments do not exist remain supported. Inputs are
aggregated observed events and exposure; arrival/censoring records are not
processed automatically by these functions.

## Best-arm probability and tuning

`arand_best_probability(parameters, family="beta" | "inverse_gamma",
maximize=True)` evaluates already specified independent distributions. It
returns immutable probability and absolute-error arrays. Each maximum
probability is a one-dimensional integral of the arm's density times the
**product of the other arms' CDFs**. For a minimum, use the product of survival
functions. Multiplying pairwise winning probabilities would be incorrect because
the pairwise comparisons share the same random parameter.

All-identical distributions use exact exchangeability, giving `1 / arms`.
Two-arm comparisons reuse the validated beta quadrature and inverse-gamma/beta
identity. For more arms, integration uses the chosen arm's uniform probability
coordinate, with competitor-quantile breakpoints to expose sharp CDF transitions.
Beta coordinates are represented as logits and inverse-gamma coordinates as log
times. Competing-arm CDFs are vectorized; identical arms reuse the same integral.
Unrepresentable quantiles or failed integration raise an error.

Quadrature reports include an explicit bound on discarded integration tails;
the quadrature portion is an estimate, not a rigorous floating-point bound.
Probabilities must pass a sum-to-one check and are not renormalized to conceal
failure. `absolute_tolerance` defaults to `1e-9`; it does not guarantee relative
accuracy for arbitrarily rare winning events. A probability returned as zero
can represent a tail below floating-point range.

For the posterior wrappers, raw new-patient allocation is proportional to
`best_probability**tuning`. The default tuning is .5; zero gives equal allocation.
Log weights are shifted before exponentiation, avoiding overflow for large
finite tuning. For positive tuning, unresolved winning tails (zero or no larger
than their quadrature error estimate) raise an error instead of amplifying
uncertain tails with a fractional power. A tighter tolerance may resolve the
issue; `arand_best_probability` remains available to inspect raw probabilities
and errors, and tuning zero does not require resolving relative tail accuracy.

These allocations precede the original controller's equal-allocation period,
minimum allocation probabilities and arm-status rules. No floor, suspension or
winner selection is silently substituted. The existing parameter-distribution
solvers can be used separately when priors are specified by moments or quantiles.

## Numerical evidence and remaining scope

[Independent R integration](arand-posterior-reference.json) evaluates 52 winning
probabilities for beta and inverse-gamma models, three and ten arms, maximizing
and minimizing. R integrates in the original parameter domain; Python uses
uniform quantile coordinates. Maximum absolute disagreement is about 6.3e-11.
Reproduce fixtures using `Rscript tools/reference_arand.R`.

Focused tests additionally use exact polynomial beta identities, exchangeability,
posterior updates, censored exposure, mean/median conversion and time-scale
changes of 1e200 and 1e-200. They verify that unresolved tails are rejected before
fractional-power allocation. Existing two-arm inequality tests are reused.
Vectorized CDF evaluation reduced one local pass through the 52-reference
workload from 2.62 to 1.12 seconds; this is a workload measurement, not a general
performance guarantee or native program benchmark.

The archive's design controls and full simulation kernel have not been ported.
The remaining work includes native timing and accrued-outcome rules, allocation
floor semantics, suspended/futile-arm inclusion in probability calculations,
minimum-enrollment gates, final follow-up and selection, and simulation reports.
No native numerical-engine or complete trial-controller parity is claimed.
