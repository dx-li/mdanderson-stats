# Calibration of Bayesian success criteria

`success_calibration` implements the decision-error framework of Peng Yang,
Li Wang and Ying Yuan, [*On the Calibration of Bayesian Success Criteria and
Operating Characteristics for Clinical Trials*](https://arxiv.org/abs/2603.20015)
(v1, March 20, 2026). The [MD Anderson application](https://biostatistics.mdanderson.org/shinyapps/BayesianCalibration/)
identifies V1.0.1.0, April 6, 2026. This is an independent mathematical port.

The **design prior** generates the true effect and observations when evaluating
operating characteristics. The **analysis prior** determines the posterior
probability used to declare success. They can differ. Success always requires
posterior probability **strictly greater** than the cutoff; direction `greater`
means an effect above the clinical margin, and `less` means below it.

The result contains four unconditional joint probabilities (true/false positives
and true/false negatives), Bayesian power, Bayesian conditional power, Bayesian
Type I error, probability of incorrect decision (PID), false omission rate (FOR),
and frequentist success probability at the supplied null. In particular,
PID = FP/(TP+FP); it is not the Bayesian Type I error FP/(FP+TN). Zero-probability
conditioning events produce `None`, not a fabricated zero error rate.

## Binary outcomes

```python
from functools import partial
import numpy as np
from mdanderson_stats import binary_success_oc, calibrate_success_cutoff

operating_characteristics = partial(
    binary_success_oc,
    40,
    margin=0.3,
    design_prior=(2, 4),
    analysis_prior=(1, 1),
)
result = operating_characteristics(0.95)
calibrated = calibrate_success_cutoff(
    operating_characteristics,
    target=0.05,
    candidates=np.linspace(0.5, 0.999, 500),
)
```

`binary_success_oc` exactly enumerates the single-arm response counts using the
beta-binomial predictive distribution and incomplete beta probabilities. Both
posterior tails are evaluated directly. The frequentist null defaults to the
margin and can be explicitly changed with `null_rate`.

`binary_two_arm_success_oc(n_treatment, n_control, cutoff, ...)` implements the
paper's Section S2.2.2 **zero risk-difference margin** case. Supply separate beta
shape pairs as `design_treatment`, `design_control`, `analysis_treatment` and
`analysis_control`; all default to `(1,1)`. The null uses a common `null_rate`
(default `.5`) in the two arms. Every response-count pair is enumerated; posterior
ordering probabilities use the existing deterministic beta quadrature. A cutoff
within its numerical error estimate raises an error, except exact symmetry ties
are known to be `.5`. The default absolute integration tolerance is `1e-10`.

Two-arm enumeration is bounded at 40,000 count pairs and 1,000 patients per arm;
single-arm enumeration allows 100,000 patients. Two-arm computation is more
expensive than single-arm: each posterior comparison requires quadrature.
Repeated calibration calls currently recompute the two-arm comparison table.
This is a remaining optimization opportunity; no native-app speed comparison is
claimed. Absolute integration accuracy does not guarantee relative accuracy for
arbitrarily rare conditioning events.

## Normal outcomes and survival approximation

```python
from mdanderson_stats import normal_success_oc

# One normal mean with known patient SD=1 and n=74.
single = normal_success_oc(
    0.975,
    standard_error=1 / np.sqrt(74),
    design_mean=0,
    design_sd=0.15,
    analysis_mean=0,
    analysis_sd=1000,
)
# Two independent arm means, with separate prior distributions.
two_arm = normal_success_oc(
    0.95,
    standard_error=[0.2, 0.3],
    design_mean=[0.4, 0.1],
    design_sd=[0.3, 0.4],
    analysis_mean=[0, 0],
    analysis_sd=[1, 1],
    null_mean=[0, 0],
)
# Approximate log(HR) sampling SE from D=120 events and 2:1 allocation.
r = 2 / 3
survival = normal_success_oc(
    0.95,
    standard_error=1 / np.sqrt(120 * r * (1 - r)),
    design_mean=np.log(0.8),
    design_sd=0.2,
    analysis_mean=0,
    analysis_sd=1,
    direction="less",
)
```

For a scalar effect, `standard_error` is its sampling standard deviation (not
patient-level SD). With two-element inputs, positions are treatment and control;
the effect is treatment minus control. Scalar prior parameters broadcast. Arms
are independent with known sampling variances. Two-arm evaluation requires
explicit null means because their common level can matter with unequal analysis
prior weights. The frequentist quantity is evaluated at the supplied point;
choose means consistent with the intended null hypothesis.

The normal calculation forms the joint distribution of the true effect and
posterior-mean decision statistic. For unequal arm priors it preserves each
arm's separate conjugate weight. Four bivariate-normal quadrants are integrated
directly, avoiding subtraction of tiny tails from near-one probabilities.
Posterior weights use scaled standard deviations. Numerically singular
correlations or failed quadrature raise errors.

For survival, the paper approximates the log-hazard-ratio estimator as normal
with variance `1/(D*r*(1-r))`. This API evaluates that approximation; it does not
fit Cox models, simulate censoring, or estimate the required event count.

## Calibration, evidence and remaining coverage

`calibrate_success_cutoff` returns the smallest supplied candidate whose PID
meets the target and whose success probability is positive. Candidates are
sorted and deduplicated. It does not assume continuity or monotonicity of a
user-supplied evaluator, and raises when no candidate is feasible. This is a
grid-constrained search, not a claim of the original app's optimizer parity.

Validation includes exact single- and two-arm uniform-prior identities,
independent integration of single-arm rejection probabilities against the design
prior, the analytic normal quadrant identity involving `asin(rho)`, a 300,000-trial
independent simulation with unequal normal arm priors, direction-reflection
identities, a success probability below `1e-20`, and a discrete calibration case.
The separate beta-ordering routine has existing independent R validation.

This catalog entry remains **partial**: nonzero binary risk-difference margins,
original calibration-search parity, and native application report/plot parity
remain unverified or unimplemented. Source PDF SHA-256:
`eeb51252e5896c8fd2beedfd57d81de7f0d38d2978959af6a8478dbff22f4b81`.
No original application code or paper PDF is distributed.
