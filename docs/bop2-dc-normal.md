# Continuous Normal endpoints in BOP2-DC

`bop2_dc_normal_design` monitors a continuous outcome whose larger values
represent benefit. It implements the Normal–inverse-gamma model in §2.1.2 of
the [primary paper](https://arxiv.org/abs/2112.10880), with separate LRV and CMV
thresholds and go/consider/no-go decisions.

```python
from mdanderson_stats import (
    bop2_dc_normal_design, run_bop2_dc_normal_trial, simulate_bop2_dc_normal,
)

design = bop2_dc_normal_design(
    20, theta_lrv=0, theta_cmv=.5,
    prior_mean=0, prior_precision=.5, prior_shape=1.5, prior_scale=.75,
    looks=[10, 20], lambda_lrv=.8, lambda_cmv=.5,
    gamma_lrv=.5, gamma_cmv=.5,
)
trial = run_bop2_dc_normal_trial(design, [-1, .5, 2, 3] * 5)
print(trial.decision, trial.enrolled)

oc = simulate_bop2_dc_normal(design, true_mean=.3, true_sd=1.1,
                            n_trials=256, rng=913)
print(dict(zip(oc.decision_labels, oc.decision_probability)))
print(oc.decision_mcse, oc.mean_enrollment, oc.enrollment_mcse)
```

These inputs demonstrate the API; supplied cutoffs are not automatically
calibrated. The small simulation count gives a quick example, not a precise
error-rate assessment. [Normal parameter-grid calibration](bop2-dc-normal-calibration.md)
is a separate workflow with independent validation.

## Model and posterior

Observations are independent `Normal(theta, sigma²)`. The explicit prior is
`theta | sigma² ~ Normal(prior_mean, sigma² / prior_precision)` and
`sigma² ~ InvGamma(prior_shape, prior_scale)`, with density proportional to
`v**(-prior_shape-1) * exp(-prior_scale/v)`. Prior mean and both thresholds
may be negative. The other prior parameters must be strictly positive, and
LRV must be smaller than CMV.

For `n` observations, the conjugate update uses precision `k = k0 + n`,
shape `a = a0 + n/2`, location `(k0*m0 + n*mean(y))/k`, and scale
`b = b0 + SSE/2 + k0*n*(mean(y)-m0)**2/(2*k)`. The marginal mean posterior
is Student-t with `2*a` degrees of freedom and scale `sqrt(b/(a*k))`.
`posterior_scale` is this Student-t scale, not its standard deviation.

`design.monitor(observations)` returns both upper-tail probabilities and a
decision. It centers the data, prior mean and clinical thresholds on the same
observation before calculating probabilities. The returned
`posterior_location_centered` and `location_offset` preserve that representation;
`posterior_location` is their rounded absolute sum. Use the centered pair when
a large measurement offset would obscure small differences. Positive changes
of units require scaling the prior inverse-gamma scale by the square of the
unit change. Unrepresentable intermediate parameters raise an error.

## Decisions, replay and simulation

At configured interim looks, both probabilities must be strictly below their
time-scaled cutoffs to stop. Between looks the decision is `continue`. At the
maximum sample size, both must strictly exceed the final cutoffs for go, or
both must strictly fall below them for no-go. Equality or mixed evidence
produces consider.

`run_bop2_dc_normal_trial` takes a full potential-outcome vector and stops at
the first reached no-go analysis. It returns only reached analysis states;
future observations do not affect earlier decisions. Calling `monitor` alone
does not remember a previous stop.

`simulate_bop2_dc_normal` generates independent complete Normal outcomes in
coordinates centered on the truth mean, shifting the prior and clinical
thresholds by the same amount. This preserves simulated variation at large
absolute means. It replays each trial serially. Its four decision labels are early no-go, final
go, final consider and final no-go. Total no-go probability is the sum of early
and final no-go. Returned probabilities have binomial Monte Carlo standard
errors; expected enrollment has an ordinary sample standard error. Compact
per-trial decisions and enrollment are retained. `rng_seed` recreates the run
in the same numerical environment; a supplied Generator contributes one seed.

There is no accrual or delayed-observation model. Maximum sample size is 1,000
and simulations are bounded by 100,000 trials, one million potential patient
outcomes and 30 million repeated-look work units, checked before RNG use.
No candidate-by-patient-by-trial allocation is needed.

See the [independent numerical audit](../research/bop2-dc-normal-audit.md) for
posterior, decision, affine-unit and fixed-path replay evidence. Native app
prior defaults, optimizer behavior and random-seed parity are not inferred.
