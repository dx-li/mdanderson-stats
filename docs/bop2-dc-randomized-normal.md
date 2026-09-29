# Randomized Normal outcomes in BOP2-DC

`bop2_dc_randomized_normal_design` compares the experimental mean minus the
control mean with two signed margins, following §2.4 of the
[primary paper](https://arxiv.org/abs/2112.10880). Each arm has an independent
Normal-Inverse-Gamma prior. The posterior mean difference is computed from
the two Student-t distributions by numerical integration.

```python
from mdanderson_stats import (
    bop2_dc_randomized_normal_design,
    simulate_bop2_dc_randomized_normal,
)

design = bop2_dc_randomized_normal_design(
    4, theta_lrv=0, theta_cmv=.5,
    control_prior=[.25, 4, 2, 3], treatment_prior=[.25, 4, 2, 3],
    arm_assignments=[0, 1, 0, 1], looks=[2, 4],
    lambda_lrv=.5, lambda_cmv=.5, graduate_at_interim=True,
)
trial = design.replay([0, .5, .25, 1])
print(trial.terminal_decision, len(trial.outcomes_observed))
oc = simulate_bop2_dc_randomized_normal(
    design, control_mean=.25, control_sd=1, treatment_mean=.5, treatment_sd=1,
    n_trials=12, rng=815,
)
print(dict(zip(oc.decision_labels, oc.decision_probability)))
print(oc.expected_sample_size, oc.enrollment_mcse)
```

The small trial count illustrates the interface. Cutoffs in this example have
not been calibrated to clinical error constraints.

## Priors, allocation and decisions

Each prior is `(mean, mean_precision, variance_shape, variance_scale)`:
`variance ~ IG(shape, scale)` and `mean | variance ~ Normal(location,
variance / mean_precision)`. All four values are explicit; precision, shape
and scale must be positive. The two arms may have different priors and variances.

The fixed assignment tape uses 0 for control and 1 for treatment. Looks are
total enrolled sample sizes and must end at the maximum. Allocation can be
unequal; probabilities from simulation are conditional on this tape. No
randomization schedule is generated internally.

`design.monitor(control_observations, treatment_observations)` uses fully
observed outcomes and validates arm counts against the allocation prefix.
An empty arm retains its proper prior. Between scheduled looks, the action is
continue. `design.replay(outcomes)` follows the scheduled looks and absorbs
the first no-go or graduation decision, retaining only observed outcomes and
reached states. Monitoring a state alone does not remember previous decisions.

Interim no-go requires both posterior probabilities strictly below their
information-scaled cutoffs. Optional graduation requires both strictly above
their O'Brien-Fleming cutoffs. At the final look, the outcomes are go, consider
and no-go; equality belongs to consider. The shared rules are described in
the [randomized binary guide](bop2-dc-randomized-binary.md).

Posterior states expose both probabilities and numerical error estimates,
arm counts, Student-t scales/degrees of freedom and the mean-difference
location. Arm locations are stored relative to a common `location_offset`;
the absolute location properties are convenient but can round at very large
offsets. Difference margins remain unchanged when a common location is added
to both arms. A difference of independent symmetric Student-t variables has
probability exactly one half above its center, even with unequal scales or
degrees of freedom.

## Simulation and numerical limits

Simulation draws independent complete Normal outcomes with explicit truth
means and SDs. It generates outcomes relative to the control truth and shifts
both priors to preserve residual variation under a common location shift.
Trials run serially; the default 100 trials is a Python workload choice.
Reusing `rng_seed` reproduces the call in the same numerical environment.

The five decision columns are `stop_no_go`, `graduate`, `final_go`,
`final_consider`, `final_no_go`. The result includes counts, probabilities and
binomial Monte Carlo standard errors. `look_decision_probability` and its MCSE
have shape `(looks, 5)` and describe the terminal decision at each look.
`sample_size_probability` sums those terminal decisions by look. Expected
sample size and its MCSE describe total enrollment; its MCSE is undefined
(`NaN`) for one trial. Add graduation and final go for total favorable probability.
`maximum_quadrature_error` covers every reached analysis. Output arrays are
owned and read-only; per-patient simulation histories are not retained.

Monitoring permits at most 1,000 planned patients and 100 looks. Each tail
uses bounded one-dimensional quadrature with an explicit truncation allowance.
The code raises if reported numerical error could change a combined decision;
the estimates are not rigorous error bounds. Simulation caps patient cells,
repeated-look scans, retained summaries and total quadrature work before
consuming the random stream. Resource caps can make feasible simulation counts
smaller than the nominal 100,000-trial limit. No Normal approximation to the
posterior difference is substituted when integration fails.

Independent R density integration, analytic Cauchy differences and deterministic
trial tapes provide validation; see the
[numerical audit](../research/bop2-dc-randomized-normal-audit.md).
Randomized Normal calibration remains open.
