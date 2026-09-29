# Randomized paired endpoints in BOP2-DC

Randomized paired designs compare treatment and control for two outcomes observed
on each patient. The model follows §§2.1.4, 2.2 and 2.4 of the
[BOP2-DC paper](https://arxiv.org/abs/2112.10880): each arm has its own joint
Dirichlet posterior, and the endpoint decisions use the corresponding marginal
Beta difference probabilities across independent arms.

The four outcome categories are **both events, first only, second only,
neither**. For efficacy/toxicity, the first event is efficacy and the second is
toxicity. Each arm needs four positive prior parameters; these are Dirichlet
shapes, not probabilities. A patient's category is coded 0, 1, 2 or 3 in that
same order. Allocation uses 0 for control and 1 for treatment. Looks count total
patients across both arms.

## Monitoring and replay

```python
from mdanderson_stats import bop2_dc_randomized_paired_design

design = bop2_dc_randomized_paired_design(
    4,
    "efficacy_toxicity",
    lrv=(0, 0),
    cmv=(0.3, -0.1),
    control_prior=(2, 1, 1, 2),
    treatment_prior=(1, 3, 2, 1),
    arm_assignments=(0, 1, 0, 1),
    looks=(2, 4),
    lambda_lrv=(0.35, 0.45),
    lambda_cmv=(0.4, 0.6),
    gamma_lrv=(0.5, 0.5),
    gamma_cmv=(0.5, 0.5),
    graduate_at_interim=True,
)
state = design.monitor((0, 0, 1, 0), (0, 1, 0, 0))
print(state.posterior_probability, state.decision)
trial = design.replay((2, 1, 3, 1))
print(trial.terminal_decision, len(trial.outcomes_observed))
```

This tiny example demonstrates the interface; its cutoffs have not been
calibrated to clinical error targets. `monitor` receives separate arm category
counts and checks them against the prefix of the allocation tape. Its posterior
and error arrays have axes `(endpoint, criterion)`, ordered first/second endpoint
and LRV/CMV. Between configured looks it returns continue. `replay` consumes a
complete potential outcome tape but retains only observations and analyses
reached before its first combined terminal decision. A one-time monitor call
does not remember earlier stops.

## Margins and endpoint decisions

All margins describe raw **treatment minus control** event probabilities.
For `multiple_efficacy`, both endpoints favor a larger difference and each
CMV must exceed its LRV. For `efficacy_toxicity`, efficacy favors a larger
difference, while toxicity favors a smaller difference: toxicity CMV must be
below toxicity LRV. A toxicity CMV of `-.1` requires a reduction of ten percentage
points relative to control. Its posterior criterion is the probability that
the raw toxicity difference is below the margin. The implementation evaluates
that lower tail directly to retain small probabilities.

Each endpoint has separate LRV/CMV probability cutoffs and information
exponents. For multiple efficacy endpoints, either endpoint can establish go;
both must establish no-go to stop for futility. For efficacy/toxicity, both
must establish go and either can establish no-go. Remaining final combinations
produce consider. Exact equality to a cutoff does not satisfy a strict go or
no-go condition.

Optional interim graduation combines each endpoint's O'Brien–Fleming superiority
rule using the same OR/AND composition. This combines the paper's general
endpoint-composition rule with its randomized superiority rule; it does not
claim a separately verified native paired-graduation implementation. Trial
replay absorbs the first combined no-go or graduation decision.

## Dependence and numerical interpretation

Four cell counts determine each arm's two marginal Beta posteriors. Marginal
posterior probabilities are used in the source's endpoint-decision composition;
they are not multiplied to create a posterior probability of joint benefit.
Operating characteristics must use the full joint outcome distribution within
each arm. Two pairs of joint distributions can have identical marginal event
rates but different trial decision probabilities.

The posterior comparison returns numerical error estimates. The combined
trial action must be unchanged at every corner of the four probability/error
intervals (two endpoints by two criteria). An individual endpoint may remain
numerically ambiguous while another makes the combined trial action definite.
Reported errors include quadrature estimates, not rigorous mathematical bounds.
`nominal_endpoint_decisions` (also available as `endpoint_decisions`) reports
the separate endpoint actions at the computed probabilities; only the combined
`decision` has passed the numerical-ambiguity check.

## Exact operating characteristics and calibration

```python
from mdanderson_stats import (
    bop2_dc_randomized_paired_operating_characteristics,
    optimize_bop2_dc_randomized_paired,
)

# Rows are control and treatment; columns retain the four-category order.
futile = ((0.10, 0.20, 0.15, 0.55), (0.12, 0.18, 0.18, 0.52))
effective = ((0.10, 0.20, 0.15, 0.55), (0.05, 0.60, 0.05, 0.30))
oc = bop2_dc_randomized_paired_operating_characteristics(
    design,
    (futile, effective),
)
print(oc.decision_labels, oc.expected_sample_size)

fit = optimize_bop2_dc_randomized_paired(
    4,
    "efficacy_toxicity",
    lrv=(0, 0),
    cmv=(0.3, -0.1),
    futile_joint_probabilities=futile,
    effective_joint_probabilities=effective,
    control_prior=(2, 1, 1, 2),
    treatment_prior=(1, 3, 2, 1),
    arm_assignments=(0, 1, 0, 1),
    looks=(2, 4),
    lambda_lrv_grid=((0.35, 0.45), (0.55, 0.65)),
    lambda_cmv_grid=((0.4, 0.6), (0.5, 0.7)),
    gamma_lrv_grid=(0, 0.5),
    gamma_cmv_grid=(0, 0.5),
    false_go_limit=0.1,
    false_no_go_limit=0.85,
    objective="cgr",
    graduate_at_interim=True,
)
print(fit.selected_index, fit.candidates.correct_go_rate)
```

This small example illustrates the calculation, including deliberately loose
error limits; it is not a recommended trial design. The effective truth must
satisfy the clinical-go rule: at least one efficacy difference reaches CMV for
multiple efficacy, or efficacy reaches its CMV and toxicity is at or below its
CMV for efficacy/toxicity. The futile truth is an explicit caller declaration.

Exact OCs propagate the four joint outcome categories along the supplied
allocation tape and remove paths after stopping or graduation. Output retains
a scenario axis even for one `(2,4)` truth: `decision_probability` has shape
`(scenarios, looks, 5)`, and `sample_size_probability` gives the probability of
terminal enrollment at each look. Decision labels distinguish interim no-go,
graduation, final go, final consider and final no-go. Expected enrollment sums
the terminal sample sizes weighted by those probabilities.

Calibration evaluates the Cartesian product of four explicit grids. Each
one-dimensional grid broadcasts its values to both endpoints; an `(m,2)` grid
instead supplies separate endpoint settings in each row. Candidate parameters,
all per-look probabilities, expected enrollment and constraint flags are
retained. Candidate probability arrays have axes `(scenario, candidate, look,
decision)`, with futile then effective scenarios. `cgr` maximizes correct-go
probability and then minimizes futile expected enrollment; `ess_futile`
reverses those priorities. Remaining ties retain input order, allowing only
relative floating-point roundoff tolerance.

False-go includes graduation and final go under the futile truth. False-no-go
includes interim and final no-go under the effective truth. An optional
false-consider limit applies to the larger final-consider probability across
the two truths. No feasible candidate raises
`BOP2DCRandomizedPairedInfeasibleError`. These are conditional probabilities
for the supplied truths and allocation; checking two scenarios does not
establish error control over every possible null distribution.

The four-dimensional exact state lattice grows rapidly with enrollment. The
implementation checks state, quadrature, candidate and working-memory bounds
before constructing it, and shares posterior comparisons across candidates.
Increasing a grid or sample size can exceed those bounds even when monitoring
the same design is inexpensive.

## Simulation for larger designs

```python
from mdanderson_stats import simulate_bop2_dc_randomized_paired

larger = bop2_dc_randomized_paired_design(
    40,
    "multiple_efficacy",
    lrv=(0.1, 0.15),
    cmv=(0.3, 0.25),
    control_prior=(1, 2, 3, 1),
    treatment_prior=(2, 1, 1, 3),
    arm_assignments=(0, 1) * 20,
    looks=(10, 20, 40),
    lambda_lrv=(0.35, 0.45),
    lambda_cmv=(0.4, 0.6),
    graduate_at_interim=True,
)
simulation = simulate_bop2_dc_randomized_paired(
    larger,
    (0.1, 0.2, 0.15, 0.55),
    (0.4, 0.2, 0.15, 0.25),
    n_trials=100,
    rng=2026,
)
print(simulation.terminal_names, simulation.terminal_probabilities)
print(simulation.terminal_mcse, simulation.expected_sample_size)
```

The simulation samples one joint category per patient under the fixed arm
allocation, preserving the specified endpoint association. It runs trials
serially and caches repeated marginal posterior comparisons; it does not build
the exact four-dimensional lattice. Trial count, patient work, cached posterior
comparisons and retained outputs are bounded before generating random seeds.
The example's 100 trials are enough to demonstrate the interface, not to
establish precise operating characteristics or error control.

Results include each trial's seed, terminal decision and enrollment, plus
aggregate terminal and per-look action counts, probabilities and Monte Carlo
standard errors. Per-look probabilities divide by **all simulated trials**,
not just trials reaching that look; `look_reached_counts` supplies the latter
denominator if needed. Monte Carlo errors describe simulation variability and
do not replace the posterior quadrature checks. Expected-enrollment MCSE uses
the sample standard deviation across trials and is undefined (`nan`) for one
trial.

To replay a saved trial, initialize `numpy.random.default_rng(int(trial_seed))`
and make one `choice(4, p=arm_probabilities)` draw per allocation-tape patient,
in order. Pass the complete category tape to `larger.replay`. Drawing the
potential outcomes after a trial would stop does not change its observed prefix.

The paper also describes arbitrary categorical cells with 0/1 endpoint
indicators. For two such indicators, aggregate the joint counts, prior shapes
and truth probabilities into their four indicator combinations above. Dirichlet
aggregation preserves the posterior and trial likelihood relevant to these
indicators exactly. Do not substitute arbitrary real-valued utility weights
for binary indicators; that is a different posterior calculation. More than
two decision endpoints require a separate generalized workflow.

All calculations are conditional on the caller's allocation tape and assume
complete paired observations at each scheduled analysis. Native randomization,
delayed-outcome handling, report formats and supplement-table settings are not
implied. The published supplementary tables are listed by Wiley but were not
available for inspection during this implementation.
