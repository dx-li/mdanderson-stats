# BOP2-DC with joint categorical endpoints

`bop2_dc_categorical_design` extends the paired implementation to an arbitrary
bounded set of binary endpoint indicators over a joint categorical outcome.
This follows the Multinomial–Dirichlet construction and endpoint-decision
composition in §§2.1.4 and 2.2 of the source paper. It does not claim parity
with an unavailable native calculator.

Each column of `indicators` is a joint outcome category, and each row selects
the categories counted as one endpoint event. Rows must contain only zero and
one, and each must select at least one but not all categories. For example,
with the five joint categories below, three rows define three clinical events:

```python
indicators = (
    (1, 1, 0, 0, 0),
    (0, 1, 1, 0, 0),
    (0, 0, 0, 1, 1),
)
```

The endpoint probability has a Beta posterior. Its two shapes are the prior
Dirichlet mass selected and not selected by that row, plus the corresponding
observed category counts. In randomized mode this calculation is done
independently in each arm and `compare_beta_difference` evaluates the
treatment-minus-control margin. `directions="greater"` uses the posterior
probability above each margin; `"less"` uses the probability below it. For a
less-direction endpoint, `cmv` must be numerically below `lrv`.

```python
from mdanderson_stats import bop2_dc_categorical_design

design = bop2_dc_categorical_design(
    6,
    indicators,
    combination="any",
    directions=("greater", "less", "greater"),
    lrv=(0.10, 0.80, 0.10),
    cmv=(0.30, 0.60, 0.25),
    prior=(0.2, 0.2, 0.2, 0.2, 0.2),
    looks=(3, 6),
)
state = design.monitor((1, 1, 0, 0, 1))
```

The prior is a vector of positive Dirichlet shapes, not probabilities. `any`
means any endpoint may establish benefit; `all` means every endpoint must
establish benefit. The same source-defined composition applies at interim and
final looks. Combined decision ambiguity is checked using the all-lower and
all-upper corners of the endpoint probability/error intervals. Since each
combined rule is monotone in favorable endpoint probabilities, those two
corners bound all intermediate combinations without enumerating `4**M`
corners. The reported quadrature errors are estimates, not formal error bounds.

Supply both `control_prior` and `arm_assignments` to select randomized mode.
The allocation tape is fixed before outcomes are generated; zero denotes
control and one treatment. Category counts passed to `monitor` have shape
`(2, K)` and must match the allocation prefix at that total sample size.
Intermediate looks use the source futility cutoffs; final looks use strict
dual-criterion cutoffs. Optional interim graduation uses the paper's fixed
single-arm cutoffs or randomized O'Brien–Fleming cutoffs, with the same `any`
or `all` endpoint composition. Endpoint actions use `no_go`; combined terminal
actions use `stop_no_go`.

`replay` takes a full potential category tape and stops at the first terminal
look. For randomized replay, outcome draws are generated in arm-grouped order
(control patients first, then treatment patients) and assigned to the fixed
allocation slots. The returned per-trial seeds therefore reproduce the tape
under this documented ordering; they do not represent chronological
one-patient-at-a-time random draws.

## Simulated operating characteristics and finite candidate comparison

`simulate_bop2_dc_categorical` accepts a caller-declared joint category
distribution (one vector per single arm, or a control/treatment matrix for an
RCT), runs serial Monte Carlo trials, and returns unconditional look and
terminal action probabilities with binomial Monte Carlo standard errors,
expected sample size, the generated trial seeds, and terminal sizes. The
probability vectors must sum to one. Fixed allocation is honored, so the RCT
truth is the pair of joint outcome distributions, not endpoint-wise marginal
rates.

`calibrate_bop2_dc_categorical` compares a finite supplied sequence of designs
that differ only in probability cutoffs and information exponents. The caller
declares a futile and an effective joint truth. The `cgr` objective maximizes
correct-go probability; `futile_ess` minimizes expected sample size at the
futile truth, subject to estimated false-go and false-no-go limits and an
optional false-consider limit. Feasibility uses Monte Carlo point estimates,
not confidence-bound guarantees. Reusing trials to select and report a design
creates selection optimism; use a new seed for independent confirmation.

The small serial examples below simulate one declared truth and compare two
supplied cutoff candidates. The truth labels are caller choices, and these
trial counts demonstrate the API rather than calibrating a real trial.

```python
from mdanderson_stats import (
    calibrate_bop2_dc_categorical,
    simulate_bop2_dc_categorical,
)

truth_futile = (0.10, 0.10, 0.10, 0.35, 0.35)
truth_effective = (0.35, 0.25, 0.10, 0.15, 0.15)
simulation = simulate_bop2_dc_categorical(
    design, truth_effective, n_trials=20, rng=20261003
)
candidate = bop2_dc_categorical_design(
    6,
    indicators,
    combination="any",
    directions=("greater", "less", "greater"),
    lrv=(0.10, 0.80, 0.10),
    cmv=(0.30, 0.60, 0.25),
    prior=(0.2, 0.2, 0.2, 0.2, 0.2),
    looks=(3, 6),
    lambda_lrv=0.85,
)
calibration = calibrate_bop2_dc_categorical(
    (design, candidate),
    truth_futile,
    truth_effective,
    n_trials=20,
    false_go_limit=1.0,
    false_no_go_limit=1.0,
    rng=20261004,
)
```

These serial workflows have explicit caps on candidate count, trial count,
patient paths, result storage, and posterior comparison work. They are bounded
research interfaces, not an optimizer over an implicit native design grid.
