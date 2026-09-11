# ROSE dose selection

The Python port covers the main ROSE application's one- and two-stage design
calculations, decisions and binary-response simulations, with exact operating
characteristics available for both normal-approximation and exact designs.

Source: Wang, Yuan and Liu, *ROSE: Randomized Optimal Selection Design for Dose
Optimization* (2025), [paper and supplement](https://arxiv.org/abs/2505.03898), and
[MD Anderson application](https://biostatistics.mdanderson.org/shinyapps/ROSE/).
The application identifies version V1.2.1.0, January 6, 2026. This is an independent
implementation of the published mathematics; original backend and RNG parity
have not been established.

```python
from mdanderson_stats import (
    rose_design,
    rose_operating_characteristics,
    rose_select,
    simulate_rose,
)

design = rose_design(
    p_high=0.3,
    margin=0.1,
    pcs_low=0.6,
    pcs_high=0.6,
    interim_fraction=0.5,
    method="exact",
)
assert design.n_low == design.n_high == 19
low_scenario = rose_operating_characteristics(design, 0.3, 0.3)
high_scenario = rose_operating_characteristics(design, 0.2, 0.3)
assert low_scenario.select_low >= 0.6
assert high_scenario.select_high >= 0.6
simulation = simulate_rose(design, 0.2, 0.3, trials=10000, rng=168)
decision = rose_select(design, responses_low=4, responses_high=8)
```

## Design and decision conventions

The low scenario has equal response probabilities at the two doses. The high
scenario has low-dose response probability `p_high - margin`. `pcs_low` and
`pcs_high` specify correct selection probabilities in these scenarios. The
method assumes the candidate doses have already been judged safe; it does not
model toxicity. Outcomes are independent Bernoulli responses with complete
assessment at each analysis.

`ratio` is high:low enrollment. Final low-arm sample size is searched over
integers, high-arm sample size is rounded upward from `ratio * n_low`, and each
interim count is rounded upward from its final count times `interim_fraction`.
Both stages must enroll patients in both arms. Set `interim_fraction=None` for a
single analysis. Counts are limited to 1000 per arm; `max_per_arm=200` is the
default design-search bound. An infeasible bounded search raises an error.

Select high only if its observed response rate minus the low response rate is
**strictly greater** than the boundary. Ties select low at final analysis and
continue at interim. The only early selection is high. Supplied decimal
boundaries are compared with integer response counts without floating-point
rounding changing a discrete tie. `rose_select(..., interim=True)` evaluates the
planned interim; it does not maintain a patient's or trial's history.

Normal one-stage design uses the analytic solution; normal two-stage design
uses deterministic bivariate-normal integration and O'Brien–Fleming spending.
As in the paper, the normal power search uses continuous information fractions
before rounding interim enrollment. Its boundary retains full numerical
precision, rather than the three decimals printed in the paper.

**Normal designs need not achieve their nominal PCS under the exact binomial
model.** For example, `p_high=.4`, `margin=.1`, both PCS `.65`, and an interim at
`.5` gives 31 patients per arm. Its exact high-scenario PCS is about `.6118`.
Use `rose_operating_characteristics` to assess a rounded design and
`method="exact"` when exact scenario constraints are required.

The exact search uses final boundaries on `0, step, 2*step, ... <= margin`
(default step `.002`). For a two-stage design it first chooses the smallest
interim grid boundary satisfying the low-scenario O'Brien–Fleming early-error
budget; the interim grid additionally includes 1. It then chooses the smallest
feasible final grid boundary at the first feasible sample size. Final boundaries
cannot exceed the interim boundary. This is optimal within this specified
search and spending convention, not over every continuous boundary or every
possible interim policy. The supplement's alternative jointly optimized/free
interim-boundary searches are not exposed by this API.

## Numerical validation and simulation

Exact operating characteristics marginalize stage response-count differences,
including unequal arm sizes and unequal rounded interim fractions. Both tails
are accumulated directly, avoiding subtraction of a tiny upper tail from one.
The API returns low/high selection, early high selection and expected enrollment
in each arm. Probabilities can be supplied in either order for robustness checks.

`simulate_rose` samples independent binomial stage outcomes, stops early for high,
and only enrolls the remaining patients in continuing trials. It returns
empirical operating characteristics, the number of trials, and Monte Carlo
standard errors for high selection and early selection. Low selection has the
same MCSE as high selection. A seed makes Python simulations reproducible; it
does not reproduce the original application's random stream.

Focused checks cover eight normal designs from published Table 1 and the exact
example (23 patients per arm for one stage, 19 for two stages). A separate direct
joint-count enumeration checks unequal-arm one- and two-stage probabilities to
absolute tolerance `2e-14`. Simulations of 100,000 trials agree within six MCSEs;
decimal ties and deterministic response scenarios are also checked. These are
mathematical and simulation checks, not an audit of the original Shiny backend.
