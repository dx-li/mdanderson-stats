# BOP2 binary sample-size optimization

BOP2 catalog entry **112** now supports expected-sample-size and minimax searches
for binary efficacy and binary toxicity endpoints. These implement the variable
sample-size criteria in section 2.3 of
[Zhou, Lee and Yuan (2017), DOI 10.1002/sim.7338](https://onlinelibrary.wiley.com/doi/10.1002/sim.7338)
([paper mirror](https://eurekamag.com/research/059/454/059454786.pdf)).
Sample-size searches for the complex endpoint types remain pending.

## Search over maximum sample sizes

```python
from mdanderson_stats import optimize_bop2_binary_sample_size

result = optimize_bop2_binary_sample_size(
    range(10, 51),
    null_rate=0.2,
    alternative_rate=0.4,
    minimum_power=0.8,
    type1_error=0.1,
)
fit = result.best
print(fit.calibration_design.max_subjects)
print(fit.calibration_design.looks, fit.calibration_design.futility_max)
print(fit.calibration_success_probability)  # null error, alternative power
print(fit.calibration_oc.expected_sample_size)
print(result.feasible_sample_sizes)
```

The search jointly considers maximum sample size `N`, cutoff scale and exponent.
Every candidate must have exact numerical null error at or below `type1_error`
and alternative success probability at or above `minimum_power`.

- `objective="expected_sample_size"` minimizes expected enrollment under the null.
  Ties prefer smaller maximum sample size, then higher power, then input parameter
  grid order.
- `objective="minimax"` minimizes maximum sample size. Ties prefer lower expected
  enrollment under the null, then higher power, then input parameter grid order.

The search retains the minimum-expected-enrollment feasible design at each `N`,
then compares these designs under the chosen criterion. It does **not** first
maximize power at each `N`: that approach can miss a design that meets the power
requirement with fewer expected subjects.

`sample_sizes` must be a nonempty, strictly increasing integer vector in `[1,200]`.
By default, each design starts monitoring after 10 subjects, every five thereafter,
and always at its final sample size. `min_subjects` and `cohort_size` customize
that schedule. Each candidate `N` must be at least `min_subjects` with this default
schedule construction.

Alternatively, `interim_looks` supplies one strictly increasing vector of planned
interim counts. For each `N`, looks below `N` are retained and the final look `N`
is appended. An empty vector requests final-analysis-only designs. This option
supersedes `min_subjects` and `cohort_size`.

The same `cutoff_scales` and `gammas` grid is searched at each size, using the
[existing binary defaults](bop2-binary.md): 50 scales and 21 exponents. Repeated
stopping boundaries are evaluated once per size. Each parameter grid permits at
most 100,000 pairs. The result is optimal within the declared size, parameter and
schedule choices; no continuous optimum or native app optimizer parity is claimed.

All requested sizes are searched. Discreteness can make feasibility nonmonotone
on a finite parameter grid, so the search does not assume that every size above
the first feasible size is feasible. `searched_sample_sizes` records the full grid;
`feasible_designs` contains one optimizer result per feasible size, in ascending
size order. `feasible_sample_sizes` exposes their sizes, and `best` is the chosen
optimizer result. The inner results record `objective="expected_sample_size"`,
because they minimize expected enrollment within a fixed `N`; the outer result
records the requested across-size objective.

If all sizes are infeasible, `BOP2InfeasibleError` is raised. This is a `ValueError`
subclass specifically for an empty feasible grid. Invalid inputs and numerical
failures propagate normally and are not treated as infeasible sample sizes.

## Toxicity and informative priors

For toxicity, use `endpoint="toxicity"`, with a high unacceptable null toxicity
rate and a lower alternative toxicity rate. Success means concluding safety.
The same expected-enrollment and minimax criteria apply.

All optimization uses the null-centered prior. An optional `analysis_prior`
recomputes each selected parameter pair's analysis boundaries and OC without
changing calibration or selection of the best sample size. Error and power
constraints apply to the **calibration** results; an informative analysis prior
can invalidate them. Inspect `analysis_success_probability` separately.

## Optimize enrollment at a fixed maximum size

```python
from mdanderson_stats import optimize_bop2_binary

fit = optimize_bop2_binary(
    40,
    0.2,
    0.4,
    objective="expected_sample_size",
    minimum_power=0.8,
)
print(fit.calibration_oc.expected_sample_size)
print(fit.calibration_success_probability)
```

The existing fixed-size optimizer defaults to `objective="power"`, preserving
its previous behavior. `minimum_power` is optional for that objective; if supplied,
it filters out candidates below the requested power. Expected-sample-size
optimization requires `minimum_power` and `error_control="strict"`. It rejects
closest-to-nominal error relaxation. Power must lie in `(0,1]`; infeasible
constraints raise `BOP2InfeasibleError`.

Validation independently enumerates all combinations of sample size, scale and
exponent for efficacy and toxicity, checking both objectives and every retained
per-size result. It also checks that fixed-size expected-enrollment optimization
differs from power maximization, informative-prior separation, and the distinction
between an infeasible search and invalid inputs.

The first example searched 41 sample sizes in about 1.5 seconds locally. It selected
`N=25`, with approximately 0.09659 null error, 0.80944 power and 16.19 expected
subjects under the null. Performance and the selected design depend on the grids
and assessment schedule.
