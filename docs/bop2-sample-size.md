# BOP2 categorical sample-size optimization

BOP2 catalog entry **112** now supports expected-sample-size and minimax searches
for binary efficacy, binary toxicity, ordinal/multiple efficacy, and joint
efficacy/toxicity endpoints. These implement the variable
sample-size criteria in section 2.3 of
[Zhou, Lee and Yuan (2017), DOI 10.1002/sim.7338](https://onlinelibrary.wiley.com/doi/10.1002/sim.7338)
([paper mirror](https://eurekamag.com/research/059/454/059454786.pdf)).
The joint search retains the additional BOP2-TE partial-null constraints.
[Survival monitoring](bop2-survival.md) is available; survival calibration
and sample-size searches remain pending.

## Binary search over maximum sample sizes

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

## Ordinal and multiple efficacy

```python
from mdanderson_stats import optimize_bop2_paired_sample_size

result = optimize_bop2_paired_sample_size(
    [20, 25, 30, 35, 40],
    null_rates=[0.15, 0.3],
    alternative_rates=[0.25, 0.5],
    endpoint="ordinal",
    minimum_power=0.8,
    type1_error=0.1,
)
fit = result.best
print(fit.calibration_design.max_subjects)
print(fit.calibration_oc.success_probability)
print(fit.calibration_oc.expected_sample_size)
```

`optimize_bop2_paired_sample_size` applies the same size criteria to the
[ordinal and multiple efficacy model](bop2-paired.md). Both marginal futility
conditions must hold to stop. `endpoint="multiple"` requires `null_joint_rate`
and `alternative_joint_rate`, preserving each scenario's specified correlation.
Null error and expected enrollment refer to that specified joint null distribution;
there is no search over all possible null correlations.

`interim_looks`, `cutoff_scales`, `gammas`, `analysis_prior`, `min_subjects`, and
`cohort_size` have the same roles as in the binary size search. The returned
`BOP2PairedSampleSizeOptimization` exposes `best`, `feasible_designs`,
`searched_sample_sizes`, and `feasible_sample_sizes`. Both `expected_sample_size`
and `minimax` objectives are supported, with the same documented tie breaking.
All comparisons use calibration OC rather than informative-prior analysis OC.

## Joint efficacy and toxicity

```python
from mdanderson_stats import optimize_bop2_efftox_sample_size

result = optimize_bop2_efftox_sample_size(
    [24, 30, 36, 42],
    null_rates=[0.3, 0.4],
    alternative_rates=[0.6, 0.2],
    minimum_power=0.8,
    type1_error=[0.05, 0.1, 0.1],
    efficacy_interim_looks=[18, 36],
    toxicity_interim_looks=[6, 12, 18, 24, 30, 36],
)
fit = result.best
print(fit.calibration_design.max_subjects)
print(fit.calibration_oc.success_probability)  # H00, H01, H10 errors, then H11 power
print(fit.calibration_oc.expected_sample_size)
```

`optimize_bop2_efftox_sample_size` preserves the
[joint design's three error constraints](bop2-efftox.md) at every candidate size.
Power refers to the efficacious-and-safe alternative H11, while expected
enrollment is minimized under the futile-and-toxic global null H00.
This applies the original BOP2 expected-enrollment/minimax criteria while retaining
the additional BOP2-TE partial-null constraints. It is not a claim of native app
sample-size optimizer parity.

The efficacy and toxicity interim vectors are independently truncated below each
candidate `N`, then `N` is appended to each. Empty vectors request final-only
assessment for that endpoint; omitted vectors use the regular default schedule.
`joint_rates` can specify the four scenarios' joint probabilities; independence
remains the default. The existing `efficacy_scales`, `toxicity_scales`, `gammas`,
`toxicity_exponent_factor`, and `equality_continues` options are preserved.
`analysis_prior` remains separate from calibration and sample-size selection.
The returned `BOP2EffToxSampleSizeOptimization` has the same search-result fields
as the binary and paired versions.

The sample-size limits and infeasibility handling described above apply to both
complex-endpoint searches. Each size searches the full specified parameter grid;
all requested sizes are considered even if feasibility is nonmonotone. Per-size
parameter-grid limits are unchanged: 100,000 pairs for paired efficacy and 100,000
triples for joint efficacy/toxicity. Runtime grows with the size grid and the
number of distinct boundaries, so explicitly choose the sizes and grids to search.

Both fixed-size optimizers, `optimize_bop2_paired` and `optimize_bop2_efftox`, also
accept `objective="expected_sample_size"` with a required `minimum_power`.
Their default remains power maximization; an optional power floor can also be
used with that default. Expected-enrollment optimization enforces strict error
control. All results record the objective and power floor used.

Complex-endpoint validation independently enumerates complete size/parameter grids
for ordinal, correlated multiple-efficacy, and correlated efficacy/toxicity trials.
It checks both objectives, the optimum retained at every feasible size, distinct
assessment schedules, all three joint error constraints, and informative-prior
separation. The five-size ordinal and multiple-efficacy examples took about
0.5 seconds each locally. The four-size joint example took about 4.3 seconds,
selecting `N=36` with approximately 0.8083 power and 12.48 expected subjects under
H00 while satisfying all three error limits.
