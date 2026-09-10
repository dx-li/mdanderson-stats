# BOP2 ordinal and multiple efficacy endpoints

Catalog **112** now supports specified-parameter designs, monitoring of fully
observed outcomes, exact operating characteristics, and power-maximizing grid
calibration for two more BOP2 endpoint types. Binary efficacy/toxicity are described
[separately](bop2-binary.md). Joint efficacy/toxicity, time-to-event endpoints,
sample-size optimization and integrated reports remain pending.

The method follows sections 2.1–2.3, examples 2 and 3 of
[Zhou, Lee and Yuan (2017), DOI 10.1002/sim.7338](https://onlinelibrary.wiley.com/doi/10.1002/sim.7338)
([accessible paper mirror](https://eurekamag.com/research/059/454/059454786.pdf)).
The current app's endpoint/prior guides are recorded in [provenance](bop2-sources.json).

## Model and event ordering

| Endpoint | Category order for counts, prior shapes and true probabilities | Monitored rates |
| --- | --- | --- |
| `ordinal` | CR, PR, other | CR; CR+PR |
| `multiple` | Both responses (11), first only (10), second only (01), neither (00) | First; second |

For ordinal efficacy, SD and PD are combined into `other`. The Dirichlet
aggregation property makes this equivalent for these monitoring criteria to
separately modeling SD and PD and adding their prior shapes and counts.
Multiple efficacy permits any valid joint outcome probabilities, including
independent, positively correlated and negatively correlated outcomes.

For category counts `x`, the posterior is `Dirichlet(prior + x)`. Each monitored
marginal has beta shapes equal to the sums of posterior shapes for categories
inside and outside its event. Stop for futility only when **both** marginal
posterior probabilities of exceeding their null rates are strictly below
`cutoff_scale * (n / max_subjects)**gamma`. Equality continues. There is no early
success stopping; at the final look, a trial that has not crossed this joint
futility boundary is positive. This follows the paper's “and” rule, where
improvement in either efficacy endpoint can make treatment promising.
It is not an intersection-union test requiring both endpoints to improve.

The default prior has effective sample size one, centered on null cell probabilities:

- Ordinal: `[pCR, pORR-pCR, 1-pORR]`.
- Multiple: `[p11, p1-p11, p2-p11, 1-p1-p2+p11]`.

`null_joint_rate=p11` is required for multiple efficacy. All prior shapes must be
positive; a null scenario with a zero cell requires an explicitly supplied proper
prior for a specified design. Grid calibration requires a proper null-centered
prior and therefore strictly positive null cells. Null marginal thresholds lie
strictly between zero and one. Alternative and OC probabilities may include zero
or one. Multiple joint rates must respect the usual probability bounds, and
ordinal rates must satisfy `CR <= CR+PR`.

The app's `OEhelp.pdf` incorrectly includes an impossible CR-without-CR/PR cell,
prints four prior shapes for three categories, and gives inconsistent marginal
formulas. `CPEhelp.pdf` omits the joint cell from the two marginal numerators.
The implementation uses the paper's explicit event definitions and Dirichlet
aggregation property, not these inconsistent formulas. App optimizer parity
has not been established.

## Monitoring and exact operating characteristics

```python
from mdanderson_stats import bop2_paired_design

design = bop2_paired_design(
    40,
    null_rates=[0.15, 0.30],
    endpoint="ordinal",
    cutoff_scale=0.91,
    gamma=0.95,
    looks=[10, 15, 20, 25, 30, 35, 40],
)
print(design.futility_max)  # columns: maximum CR and CR+PR counts for joint futility
state = design.monitor([1, 2, 7])  # 10 subjects: one CR, two PR, seven other
print(state.decision, state.marginal_success_probability)
oc = design.operating_characteristics([[0.15, 0.15, 0.70], [0.25, 0.25, 0.50]])
print(oc.success_probability)
print(oc.expected_sample_size, oc.early_stop_probability)
```

`monitor` accepts counts with a final category axis and broadcasts across leading
axes. Results include posterior Dirichlet shapes, both marginal success
probabilities, and the decision. Off-schedule evaluations continue.
`monitor_outcomes` accepts zero-based category codes on a final patient axis
and retains the first stopping decision throughout a full outcome path; posterior
summaries at each position still describe the supplied cumulative observations.

Exact OC uses a two-dimensional forward recursion over both marginal counts.
Each patient increments them according to the joint category probabilities;
this preserves outcome dependence without multiplying marginal trial-success
probabilities. At every scheduled look, probability mass in the joint futility
rectangle is removed. All surviving mass at the final look is success.

`stop_probability` includes futility at **every** look, including the final one.
`early_stop_probability` excludes the final look. `sample_size_probability`
includes success and futility at completion. Outputs also include the expected
sample size and its standard deviation. No trial simulation is required.

The design supports up to 200 subjects. OC broadcasts across leading scenario
axes, with a memory guard of five million paired-count states per batch; split
larger batches. Category probabilities must sum to one within `1e-14` and are
normalized only within that rounding tolerance. Default looks start at 10 and
repeat every five subjects, always including the final analysis.

## Exact finite-grid calibration

```python
from mdanderson_stats import optimize_bop2_paired

fit = optimize_bop2_paired(
    40,
    null_rates=[0.1, 0.2],
    alternative_rates=[0.25, 0.4],
    endpoint="multiple",
    null_joint_rate=0.05,
    alternative_joint_rate=0.15,
    looks=[10, 15, 20, 25, 30, 35, 40],
    type1_error=0.1,
)
print(fit.cutoff_scale, fit.gamma)
print(fit.calibration_design.futility_max)
print(fit.calibration_oc.success_probability)  # point-null type I error, point-alternative power
print(fit.distinct_boundaries, fit.parameter_pairs)
```

The optimizer uses a common scale and exponent for both efficacy margins, defaults
to 1,050 parameter pairs (`0.50,...,0.99` by `0,0.05,...,1`), and evaluates each
unique pair of integer boundary sequences once. It maximizes exact alternative
power subject to the specified null error constraint. Ties favor lower null
expected sample size, then input grid order. The guarantee applies to the
specified **joint null distribution**; it is not a search over all null
correlations or strong family-wise error control.

`error_control="strict"` enforces numerical type I error at or below nominal;
`"closest"` first minimizes distance to nominal and can exceed it. Custom grids
support up to 100,000 parameter pairs. Infeasible strict grids raise an error.
This is a finite-grid optimum, not a continuous optimization guarantee.

Calibration always uses the null-centered prior. Optional `analysis_prior`
recomputes boundaries and OC after selecting parameters, without changing the
calibration. Both designs and OC are returned, in `[null, alternative]` order.
An informative analysis prior can increase the achieved type I error.

Focused validation checks Dirichlet marginal probabilities against exact rational
beta-binomial identities, including equality at a cutoff; enumerates all 729
ordinal and 4,096 multiple-efficacy paths in a six-subject trial; checks correlated
OC, sample-size moments and sticky stopping decisions; and verifies a small-grid
optimum and the separation of informative-prior analysis from calibration.
