# STPLAN inverse planning

`stplan_solve` finds a design parameter from a target power using any of the
[25 STPLAN forward methods](stplan.md). It supports sample sizes, effects,
probabilities, standard deviations, significance levels, accrual, follow-up,
and the other numerical arguments exposed by those methods. It solves one study
design at a time; the forward APIs retain their array broadcasting support.

```python
from mdanderson_stats import stplan_solve

plan = stplan_solve(
    "stplan_normal_one_sample_power",
    compute="sample_size",
    target_power=0.8,
    bounds=(2, 1000),
    parameters={"difference": 0.5, "sd": 1},
)
print(plan.value)           # approximately 26.137504
print(plan.achieved_power)  # approximately 0.8
```

The method name is the full public forward-function name. `parameters` contains
its fixed inputs by name, with the computed argument omitted. Optional arguments
retain their forward-method defaults. `STPLAN_METHODS` describes each method's
computable arguments, intrinsic integer arguments, tied sizes, and group vectors.
Flags such as `sides`, `model_arm`, and `continued_followup`, and the numerical
`tail_tolerance` setting, are fixed inputs rather than quantities to solve.

The result contains the solved value, achieved and target power, completed
forward inputs, search bounds and evaluation count. The inputs are protected
against mutation so the reported design remains reproducible. Tied-size results
contain one solved value for each tied argument.

## Continuous parameters and branches

Continuous planning solves an equation within the explicit `bounds`. The bounds
must be valid forward inputs and enclose a crossing of the requested power.
Choose the intended effect direction: a detectable difference can have positive
and negative solutions, and exposure frequency or survival parameters can have
more than one solution. One bracketed solution is not a claim to find all roots
or the globally smallest feasible design.

After solving, the function evaluates power again and checks it against
`target_power`, with default absolute `power_tolerance=1e-8`. A numerical root
finder can appear to converge at a jump without attaining the requested power;
that case raises an error. Exact-test power is particularly prone to such jumps
when a null probability, significance level, or event exposure changes its
rejection boundary. Choose a different attainable target or use an appropriate
integer attainment search when equality is not the design question.

Positive brackets are searched on a logarithmic scale, preserving relative
precision when time or measurement units change. The forward methods' input,
tail-accuracy, and resource limits continue to apply.

Each solve allows at most 20,000 forward evaluations and five million cumulative
mixture terms or group entries. `max_evaluations` can lower the evaluation limit.
Integer searches reject a candidate range larger than that limit before starting.
These bounds keep broad searches from consuming unbounded work.

```python
# Significance needed for 80% power with a fixed design.
significance = stplan_solve(
    "stplan_normal_one_sample_power",
    compute="alpha",
    target_power=0.8,
    bounds=(0.0001, 0.49),
    parameters={"difference": 0.5, "sd": 1, "sample_size": 20},
)
# Approximately 0.09002915.

# Common fractional planning size for two equal groups.
balanced = stplan_solve(
    "stplan_normal_two_sample_power",
    compute=("n1", "n2"),
    target_power=0.8,
    bounds=(2, 1000),
    parameters={"difference": 0.5, "sd": 1},
)
# Approximately 50.150783 per group.
```

The solver inverts the package's forward power formula. Some native inverse
formulas use an additional approximation. For example, the original one-sample
normal routine adds two central-t quantiles to find a detectable effect or SD.
With ten observations, SD one, one-sided alpha 0.05, and target power 0.8,
it returns difference `0.8590380`, whose forward power is about `0.8049910`.
Solving the noncentral-t power gives difference `0.8528375` and power 0.8.
The Python result intentionally follows the latter calculation.

## Integer design questions

Exact-binomial sample sizes, matched case-control pair counts, and retention
counts automatically use integer search. Other parameters, including approximate
planning sample sizes, can request `integer=True`. The search checks candidates
exhaustively within integer-valued bounds, because exact-test power can fall at
some successive sample sizes.

The usual default `integer_goal="smallest"` finds the smallest candidate with
computed power at least the target. The `minimum_remaining` retention question
defaults to `"largest"`. `integer_goal="largest"` searches in descending order and
finds the largest feasible candidate. `previous_value` and `previous_power`
describe the candidate immediately preceding the chosen value **in search
order**. For a largest-value search this is the next larger candidate; either
field is absent if the first candidate already meets the target. Integer
attainment may exceed target power and is not reported as an exact equation root.

```python
exact = stplan_solve(
    "stplan_exact_binomial_power",
    compute="sample_size",
    target_power=0.8,
    bounds=(1, 100),
    parameters={"null_probability": 0.2, "alternative_probability": 0.4},
)
# n=35, power about 0.8048255; n=34 gives about 0.7669190.

remaining = stplan_solve(
    "stplan_retention_probability",
    compute="minimum_remaining",
    target_power=0.9,
    bounds=(0, 100),
    parameters={"initial_size": 100, "dropout_rate": 0.05, "duration": 3},
    integer_goal="largest",
)
# 81 remaining: probability about 0.9283302; requiring 82 gives 0.8848542.
```

## K-group binomial designs

For `stplan_binomial_k_sample_power`, fixed group inputs are one-dimensional
vectors. To solve one group's probability or size, pass `index` (zero-based) and
include the full vector in `parameters`; the selected element is replaced.
The other elements remain fixed.

To solve total sample size, use `compute="sample_sizes"` without an index and
omit that vector. `allocation_weights` sets the relative group allocation;
the default is equal allocation. The solved value is total sample size, with
completed group sizes in `result.inputs["sample_sizes"]`.

```python
multi_group = stplan_solve(
    "stplan_binomial_k_sample_power",
    compute="sample_sizes",
    target_power=0.8,
    bounds=(1, 2000),
    parameters={"probabilities": [0.1, 0.2, 0.3]},
    allocation_weights=[1, 2, 1],
)
# Total approximately 308.310044, allocated 1:2:1.
```

Whole-vector proportional allocation retains fractional planning counts, as in
the native model, and rejects `integer=True`. Integer enrollment allocation and
its achieved power must be chosen separately. A single indexed group size can
use integer search.

## Validation and remaining workflows

`tools/reference_stplan_planning.R` supplies 33 independent inverse solutions,
covering all 25 forward methods, both kinds of integer retention questions,
equal group sizes, and total/indexed K-group planning. It uses R probability
functions, finite probability sums, numerical integration, root finding, and
finite integer enumeration. The Python checks verify the solved values, achieved
power, preceding failed integer candidates, and usability of the completed
forward inputs.

`tools/reference_stplan_inverse.f90` probes separately acquired original STPLAN
routines. Five native numerical roots provide further comparisons, and two
normal effect/SD cases record the native approximation discussed above. The
source's original tolerances explain small differences in numerical roots.
Original source is not redistributed. See [provenance](stplan-sources.json).

Native automatic bound/branch selection, discrete significance planning by
critical-region selection, integer allocation of proportional K-group totals,
joint accrual-time/control-allocation optimization for historical controls,
the inactive matched-pairs procedure, and session/report workflows remain open.
Alternative survival-curve inputs are available through the
[survival input converters](stplan-survival-inputs.md).
