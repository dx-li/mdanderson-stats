# BOP2 joint efficacy/toxicity monitoring

Catalog **112** now includes joint efficacy/toxicity monitoring, exact correlated
operating characteristics, and finite-grid power calibration with three null
error constraints. Binary and paired-efficacy methods are documented
[here](bop2-binary.md) and [here](bop2-paired.md). Time-to-event endpoints,
sample-size optimization for joint endpoints and integrated reports remain pending.

The current app's `EffToxT1e1Help-YY.pdf` extends the original 2017 BOP2 design
with the global and partial null hypotheses developed in
[Chen et al., BOP2-TE](https://arxiv.org/abs/2408.05816).
The implementation follows that paper's sections 2.1–2.4 and permits the distinct
efficacy/toxicity assessment schedules illustrated in the app's
[interim guide](https://biostatistics.mdanderson.org/shinyapps/BOP2/Interims2Help.pdf).
Archived guides and the paper have [source provenance](bop2-sources.json).

## Specified design

```python
from mdanderson_stats import bop2_efftox_design

design = bop2_efftox_design(
    30,
    null_rates=[0.3, 0.4],
    cutoff_scales=[0.85, 0.85],
    gamma=0.6,
    efficacy_looks=[15, 30],
    toxicity_looks=[6, 12, 18, 24, 30],
)
print(design.looks)  # union of assessment schedules
print(design.futility_max[:, 0])  # maximum efficacy count triggering futility
print(design.toxicity_min)  # minimum toxicity count triggering an unsafe conclusion
print(design.monitor([1, 1, 1, 3]).decision)
```

Both `null_rates` and `cutoff_scales` are ordered **efficacy, toxicity**. Category
counts, prior shapes, and true category probabilities are ordered:

1. Efficacious and toxic (11).
2. Efficacious and not toxic (10).
3. Not efficacious and toxic (01).
4. Neither (00).

With a Dirichlet prior and fully observed category counts, marginal efficacy and
toxicity posteriors are beta distributions formed by summing category shapes.
At a scheduled efficacy look, require

$$
\Pr(p_E>p_{E0}\mid D_n)>\lambda_E(n/N)^\gamma.
$$

At a scheduled toxicity look, require

$$
\Pr(p_T\le p_{T0}\mid D_n)>\lambda_T(n/N)^{\gamma/3}.
$$

Stop when **either** assessed criterion fails. Both criteria must pass at the
final analysis. There is no early success stopping. Each schedule must end at
`max_subjects`; leaving either schedule unspecified uses looks beginning at 10,
every five subjects thereafter, plus the final analysis. Each endpoint is only
assessed on its own schedule. Counts include fully evaluated efficacy and toxicity
for every enrolled subject; pending endpoints and calendar-time accrual are not
modeled here.

`gamma` lies in `[0,1]` and both cutoff scales lie strictly in `(0,1)`.
`toxicity_exponent_factor` defaults to `1/3`, implementing the paper's stricter
early toxicity cutoff, and may be set within `[0,1]`. Both beta tails are evaluated
directly, avoiding subtraction of probabilities close to one. Numerically
underflowed cutoffs raise an explicit error.

`equality_continues=False` follows BOP2-TE's strict go inequality; equality stops.
Set it to `True` for the original BOP2 convention in which equality continues.
For a specified original-style common-cutoff design, also use equal scales and
`toxicity_exponent_factor=1`. These distinctions are explicit; exact current app
optimizer parity is not claimed.

The default prior has effective sample size one, centered on the specified null
cell probabilities. `null_joint_rate` defaults to `pE0*pT0` (independence); a
supplied joint rate must satisfy the joint probability bounds. Optional `prior`
is a four-element vector of positive shapes. The app's `ETprior.pdf` lists the
correct marginal equations but reverses the 11 and 00 cells in its displayed
default-prior vector. This implementation consistently uses the paper's category
order and correct null-centered cell probabilities:
`[p11, pE0-p11, pT0-p11, 1-pE0-pT0+p11]`.

The returned `BOP2PairedDesign` exposes `monitor`, `monitor_outcomes`, and
`operating_characteristics` as in [paired efficacy](bop2-paired.md), with these
joint-design interpretations:

- `futility_max[:,0]`: efficacy boundary, or `-1` where efficacy is not assessed.
  Its second column is always `-1`, since toxicity is not a low-event futility rule.
- `toxicity_min`: unsafe toxicity boundary, or `n+1` where toxicity is not assessed.
- `marginal_success_probability`: posterior efficacy and posterior safety probabilities.
- Early decisions: `stop_futility`, `stop_toxicity`, or `stop_futility_toxicity`.
- Final decisions: `final_positive` means efficacious and safe; `final_negative` means
  at least one criterion failed. A prior stop remains in force in full-path replay.
- OC `stop_probability`: combined futility/toxicity stopping mass, without double
  counting subjects who fail both criteria. It includes failure at the final look.
  `success_probability` is the probability of reaching a positive final conclusion.

The exact forward recursion preserves the joint category distribution at every
look, including correlation. It supports up to 200 subjects and batched scenarios,
with the same memory guard as paired efficacy. Sample-size distributions,
expectations, standard deviations and early-stop probabilities are returned.

## Three-null calibration

```python
from mdanderson_stats import optimize_bop2_efftox

fit = optimize_bop2_efftox(
    36,
    null_rates=[0.3, 0.4],
    alternative_rates=[0.6, 0.2],
    efficacy_looks=[18, 36],
    toxicity_looks=[6, 12, 18, 24, 30, 36],
    type1_error=[0.05, 0.10, 0.10],
)
print(fit.cutoff_scales, fit.gamma)
print(fit.calibration_oc.success_probability)  # three errors, then power
print(fit.calibration_oc.expected_sample_size)
print(fit.distinct_boundaries, fit.parameter_triples)
```

The optimizer requires higher alternative efficacy and lower alternative toxicity.
It evaluates four scenarios in this fixed order:

| Scenario | Efficacy rate | Toxicity rate | Meaning |
| --- | --- | --- | --- |
| H00 | Null | Null | Futile and toxic |
| H01 | Null | Alternative | Futile but safe |
| H10 | Alternative | Null | Efficacious but toxic |
| H11 | Alternative | Alternative | Efficacious and safe |

`type1_error` supplies limits for H00, H01 and H10. Limits must lie in `(0,1]`;
setting a partial-null limit to one disables that constraint, as described in the
app guide. Calibration strictly enforces all three limits and maximizes H11 power.
It does not implement the app's optional closest-to-nominal relaxation.

Joint rates default to independence in all four scenarios, following the paper's
calibration recommendation. For correlated scenarios, supply `joint_rates` with
four joint probabilities in the above order. The guarantees concern these specific
joint distributions; no search over all correlations or composite-null points
is performed.

The default grid uses 50 efficacy scales, 50 toxicity scales (`0.50` through `0.99`)
and 21 exponents (`0` through `1` in steps of `0.05`): 52,500 parameter triples.
Each marginal boundary sequence is precomputed, and each distinct joint boundary
pair is evaluated once. Power ties favor lower expected sample size under H00,
then input grid order. Custom grids support at most 100,000 triples. An infeasible
grid raises an error rather than relaxing an error limit. The result is an optimum
over this finite grid, not over all real-valued parameters.

Calibration always uses the proper, null-centered ESS-one prior. Optional
`analysis_prior` recomputes monitoring boundaries and OC at the chosen parameters.
Both versions are returned; informative-prior analysis can exceed the calibrated
error limits. Alternative/OC cell probabilities may be zero, but null cells must
be strictly positive to define the calibration prior.

Validation checks exact rational beta posterior probabilities, both equality
conventions, distinct assessment schedules, every categorical path in a six-subject
trial, and the independent-outcome identity
`error_H00 * power = error_H01 * error_H10`. A small independent grid search
verifies the three-constraint optimum and informative-prior separation. The
36-subject example above evaluated 52,500 triples as 1,137 distinct boundary pairs
in about 1.1 seconds locally; its three error rates and power were approximately
`[0.00811, 0.07126, 0.09288, 0.81597]`.
