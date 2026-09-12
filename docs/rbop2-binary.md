# rBOP2 binary efficacy and toxicity

Catalog 150 now supports supplied-cutoff binary monitoring, boundary tables,
and exact operating characteristics. The [source review](rbop2-binary-source.md)
describes the native rules and the calibration work that remains open.

```python
from mdanderson_stats import rbop2_binary_design

# Small illustrative design, not a calibrated clinical recommendation.
design = rbop2_binary_design(
    [[1, 1], [2, 2]],                 # experimental/control sizes at each look
    prior=[[1, 1], [1, 1]],          # beta shapes, experimental then control
    endpoint="efficacy", margin=0,
    lower_cutoffs=[0.25, 0.8], upper_cutoffs=[0.8, 0.8],
)
state = design.monitor(2, 1, sample_size=[2, 2])
print(state.probability, state.superior)  # probability is exactly 4/5
table = design.boundary_table()
oc = design.operating_characteristics([0.3, 0.6], control_rate=0.3)
print(oc.overall_positive, oc.expected_total_sample_size)
```

For efficacy, the posterior probability is `P(p_experimental > p_control + margin)`.
For toxicity, it is `P(p_experimental < p_control - margin)`. A positive margin
requires improvement; negative margins support noninferiority-style comparisons.
Each arm's events and non-events update its independent beta prior. Priors are
explicit shape pairs; prior mean `p` and effective sample size `n0` convert to
`[p*n0, (1-p)*n0]`.

Futility/toxicity stopping uses a strict comparison below the lower cutoff.
Superiority uses an inclusive comparison at or above the upper cutoff. At
interim looks a probability between the cutoffs continues. The final cutoffs
must coincide, making every final state positive or negative. A monitor call
at an unscheduled pair of sample sizes returns `continue`; it does not invent
a new interim analysis. Callers retain previous stopping decisions when
processing an actual trial's observations.

Looks declare positive integer sizes for both arms, increasing strictly in
each arm, with at most 200 patients per arm. This supports unequal planned
allocation while avoiding implicit rounding. All endpoint observations at a
look must be complete. Calendar accrual, delayed outcomes, and random variation
in arm sizes are outside this conditional count model.

Boundary tables list experimental/control event-count pairs causing each
decision at each look. The exact operating-characteristic calculation
propagates independent binomial increments and absorbs early decisions.
It reports per-look stopping probabilities, final-only conclusions,
overall conclusions including early stops, the probability of reaching the
final analysis, and expected enrollment for each arm and overall. Experimental
and control scenario rates broadcast. In the example above, overall positive
probabilities are 0.3318 and 0.6132, and expected total sizes are 3.16 and 2.92.
`sample_size_probability` gives stopping mass at each declared look, including
the final look. The last entries of `stop_futility` and `stop_superiority`
are the final-only negative and positive masses.

`overall_positive` is the achieved type-I error when evaluated at an appropriate
null scenario and power when evaluated at an alternative. Supplying cutoffs
does not guarantee a nominal error rate or optimality. The native app separates
calibration priors from informative analysis priors; this core performs no
automatic calibration and does not imply that separation on the user's behalf.

Numerical posterior calculations reuse stable beta-difference quadrature and
report estimated absolute errors. Integer-shape, zero-margin comparisons near
a cutoff can be resolved by exact rational beta integrals, preserving inclusive
equality. An unresolved numerical comparison raises an error rather than
changing the rule with a tolerance band. Error estimates are not rigorous
floating-point bounds or relative-accuracy guarantees for arbitrarily rare tails.

The implementation bounds posterior table states and operating-characteristic
work, processes scenarios sequentially, and reuses decision tables. It does not
allocate trial-by-patient simulation arrays. Returned arrays are read-only.
The sum of posterior count states over looks is capped at 20,000; a separate
5,000,000-operation matrix-work estimate includes every look and scenario.
Multiple-efficacy and efficacy/toxicity joint rules, native calibration,
allocation rounding, and integrated report export remain open.
