# Calibrating two-endpoint TOP designs

`optimize_top_multiendpoint` searches an explicit grid of C and gamma for a
[`TOPMultiEndpointDesign`](top-endpoints.md). It selects the candidate with
the highest estimated power at one supplied alternative, subject to an estimated
success probability at most `type1_error` in **every supplied null scenario**.
The search keeps the design's prior, assessment windows, analysis timing, looks
and suspension convention. Ties prefer smaller worst-null mean enrollment,
then the original candidate order.

## Specify the joint scenarios

Null scenarios form an `(S,4)` matrix; the alternative is one four-element vector.
Each row gives joint probabilities in order `(1,1), (1,0), (0,1), (0,0)`.
Zero-probability truth cells are allowed. Specifying endpoint margins alone is
insufficient because their association changes operating characteristics.

For co-primary efficacy, a null scenario has both efficacy probabilities at or
below their null thresholds; an alternative has at least one above threshold.
For efficacy/toxicity, a null scenario has efficacy at or below its threshold
or toxicity at or above its threshold. The alternative must have efficacy above
and toxicity below its threshold. Null thresholds are the design's prior margins.

Include clinically relevant partial-null scenarios and associations. For example,
with efficacy threshold .2 and toxicity threshold .3, testing only (.2,.3) can
miss larger false-success rates at (.2,.1) or (.5,.3). The
[independent exact reference](../research/top-multiendpoint-calibration-audit.md)
demonstrates this for a 12-patient final-only design: at C=.8 the both-boundary
success probability is .0573, but the latter two nulls give .1827 and .2344.

```python
from mdanderson_stats import TOPMultiEndpointDesign, optimize_top_multiendpoint

design = TOPMultiEndpointDesign(
    12, [.05, .15, .25, .55], .8, .5,
    mode="efficacy_toxicity", windows=[1, 2], looks=[12],
)
calibrated = optimize_top_multiendpoint(
    design,
    null_joint_probabilities=[
        [.05, .15, .25, .55],  # both endpoints at their null boundaries
        [.02, .18, .08, .72],  # efficacy at null, toxicity safe
        [.15, .35, .15, .35],  # efficacy effective, toxicity at null
    ],
    alternative_joint_probabilities=[.05, .45, .05, .45],
    accrual_rate=2,
    cutoff_scales=[.8, .95], gammas=[.5], type1_error=.1,
    trials=200, validation_trials=200, rng=134,
)
print(calibrated.parameter_pairs[calibrated.selected_index])
print(calibrated.validation_probability, calibrated.validation_mcse)
```

This small example illustrates the interface; 200 repetitions leave appreciable
Monte Carlo uncertainty. Its single final look makes gamma irrelevant. Supply
interim `looks` when comparing gamma values or studying early stopping.

## Selection, independent validation and uncertainty

All candidates and scenarios reuse the same underlying arrival, joint-outcome
and event-time random draws. The selected design is then evaluated using an
independent seed. Validation reports its performance without changing selection.
The design's analysis timing remains separate from the optional
`truth_timing_probabilities`; omitted truth timing matches the analysis mixture.
Calendar conventions and conditional timing independence are described in the
[simulation guide](top-endpoints-simulation.md).

Calibration arrays have a candidate axis followed by a scenario axis; scenario
order is the supplied null rows followed by the alternative. Validation arrays
have only the scenario axis. Results include success probabilities, Monte Carlo
standard errors, mean enrollment and duration, feasibility flags, selected
parameters and stage seeds. Decision-probability arrays add an action axis in
`decision_labels` order. Arrays are read-only.

The constraint concerns estimated probabilities on the supplied finite grid.
Neither the grid nor its independent validation establishes strong control over
the entire composite null. Validation can exceed the requested target; report
that result and its uncertainty instead of redrawing until it passes. An
infeasible grid raises `TOPInfeasibleError`.

Work is bounded before simulation: at most 100 candidate pairs, 20 null scenarios,
100–100,000 trials per stage, and two million endpoint-patient cells per stage.
`max_work` additionally limits scanned endpoint-patient cells across all
candidate/scenario runs and both stages, with a default of 50 million and a
hard ceiling of 100 million. Preflight counts one scan per scheduled look;
every actual analysis, including a suspension recheck, consumes the shared
runtime budget. Exhaustion raises an error instead of returning partial search
results. Scenarios run serially and retain summaries rather than full trial
histories. These are resource limits, not measures of statistical precision.
