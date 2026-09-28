# BARPO trial conduct and operating characteristics

The completed-outcome trial workflow connects [BARPO posterior monitoring and
allocation](barpo.md) to enrollment, scheduled decisions and repeated simulation.
All responses are available before the next allocation update. It does not model
delayed ascertainment.

## Enrollment and decisions

`run_barpo_trial` takes true arm response rates, Beta priors, an enrollment
maximum, an explicit monitoring schedule, and equal-randomization burn-in.
Burn-in uses balanced permuted blocks. Adaptive allocation probabilities are
held fixed within each cohort, with cohorts shortened at monitoring looks,
the end of burn-in and the enrollment maximum.

The four allocation methods are BARCP, BARN2N, BARMTV and DBCD. DBCD requires
an explicit target and positive assigned counts in every arm before adaptive
allocation. When adaptive DBCD enrollment is planned, burn-in must cover a
complete balanced block to guarantee those positive counts. Allocation floors
are removed for closed arms; remaining floors
retain their specified values. Posterior best-arm probabilities retain the
original set of competitors, matching the standalone allocation API.

Early monitoring starts only after both burn-in and the minimum enrollment
requirement are met. Under `stopping_policy="arm"`, a futility or efficacy
declaration closes that experimental arm and enrollment continues in eligible
arms. Under `"trial"`, any new early declaration ends enrollment. Contradictory
futility and efficacy declarations for the same arm raise an error.

At maximum enrollment, only the final efficacy criterion is applied to arms
that remain eligible. Earlier declarations are retained separately. With a
control arm, the first arm is the control and is never itself declared futile or
efficacious. Enrollment ends once every experimental arm has closed.

These scheduling and stopping choices are explicit Python conventions.
The official guide specifies the monitoring and allocation formulas, but
does not establish native block permutations, simultaneous-decision priorities
or partial-cohort behavior. The [trial audit](../research/barpo-trial-audit.md)
records the independent mathematical references and remaining source limits.

```python
from mdanderson_stats import run_barpo_trial

settings = dict(
    prior=[[1, 1], [1, 1]], max_n=12,
    burn_in=4, er_block_size=4, cohort_size=2,
    looks=[4, 8, 12], min_n=4,
)
trial = run_barpo_trial(
    [0, 1], **settings, method="barcp", tau=.7,
    theta_fut=.4, pfut=.7, theta_eff=.6, peff=.9,
    theta_final=.5, pfinal=.9, stopping_policy="arm",
    assignment_uniforms=[.2, .9, .7, .3, .8, .1, .55, .95, .15, .7, .4, .85],
    outcome_uniforms=[.05, .82, .25, .1, .45, .93, .3, .6, .12, .77, .9, .22],
)
assert trial.enrolled == 8
assert trial.assigned.tolist() == [2, 6]
assert trial.early_futility.tolist() == [True, False]
assert trial.early_efficacy.tolist() == [False, True]
```

Assignments use one-based arm identifiers. The example's first arm closes for
futility after four patients; the remaining arm closes for efficacy at the next
scheduled look. Early stopping leaves `final_assessed=False`.

## Reproducibility and simulation

Assignment and outcome random streams are separate. Explicit uniform tapes
allow a trial path to be replayed: assignment uniforms select arms, and outcome
uniforms determine response by comparison with that arm's true response rate.
Burn-in permutation also uses the assignment tape.

`simulate_barpo` repeats trials sequentially and aggregates allocation,
enrollment and early/final declaration summaries, including Monte Carlo
standard errors. It does not retain every trial's posterior history. Mean
allocation curves average across all simulated trials, with zero contribution
after a trial stops.

Any-arm efficacy probability is distinct from false-declaration probability.
A caller-supplied `null_arms` mask identifies which arms count as false
discoveries for familywise summaries; the simulator does not infer null status
from the supplied response rates.

```python
from mdanderson_stats import simulate_barpo

simulation = simulate_barpo(
    [.2, .65], **settings, trials=20, early_monitoring=False,
    theta_final=.5, pfinal=.9, null_arms=[True, False], rng=338,
)
assert simulation.mean_enrolled == 12
print(simulation.mean_patients_by_arm)
print(simulation.final_efficacy_probability)
```

Twenty trials keep this example quick; they give imprecise operating-characteristic
estimates. Choose an appropriate replication count and examine its Monte Carlo
uncertainty when comparing designs.

`cumulative_efficacy_probability` combines early and final declarations for
each arm; `any_cumulative_efficacy_probability` gives the chance of any such
declaration in a trial. With `null_arms` supplied,
`false_cumulative_efficacy_probability` is the probability of at least one
false declaration at either stage. The corresponding `*_mcse` fields report
Monte Carlo standard errors. Early-only and final-only rates remain available
separately; adding those trial-level rates would double-count trials with
declarations at both stages.

`assignment_probability` records the conditional probability used for each
actual assignment, including remaining-block probabilities during burn-in.
The simulation's `allocation_probability_by_enrollment` averages those vectors
and includes zero contribution after stopping. This differs from the
unconditional equal-randomization marginal, which is `1 / number_of_arms`.

The workflow supports up to 10 arms, 2,000 patients and 10,000 simulation trials.
The simulation's `max_work` ceiling bounds `trials * max_n * arms**2`; supplied
replay tapes are limited to two million entries each before conversion.
Posterior calculations and trials run sequentially, and the aggregate result
retains compact per-trial counts and declarations rather than all look histories.

The workflow adds trial conduct and operating-characteristic calculations.
Native DBCD target construction, generated reports and direct app parity
remain unverified.
