# ARAND native controller profile

This named profile implements the recovered ARAND 5.2.2 single-endpoint,
single-group candidate-time controller. The existing `ArandControllerPolicy`
workflow remains available. The two profiles intentionally differ.

`arand_native_replay` takes candidate arrival times, assignment uniforms, and
candidate-by-arm observation delays and indicators. A binary indicator is success
or failure. A survival indicator distinguishes an event from right censoring.
Observation is visible at equality with its absolute record time; censored records
count as observed for monitoring but contribute no survival event. All times use
months. Arm indices follow supplied order, are zero-based, and also define ties.

## Runnable study, simulation, saved inputs and report

```python
from mdanderson_stats import (
    ArandNativeArmLimits,
    ArandNativeMonitor,
    ArandNativeRules,
    ArandNativeStudy,
    arand_native_report,
    arand_native_study_json,
    read_arand_native_study,
    simulate_arand_native,
    simulate_arand_native_trial,
)

study = ArandNativeStudy(
    prior=((1, 1), (1, 1)),
    rules=ArandNativeRules(
        arms=(ArandNativeArmLimits(), ArandNativeArmLimits()),
        maximum_enrollment=12,
        lower=0,
        upper=1,
        runin=4,
        final_followup=1,
        stop_when_one_arm=False,
    ),
    monitor=ArandNativeMonitor(mode=0),
    true_parameters=(0.3, 0.6),
    accrual_rate=2,
    binary_delay=1,
    max_candidates=20,
)
saved = arand_native_study_json(study)
assert read_arand_native_study(saved) == study
summary = simulate_arand_native(study, trials=4, seed=2026)
assert sum(summary.selected) + summary.no_selection == 4
replayed = simulate_arand_native_trial(study, seed=summary.trial_seeds[0])
assert len(replayed.records) == 12
html = arand_native_report(summary)
assert "Saved inputs" in html
```

JSON is a versioned **Python study format**, not an import/export of original
ARAND XML. It captures priors, rules, the initial monitor state, scenario,
precision and candidate budget. Unknown or duplicate JSON keys and unsupported
versions fail. The standalone HTML captures those inputs and every replicate seed.
Use `Path(...).write_text(saved)` or `.write_text(html)` to save either string.

For survival, set `family="exponential"`, use positive true means or medians in
`true_parameters`, and choose `parameter="mean"` or `"median"`. Priors contain
inverse-gamma shape/scale pairs in the selected parameter's units. Median scenario
values are divided by log(2) to generate survival times. Set `censor_time` explicitly
to model the latent finite-horizon scenario; otherwise events are uncensored, as
in the exposed desktop default. Binary scenarios use the fixed `binary_delay`.

## Controller contract

- Trial minimum enrollment and duration gate monitoring. Arm/group minima use
  the native **any-arm / either-level** condition, not an all-arm requirement.
- Monitoring modes: 0 continuous; 1 integer time; 2 supplied time targets;
  3 enrollment increments; 4 exponential-batch increments; 5 observed-count
  targets. There are no independent timer-triggered looks. Supply the initial
  thresholds and target sequence explicitly in `ArandNativeMonitor`. Mode 4's
  `targets` are increments, with its separate `batch_index`; the native
  noninteger batch-construction/reset inconsistency is not guessed by a factory.
- Integer-time equality can cause another look. Discrete targets skip strictly
  smaller entries. Mode 4 checks exact enrollment equality. Pending outcomes
  close every arm temporarily; subsequent constraints can stop the trial.
  A nonpending batch reopens every arm before analysis.
- Ranking always includes **all** arms. Allocation eligibility is applied later.
  Replay starts with independently integrated prior probabilities cached before
  its first candidate. `arand_native_decide` also supports an explicit cached
  `ArandNativeEvidence` and analysis callback for observed-history services.
- Duration completion is strictly above the cap and precedes the enrollment cap.
  Both require trial minima. Completion analyzes again at the current candidate
  time plus final follow-up, retaining candidate overshoot. Early stopping does
  not add that follow-up. The candidate after the last enrolled patient is needed
  to trigger completion; tape exhaustion raises instead of returning a completed
  trial. Supplied uniform values must be in `[0,1)`.
- Arm maximum, arm minimum, group maximum, group minimum, then time gates execute
  in order. Minima can reopen permanently closed arms. Zero arm duration disables
  its start/duration gates. Exposed desktop scope has one group, so counts at both
  levels coincide; separate multi-group counts are not supported.
- Whole-trial selection allows open and enrollment-maximum arms; group selection
  allows open arms only. Cutoffs are strict; equal winning probabilities favor
  the later arm. Group and whole-trial selections are retained separately.
- Eligible weights use a power then iterative proportional floors. Forced equal
  run-in rounds its requirement up to an arm multiple and uses all-arm enrollment
  deficits, bypassing the ordinary tuner when more than one arm is eligible.
  Consequently it can allocate to an otherwise closed arm. The original `<=`
  inverse-CDF test is retained: an exactly zero supplied uniform can select a
  zero-weight first arm in the sampled population. Ordinary allocation samples
  only eligible arms; forced equalization samples all arms. These are explicit **legacy compatibility behaviors**,
  not recommended scientific defaults. Invalid/non-normalizable native weights
  fail rather than silently inventing probabilities.

Status values are exposed by `ArandNativeStatus`: open 0, temporary closure 1,
other-arm winner 2, permanent closure 3, enrollment maximum 4, not started 5,
and duration maximum 9. `ArandNativeRules` provides explicit closure actions,
futility and final cutoffs. Native XML missing-attribute defaults are not claimed.
A futility threshold uses the posterior probability the parameter exceeds it,
independently of the direction chosen for best-arm ranking.

## Simulation and reporting boundaries

Simulation uses three independent NumPy streams for arrivals, outcomes and
assignment, and independent replicate seeds. The first arrival is an exponential
gap from zero. No original RNG sequence, wrapper seed mapping, or native
posterior integrator identity is asserted. Each call has bounded candidates;
serial summaries also bound trial count and aggregate candidate work.

Summaries count selection over every completed simulation call, including calls
with no selected arm. Enrollment intervals are type-7 empirical 2.5/97.5
percentiles, **not confidence intervals**. MCSE is sample SD divided by the square
root of replicate count, undefined (`NaN`) with one replicate. Duration is the
final controller time, including completion follow-up and overshoot. These are
explicit Python report conventions; native report aggregation and quantiles are
still an evidence gate. An exhausted replicate raises and produces no summary.

## Evidence and remaining scope

The [source recovery](../research/arand-controller-recovery-2026-10-12.md) pins
archive/library hashes and describes 98 fixtures across 13 original-instruction
families. The new tests cover applicable single-endpoint primitives and all six
integrated controller cases; the remaining fixtures include latent multi-endpoint,
multi-group and dormant predictive behavior outside this public API. Thus
“98 fixtures regenerated” does not mean 98 complete Python workflow comparisons.
An additional deterministic tool compares 900 randomized status/selection cases
against original instructions. Independent exact beta/Gamma probabilities,
mean/median equivalence and Poisson enrollment/overshoot identities supplement
source compatibility checks.

This closes the named supplied-calendar and explicit Python-study workflow, not
the entire application. Original XML/defaults, native report semantics, stochastic
wrapper policy, and the batch-construction/reset boundary remain separately
tracked. Scientific validity, original compatibility and clinical validation are
separate claims; no clinical validation or catalog promotion is made here.
