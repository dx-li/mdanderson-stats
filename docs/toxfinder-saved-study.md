# ToxFinder saved-study workflow

`elicit_toxfinder_study_prior` connects the physician inputs in an original TMML
study to the recovered stochastic prior calibrator. `simulate_toxfinder_study`
then executes that saved design and scenario with explicit Python inference
settings. Its report captures the prior, original TMML, sampler settings,
per-trial diagnostics and, when supplied, calibration assessment.

```python
import json
from pathlib import Path
from mdanderson_stats import (
    read_toxfinder_tmml,
    elicit_toxfinder_study_prior,
    simulate_toxfinder_study,
)

study = read_toxfinder_tmml(Path("init.tmml").read_text())
elicitation = elicit_toxfinder_study_prior(study, seed=2026)
for agent in (elicitation.agent1, elicitation.agent2):
    print(agent.assessed_tail_probability, agent.assessed_probability_mcse)
    print(agent.assessed_target_residual, agent.assessed_cutoff_met)

Path("saved-design.tmml").write_text(study.to_tmml())
Path("saved-prior.json").write_text(
    json.dumps(
        {
            "mean": elicitation.prior.mean.tolist(),
            "variance": elicitation.prior.variance.tolist(),
            "calibration_seed": 2026,
            "design_tmml": study.to_tmml(),
        },
        indent=2,
    )
)

# Small execution example, not an assessment of Monte Carlo adequacy.
run = simulate_toxfinder_study(
    study,
    prior=elicitation,
    trials=2,
    seed=2027,
    draws=16,
    warmup=8,
    chains=2,
)
Path("toxfinder-study-report.html").write_text(run.to_html())
```

The report's final posterior uses **all final outcomes by default**. To reproduce
the original pre-final-cohort posterior, explicitly pass `final_policy="legacy"`.
Both choices are named in the report. Native GUI columns incorrectly labeled SD
are replaced by correctly labeled sample SD and MCSE; native constant/single-run
sentinels are not copied. The stage-one invalid predecessor access retains its
explicit safe lowest-dose correction.

To reuse a saved prior without recalibrating:

```python
from mdanderson_stats import ToxFinderPrior

saved = json.loads(Path("saved-prior.json").read_text())
assert saved["design_tmml"] == study.to_tmml()
prior = ToxFinderPrior(saved["mean"], saved["variance"])
```

A separately validated prior can also be supplied directly. Calibration convergence
alone is not sufficient: inspect independent tail-probability assessments and their
MCSE. Reports produced from a bare prior cannot reconstruct lost calibration
assessments; retain the original calibration report and saved inputs.

## Original input mapping

| TMML key | Python meaning |
| --- | --- |
| `pr-negtox`, `tl-d_star`, `pr-hightox`, `pr-toxic` | Two physical-dose columns; each drug supplies negligible, reference, high-toxicity and above-target doses. |
| `toxic-minprob`, `pi_star`, `toxic-maxprob` | Negligible probability, target and prohibitively high probability, in that increasing order. |
| `pr-alpha3`, `pr-beta3` | Interaction means, in package alpha3/beta3 order. |
| `pr-alpha3-var`, `pr-beta3-var` | **Stored variances**, passed literally. The GUI displays their square roots despite the saved comments calling them standard deviations. |
| `mcmc-trans`, `mcmc-maxiter`, `mcmc-reeval` | Retained native sampler metadata. Python `warmup`, `draws`, `chains` are explicit; no inferred thinning or convergence equivalence. |
| `rv-seed1`, `rv-seed2`, `append` | Retained native RNG metadata. Explicit Python seeds select independent streams. |

Original Java conversion methods verify the variance interpretation, including
stored3 -> displayed1.73205 -> saved3 within floating-point precision. Missing
physician inputs fail rather than inventing a fitted prior. Derived study fields
are rebuilt from saved settings before execution, so an inconsistent manually
constructed study cannot override the TMML inputs.

Observed patient recommendation and native CSV import use the same calibrated
prior through `recommend_toxfinder_native`; see the
[patient workflow](toxfinder-native-workflow.md). Unset outcomes remain excluded
from the likelihood while counting toward enrollment capacity. Exact native RNG,
GUI layout and original cohort-log bytes are not asserted. Source compatibility,
independent mathematical correctness and clinical validity remain separate claims.

## Patient conduct and editable CSV

```python
from mdanderson_stats import (
    read_toxfinder_subject_csv,
    write_toxfinder_subject_csv,
    recommend_toxfinder_study,
)

records = read_toxfinder_subject_csv(
    Path("subjects.csv").read_text(), reference_doses=study.reference_doses
)
decision = recommend_toxfinder_study(
    study, records, prior=prior, seed=2028, draws=500, warmup=250, chains=2
)
Path("patient-recommendation.html").write_text(decision.to_html())
Path("saved-subjects.csv").write_text(
    write_toxfinder_subject_csv(records, reference_doses=study.reference_doses)
)
```

Edit immutable records with `dataclasses.replace`, then save the updated tuple.
Native outcomes are 0 no toxicity, 1 toxicity and 2 unset. Dates/times remain
identifiers rather than follow-up times. CSV output restores physical doses;
comments, quoting and unique IDs are validated by the strict reader. Recommendation
reports retain patient rows, prior moments, saved settings and sampler diagnostics.
A capacity-reached result supplies no new treatment dose. The Python APIs return
values and strings; file writing occurs only in the explicit caller code above.

## Computational budgets and allocation diagnostics

The default sampler remains500 retained draws,250 warmup and two chains. Work is
counted as posterior-objective evaluations multiplied by history rows, including
slice stepping and rejected proposals. The20-million default total is a bounded
interactive budget, **not sufficient for every saved design**. Preflight now
rejects a provably insufficient budget before sampling and reports its lower bound.
For stage sizes20/40 and cohort size2, final refit analyzes history sizes
0,2,...,18;20,24,...,56;60. With six nonfixed coefficients this requires at least
14,311,062 counted evaluations per trial,28,622,124 for two. Actual work is larger.
The prior-only fit counts its chain initializations; it draws directly from the
prior rather than doing slice updates.

Opt into a larger bounded budget explicitly, without lowering sampler quality:

```python
run = simulate_toxfinder_study(
    study,
    prior=prior,
    trials=2,
    seed=2027,
    max_sampler_evaluations=200_000_000,
)
```

The total ceiling remains200 million, with at most20 million per individual fit.
Actual work stops at the specified limit. Retained final arrays have a separate
2-million-value cap. Fifty60-patient trials at the default draws/chains would
retain6.35 million values and are rejected before sampling. Run smaller batches
with distinct seeds, save each batch separately, and combine trial-level summaries
with appropriate weighting/MCSE. The API does not silently thin draws or retain
unbounded arrays; different batch boundaries do not promise identical random tapes.

Allocation results retain `map_fit`, and each simulated cohort retains the MAP
used for its allocation (shared by the two stage-two cohorts that share a posterior).
Stage one and final CKP contour selection do not use this MAP and store `None`.
Simulation, individual-trial and patient reports display both native simplex
passes' convergence, iteration counts and objective evaluations. A false flag
is visible; compatibility behavior retains the original optimizer's result,
without silently replacing it with another optimizer.

The independent R posterior reference covers a scalar alpha1 slice. It does not
validate joint six-dimensional mixing. `tools/check_toxfinder_joint_posterior.py`
adds an independent randomized importance-integration comparison for all six log
parameters and four predictive risks on a specified moderate-data example. Its
four-chain2000-draw/1000-warmup check is supporting numerical evidence for that
case, not a convergence guarantee for every prior, history or clinical use.

A recorded two-trial60-patient execution at500/250/two-chain settings used
28,573,716 and28,666,554 counted evaluations (57,240,270 total), completing in
582.84 seconds in the recorded environment with an explicit200-million cap.
This is a representative cost measurement, not a guaranteed upper bound for
other priors, histories, seeds or machines. See the saved
[budget execution evidence](../research/toxfinder-default-budget-check-2026-10-12.json)
and [joint-posterior check](../research/toxfinder-joint-posterior-check-2026-10-12.json).
