# U2OET GAO trial simulation

The GAO calendar driver connects the [explicit-prior GAO fitter](u2oet-gao-fit.md)
to U2OET patient arrival, outcome observation, dose allocation and final selection.
It uses raw dose units and the same named Gaussian prior coordinates as the
fitter, including shared log-kappa and association Fisher-z. These are explicit
Python priors; native GAO prior-file mapping remains unverified.

## Observed data and allocation

The scenario specifies a joint ordinal efficacy/toxicity probability table at
every dose pair. Arrival timing and endpoint observation windows are explicit.
At an arrival, only outcomes already observed enter the posterior: toxicity
alone contributes its marginal likelihood; a patient whose toxicity is still
pending contributes no outcome likelihood, even if efficacy is known. All
assignments count toward enrollment and dose-allocation history.

The first assignment uses the configured pair. Subsequent assignments use the
existing U2OET cohort, patient-surplus, utility, acceptability and no-skipping
rules. Complete final follow-up precedes final selection. The caller chooses
whether final selection considers all acceptable pairs or only tried acceptable
pairs. An early stop remains a stop after final follow-up.

Separate data and posterior random streams keep sampler workload from changing
the generated data stream. Results retain patient records, outcome times,
decision summaries, seeds and actual posterior work. Stochastic replicates can
be summarized by `summarize_u2oet_trials` using matching design settings.

```python
import numpy as np
from mdanderson_stats import (
    U2OETCriteria,
    simulate_u2oet_gao_trial,
    summarize_u2oet_trials,
    u2oet_scenario,
)

scenario = u2oet_scenario(
    np.broadcast_to([0.3, 0.7], (2, 2, 2)),
    np.broadcast_to([0.8, 0.2], (2, 2, 2)),
    association=0.0,
)
mean = np.array([-1, 0.3, 0.2, -0.1, 0, -0.4, -1.2, -0.05, 0.15, 0, np.log(0.4), 0])
sd = np.zeros(12)
sd[0] = 0.3
criteria = U2OETCriteria(
    efficacy_level=1,
    toxicity_level=1,
    min_efficacy=0,
    max_toxicity=1,
)


def run(seed):
    return simulate_u2oet_gao_trial(
        [1, 3],
        [2, 5],
        scenario,
        [[20, 0], [100, 50]],
        prior_mean=mean,
        prior_sd=sd,
        initial=(0, 0),
        criteria=criteria,
        max_patients=4,
        cohort_size=2,
        efficacy_window=(2, 2),
        toxicity_window=(0.5, 0.5),
        mean_interarrival=1,
        draws=16,
        warmup=8,
        chains=2,
        rng=np.random.default_rng(seed),
    )


trial = run(7711)
assert len(trial.patients.records) == 4
summary = summarize_u2oet_trials([trial, run(7712)])
print(summary.mean_enrollment, summary.selection_probability)
```

The permissive screens, two replicates and short chains keep this interface
example small; they do not demonstrate a calibrated design or reliable Monte
Carlo precision. Supply suitable criteria, priors and simulation sizes for an
actual study.

Optional `arrival_times` provides a fixed schedule starting at zero.
Optional `data_uniforms` has one row per planned patient and four columns:
allocation, joint outcome, efficacy delay and toxicity delay. Supplied uniform
tapes are marked for replay and rejected by the independent-trial summarizer.
The original input random-generator state reproduces the full simulation;
recorded substream seeds identify the data and posterior streams.

## Computation and scope

Posterior fits run sequentially and share cumulative evaluation and work
budgets. Retained-storage checks cover one posterior fit and the trial history;
the simulator does not retain every fit's parameter draws. Sampling diagnostics
remain estimates and short example chains do not establish adequate precision.

This driver implements the stated Python timing and final-selection conventions.
Native prior calibration, native file/report workflows and reproduction of
published trial operating characteristics remain separate work.

## Adaptive GAO posterior precision

The same calendar can opt into the U2OET guide's four-corner utility precision
target through `U2OETAdaptiveSettings`. Settings are explicit and shared with
the PDS/CMI calendar API; their target range and retained-draw constraints
come from the guide, while chunking and continuation are Python conventions.
The GAO-specific fit still requires the caller's explicit prior coordinates.

```python
from mdanderson_stats import U2OETAdaptiveSettings

adaptive_trial = simulate_u2oet_gao_trial(
    [1, 3],
    [2, 5],
    scenario,
    [[20, 0], [100, 50]],
    prior_mean=mean,
    prior_sd=sd,
    initial=(0, 0),
    criteria=criteria,
    max_patients=4,
    cohort_size=2,
    efficacy_window=(2, 2),
    toxicity_window=(0.5, 0.5),
    mean_interarrival=1,
    draws=16,  # retained only for fixed-mode compatibility
    warmup=8,
    chains=2,
    adaptive_precision=U2OETAdaptiveSettings(
        target_mcse_ratio=0.05,
        initial_draws=512,
        max_draws_per_chain=2048,
        batch_draws=512,
        max_total_work=10_000_000,
    ),
    rng=np.random.default_rng(7721),
)
print(adaptive_trial.final_precision_target_met)
print(adaptive_trial.final_precision_draws_per_chain)
print(adaptive_trial.final_corner_mcse_ratio)
```

For adaptive mode, `initial_draws` must be at least the warmup count. The
existing GAO fitter supports 2–16 chains; warmup also inherits the settings
validator's 10,000-draw limit. Before advancing the caller's RNG, the trial
preflights the minimum fit work over the maximum possible number of fits,
adaptive fit live memory, and the combined posterior/history/tape storage.
The whole-trial work allowance is the smaller of `max_work` and
`adaptive_precision.max_total_work`; evaluation limits continue to use
`max_likelihood_evaluations`. Both budgets are cumulative across changed
count states. Sampler rejections can consume more than minimum work; the
remaining allowance is passed to each analysis and exhaustion raises rather
than resetting the budget.

The existing sufficient-statistic cache remains in force: if complete and
toxicity-only counts are unchanged, the posterior and precision diagnostics
are reused. Each decision records the precision target status, draw count and
four corner ratios; final equivalents are available on the trial result.
An adaptive fit that reaches its cap without meeting the target raises
`ArithmeticError` before the corresponding dose assignment or final selection.
It never silently substitutes an under-precise posterior. Adaptive settings
are stored in `design_json`, so replicate summaries continue to require
matching designs. Fixed mode leaves its JSON and RNG path unchanged. This is
Python adaptive GAO coverage, not native adaptive-mode or prior-default parity.

Four independent R calendar ledgers check patient assignments, pending-outcome
snapshots, utility calculations, stopping and final selection. The
[audit](../research/u2oet-gao-trial-audit.md) records those comparisons and the
focused checks; fixed-coordinate ledgers do not establish MCMC precision.
