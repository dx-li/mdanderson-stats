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
    np.broadcast_to([.3, .7], (2, 2, 2)),
    np.broadcast_to([.8, .2], (2, 2, 2)),
    association=0.0,
)
mean = np.array([-1, .3, .2, -.1, 0, -.4, -1.2, -.05, .15, 0, np.log(.4), 0])
sd = np.zeros(12)
sd[0] = .3
criteria = U2OETCriteria(
    efficacy_level=1, toxicity_level=1, min_efficacy=0, max_toxicity=1,
)
def run(seed):
    return simulate_u2oet_gao_trial(
        [1, 3], [2, 5], scenario, [[20, 0], [100, 50]],
        prior_mean=mean, prior_sd=sd, initial=(0, 0), criteria=criteria,
        max_patients=4, cohort_size=2,
        efficacy_window=(2, 2), toxicity_window=(.5, .5),
        mean_interarrival=1, draws=16, warmup=8, chains=2,
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

Four independent R calendar ledgers check patient assignments, pending-outcome
snapshots, utility calculations, stopping and final selection. The
[audit](../research/u2oet-gao-trial-audit.md) records those comparisons and the
focused checks; fixed-coordinate ledgers do not establish MCMC precision.
