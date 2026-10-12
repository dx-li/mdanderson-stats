# ToxFinder native workflow

The named `toxfinder_native_*` APIs implement the recovered ToxFinder 1.1.1
allocation mathematics and controller. The existing `toxfinder_stage1` remains
the separately documented paper helper: native stage-one behavior differs.
Python posterior inference uses the existing, independently checked log-coordinate
slice sampler. The [native stochastic prior calibrator](toxfinder-native-elicitation.md) is also
available. Native MCMC and random-stream identity are not claimed.

## Complete simulation

```python
import numpy as np
from pathlib import Path
from mdanderson_stats import ToxFinderPrior, simulate_toxfinder_native

# Demonstration prior and small study, not a calibrated clinical design.
prior = ToxFinderPrior(np.ones(6), np.full(6, 0.25))
run = simulate_toxfinder_native(
    [[0.1, 0.1], [0.3, 0.2], [0.6, 0.7], [1.0, 1.0]],
    [[0.0, 0.2], [0.3, 0.5]],
    prior=prior,
    target=0.3,
    stage1_patients=2,
    stage2_patients=4,
    cohort_size=1,
    trials=2,
    seed=2026,
    draws=8,
    warmup=0,
)
Path("toxfinder-report.html").write_text(run.to_html(reference_doses=[1200, 600]))
print(run.mean_toxicities, run.mcse_mean_toxicities)
print(run.trials[0].final_fit.parameter_summary.split_rhat)
```

The tiny draw/trial counts only demonstrate execution. Retained final fits expose
parameter split R-hat and batch-means MCSE; assess mixing and simulation precision
with adequately sized runs. The report includes mean/SD dose pairs, estimated
and scenario risks, mean absolute estimation error, posterior SD, toxicity count,
true exposure, and MCSE of the mean toxicity count. SD explicitly uses `n-1`;
original Java execution confirms sample SD for posterior risk. Across-trial
GUI columns labeled “SD” instead return SEM, with sentinel `1e-5` for constant
replicates and `1e10` for at most one replicate. Python labels sample SD and
MCSE separately and does not reproduce those misleading labels/sentinels.

All input dose pairs are standardized by positive per-drug reference doses.
The scenario surface is a matrix indexed `[drug2, drug1]` on equally spaced
`[0,1]` axes, evaluated by the native bilinear rule. Levels strictly increase
in both agents; nonlinear piecewise L1 and endpoint `(1,1)` are supported.
Gamma priors use package order `(alpha1,beta1,alpha2,beta2,alpha3,beta3)`.
The native information workflow requires positive variances and prior means
inside the optimizer's `[0.005,30]` box. Fixed-parameter priors remain available
in the base model, but are not a native six-dimensional information workflow.

Stage sizes must be cohort multiples, with at most 100 patients. Stage one fits
and allocates each cohort. Stage two computes three recommendations, assigns
branch 2 first and branch 1 second using **the same posterior**, then refits for
the next pair. Outcomes and inference use independent seeded Python streams.
`mode="fi_ckp"` averages the FI and cancer-killing-potential choices;
`mode="fi"` uses FI alone. These are the two advertised GUI modes.

The default `final_policy="legacy"` reproduces a consequential source behavior:
final counts include every patient, but the final contour uses the posterior
from before the last cohort/pair. `posterior_subjects` makes that distinction
visible. `final_policy="refit"` explicitly incorporates all final outcomes.

## Saved studies and observed patients

`read_toxfinder_tmml(text)` reads original design and scenario settings, converts
physical doses, and preserves all key/value settings. `study.to_tmml()` supports
a semantic round-trip, including unknown settings, while discarding comments
and layout. Duplicate keys, excessive inputs and undocumented allocation modes
are rejected. `nD-start` is converted from native one-based to Python zero-based.

```python
from mdanderson_stats import read_toxfinder_tmml, simulate_toxfinder_native

study = read_toxfinder_tmml(Path("init.tmml").read_text())
run = simulate_toxfinder_native(
    study.dose_levels,
    study.true_surface,
    prior=prior,  # supply a validated prior; import never invents fitted moments
    target=study.target,
    potency=study.potency,
    start_index=study.start_index,
    stage1_patients=study.stage1_patients,
    stage2_patients=study.stage2_patients,
    cohort_size=study.cohort_size,
    mode=study.mode,
    trials=2,
    seed=2026,
)
```

Native physician inputs can be used with either explicitly named formulation:
`elicit_toxfinder_prior` solves published equations;
`elicit_toxfinder_native_prior` uses the recovered stochastic native contract,
with independent RNG and a separate assessment of calibration quality.
The TMML reader preserves native elicitation and MCMC controls as metadata and
never silently maps them to a different solver or sampler.

`read_toxfinder_subject_csv(text, reference_doses=...)` reads ten-field native
patient records. Dates and times remain literal identifiers. Outcome `2` means
unset; `recommend_toxfinder_native` retains those records but excludes them
from inference, stage selection and alternating-branch count. All admitted
patients still count toward enrollment capacity. At capacity it returns no new
dose, with a stage-one recommendation when observed history has not reached
stage two; it does not fabricate undefined extra native branches.

`read_toxfinder_simulations_csv(text, reference_doses=..., retained_draws=...)`
reads original archived simulation summaries and converts the native parameter
order to package order. Draw count and scales are explicit because old metadata
headers do not safely identify every parameter. The archived 2004 example
predates core 1.1.1; it validates recorded-draw report calculations, not modern
random-seed identity. Original cohort-log writing, GUI rendering and literal
format identity are not implemented. Across-trial SD/SEM labeling is corrected
explicitly rather than copied from the old GUI.

## Recovered numerical contract and evidence

The original core fits a MAP to **observed history**, with two bounded native
simplex passes (250 outer iterations each). Its unnormalized Gamma objective,
relative stopping rule, tie ordering, and row-zero budget exit matter. Ordinary
Nelder–Mead substitution would not establish equivalence. `fit_toxfinder_native_map`
returns both pass convergence flags instead of hiding an exhausted budget.

For each hypothetical candidate outcome, the same observed-history MAP supplies
the negative-log-posterior Hessian of augmented data. Eigenvalues are floored at
`0.1`; the score is `(1-p)*det(H0) + p*det(H1)`, with `p` the arithmetic posterior
predictive mean. **This is not expected log determinant.** The fixed grid has
30 indices `0..29`; its sweeps, crossings, strict ties, and CKP selection are
preserved. The final contour uses CKP selection without information fitting.

Native stage one uses the highest-ever treated first-drug dose, first observed
toxicity, midpoint expansion, and a scan excluding the allowed upper candidate.
At the first candidate crossing, original code apparently accesses index `-1`;
Python safely returns the lowest candidate. This intentional boundary correction
is documented, tested, and excluded from exact native parity claims.

`tools/reference_toxfinder_native.py` regenerates fixtures from an explicitly
supplied, SHA-256-pinned original executable. It executes bounded original x86
routines with disclosed runtime services (allocation/free, power, and NumPy
LAPACK eigensystem/Cholesky); controller tests supply declared prediction/score
callbacks. It does not run a Windows process or redistribute the executable.
Independent 40-digit Hessian differentiation verified the full information
chain; a separate 60-digit calculation verifies archived posterior reports.
See [source evidence](toxfinder-sources.json) and
[recovery audit](../research/toxfinder-native-recovery-2026-10-12.md).
