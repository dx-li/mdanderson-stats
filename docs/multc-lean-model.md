# Multc Lean native models and saved studies

Catalog entry **12 is implemented** for the recovered Multc Lean 2.1 functional
workflow. Python reads and writes the original desktop model schema, restores
its installed example defaults and composes exact operating characteristics
with optional legacy duration simulation. Portable HTML reports, editable
Markdown protocols, exact PMF/CDF figures and replayable JSON studies cover the
advertised output workflow. See the [functional audit](../research/multc-lean-model-audit.md),
[monitoring guide](multc.md) and [legacy timing guide](multc-legacy-duration.md).

Original numerical integrator, CLR/GUI, Windows random streams and native
Word/report formatting are compatibility differences. Python calculations have
independent numerical and original-control references; this is not a claim
that the original application has been run end to end.

## Recover the installed example

```python
from mdanderson_stats import MultcLeanModel, MultcLeanScenario, read_multc_lean_model

model = MultcLeanModel.example(scenarios=(MultcLeanScenario.independent(0.3, 0.25),))
assert model.response.historical == (30, 70)
assert model.toxicity.historical == (25, 75)
model.write_model("MultcLeanDesktop.model")
assert read_multc_lean_model("MultcLeanDesktop.model") == model
study = model.run(seed=2026)  # no timing simulation: both time fields default to zero
print(study.results[0].exact.expected_sample_size)
```

The installed reset example uses response historical `Beta(30,70)`, experimental
`Beta(0.6,1.4)`; toxicity historical `Beta(25,75)`, experimental `Beta(0.5,1.5)`;
both cutoffs 0.95, both margins zero, cap 30, minimum 1 and cohort 1. The inactive
historical constants are 0.3 and 0.25. The published statistical tutorial instead
uses toxicity historical `Beta(20,60)`; that is a distinct example with its own
validated boundaries. Native startup uses 10,000 simulation repetitions.

`MultcLeanEndpointInput` preserves the active and inactive historical fields.
Setting `use_standard_constant=True` activates `standard_constant`; otherwise
`standard_a` and `standard_b` define historical uncertainty. Experimental shapes
always define an experimental beta prior. Inactive historical values must be
finite but need not define a valid distribution; active values are validated.

## Run timed scenarios and save outputs

```python
from pathlib import Path
from mdanderson_stats import (
    MultcLeanEndpointInput,
    MultcLeanModel,
    MultcLeanScenario,
    replay_multc_lean_study,
)

endpoint = MultcLeanEndpointInput(1, 1, 1, 1, True, 0.5)
model = MultcLeanModel(
    endpoint,
    endpoint,
    max_subjects=6,
    response_cutoff=0.95,
    toxicity_cutoff=1,
    scenarios=(
        MultcLeanScenario((0.1, 0.2, 0.3, 0.4), 0.25, 2),
        MultcLeanScenario.independent(0.3, 0.25),
    ),
)
study = model.run(
    seed=2026,
    trials=32,
    max_unit_exponentials_per_trial=1000,
    scenario_names=("timed joint scenario", "untimed independent scenario"),
    show_potential_boundaries=True,
)
model.write_model("multc-study.model")
study.write_json("multc-study.json")
study.write_report("multc-study.html")
study.write_protocol("multc-protocol.md")
replayed = replay_multc_lean_study(Path("multc-study.json").read_text(encoding="utf-8"))
assert replayed.to_json() == study.to_json()
simulation = study.results[0].simulation
assert simulation is not None
assert simulation.replay_trial(0).duration == simulation.summary.duration[0]
assert study.results[1].simulation is None
```

Joint probabilities are `[both, response only, toxicity only, neither]`.
They must sum to one within 1e-12; Python never silently normalizes them. The
source UI's tolerance is 1e-10, so a rounded desktop model may need corrected
probabilities before Python accepts it. Independence is explicit through
`MultcLeanScenario.independent`; imported four-category probabilities retain
association.

The desktop enables simulation only when **both** mean interarrival and response
window are positive. Zero in either disables it, even when the other is positive.
Times use the same units and each lies in `[0,10000]` in the desktop profile.
The original runner first computes exact count operating characteristics, then
copies only simulation duration and balk means into its results. Python keeps
exact enrollment, responses, toxicities and sample-size PMF separate from the
legacy Monte Carlo duration/balk estimates and their additional MCSEs. The
underlying simulation also exposes its simulated enrollment/outcome summaries.
One-replicate MCSE is undefined.

The mandatory prior screen precedes minimum enrollment and returns zero enrollment
when it rejects. The cap records completion, not an adverse posterior stop.
Legacy simulation uses clipped/shared follow-up, latent counts and balked
arrivals; the observation-aware calendar API in the monitoring guide has its
own explicit timing assumptions. Use the legacy workflow for the recovered
historical timing calculation.

`seed` is required and captured. Python PCG64 replaces the original clock-seeded
Windows RNG; child scenario and replicate seeds allow replay within the recorded
Python dependency environment. JSON captures all input/run settings and names;
HTML captures exact results, timing means/MCSEs, bounds and seeds. Native model
text captures only the model/scenarios, as in the desktop; it has no seed,
scenario-name or repetition fields. Write methods replace files atomically.

## Export stopping plots

```python
import matplotlib.pyplot as plt

figure = study.plot()  # exact cumulative stopping probability
figure.savefig("multc-cdf.png", dpi=160)
plt.close(figure)
figure = study.plot(cumulative=False)  # exact sample-size probability
figure.savefig("multc-pmf.png", dpi=160)
plt.close(figure)
```

Plots require `pip install 'mdanderson-stats[plot]'`. The protocol includes actual
priors, strict rules, prior rejection, cohort bounds and replay inputs. It is
editable Markdown rather than a reproduction of the author's Word template.
The optional potential-boundary table reports reachable intervals; the cap row
is labeled completion. Empty models can produce bounds/protocols but have no
scenario plot.

## Desktop file schema and limits

Leading `%` header lines precede these data rows, separated by single ASCII spaces:

| Row | Fields in native order |
| --- | --- |
| Response | `standard_a standard_b experimental_a experimental_b use_standard_constant standard_constant` |
| Toxicity | Same six fields |
| Controls | `max_subjects min_subjects cohort_size response_cutoff response_margin toxicity_cutoff toxicity_margin` |
| Count | Number of scenario rows |
| Each scenario | `p_both p_response_only p_toxicity_only p_neither mean_interarrival response_window` |

`parse_multc_lean_model(text)` and `read_multc_lean_model(path)` accept UTF-8,
optional BOM, CRLF/LF/CR endings and finite decimal/scientific numbers. Python
writes 17-digit floats to preserve precision, instead of the desktop's shorter
.NET general format; timestamps/line-ending bytes differ. Extra rows, malformed
columns, invalid flags and invalid active parameters raise errors. The source UI
silently falls back to the example on some read errors; Python does not.

Inputs are bounded at 1 MiB and 100 scenarios. Cap/prior/cohort restrictions are
those in the monitoring guide. Exact work is limited to 10 million count states
per scenario and 50 million aggregate by default. Legacy simulation accepts
1–10,000 repetitions and 1–1,000,000 exponential variates per trial. The study's
default configured-draw budget is 500 million and storage budget is 64 MB;
default exponential allocation is 10,000 per trial. Work charges configured
arrays including unused variates. Aggregate preflight runs before numerical
work or RNG creation. Stream exhaustion raises without returning a partial study.
A scientifically valid large/slow-accrual model may need a different explicit
budget or fewer repetitions. Numerical tolerances and prior-screen differences
are recorded in the linked audits.

The Tools-menu calculations already have Python APIs:
`solve_distribution_moments`, `solve_distribution_quantiles` and
`compare_beta_difference`; see [Parameter Solver](parameter-solver.md) and
[Inequality Calculator](inequality-calculator.md). Original installers, EXE,
DLL, configuration and Word template remain excluded under the original
redistribution restriction; only independently authored code and synthetic
factual reference outputs ship.
