# General Multc99 workflows in Python

Catalog entry **3 is implemented** against the recovered institutional
Multc99 2.1 C source. It covers compound and conditional outcome monitoring,
single and mixture historical Dirichlet priors, prior elicitation, precision
planning, trial replay, cohort/period-count simulation, randomized arms,
manual boundaries, probability-curve data and portable study files. This is
a corrected Python workflow; native executable bytes, menu files, random
streams and undefined C behavior are not reproduced.

The fixed-reference Phase IIa subset remains available through the
[existing Multc guide](multc.md). Multc Lean is a separate version contract
and remains partial because its generated endpoint-timing rules are unresolved.

## Define elementary outcomes and events

Each participant has exactly one mutually exclusive elementary category.
Compound events select categories with binary masks. For conditional events,
the numerator mask is the intersection and the conditioning mask is the
larger parent subset. Counts outside that subset do not update its rate.

```python
from mdanderson_stats import Multc99Event, multc99_design, simulate_multc99

# Categories: response+toxicity, toxicity only, response only, neither.
events = [
    Multc99Event(
        "response", (1, 0, 1, 0), lower_margin=0.05, lower_cutoff=0.10, event_type="efficacy"
    ),
    Multc99Event(
        "toxicity", (1, 1, 0, 0), upper_margin=0.10, upper_cutoff=0.90, event_type="adverse"
    ),
    Multc99Event(
        "response without toxicity",
        (0, 0, 1, 0),
        conditioning_definition=(0, 0, 1, 1),
        lower_cutoff=0.10,
    ),
]
design = multc99_design(
    experimental_prior=[0.5, 1, 1.5, 2],
    historical_prior=[[2, 3, 5, 10], [4, 2, 2, 2]],
    historical_weights=[0.3, 0.7],
    events=events,
    max_subjects=20,
    min_subjects=6,
    cohort_size=2,
)
state = design.monitor_counts([0, 1, 1, 4])
probability = design.event_probability("response", [0, 1, 1, 4], margin=0.05)
result = simulate_multc99(design, [0.1, 0.1, 0.4, 0.4], trials=100, seed=17)
print(result.early_stop_probability, result.mean_sample_size)
```

Experimental counts update a Dirichlet prior; each selected rate has its
aggregated Beta posterior. Historical rates are independent and unupdated.
Mixture component weights remain fixed and must be supplied explicitly.
`event_probability` returns `Pr(experimental - historical > margin)` and a
quadrature error estimate. Proper priors and explicit inputs are required.

Lower stopping is strict (`probability < lower_cutoff`); general-design upper
stopping is inclusive (`probability >= upper_cutoff`), as in the recovered
boundary implementation despite strict native prose. None disables a side.
Before `min_subjects`, compound events use the semi-free-ride limits implied
by the bounds at the minimum. Conditional tables use their parent counts;
the source also applies lower-bound run-back to those tables. All simultaneous
event/side hits are retained. There is no prior-only stop or interim stop at
the cap. Tables use -1 and n+1 for unattainable stops rather than +/-99.

## Prior elicitation and sample-size planning

```python
from mdanderson_stats import multc99_prior_from_interval, multc99_precision_sample_size

prior = multc99_prior_from_interval(
    [0.1, 0.2, 0.3, 0.4],
    [1, 0, 1, 0],
    width=0.2,
    coverage=0.9,
)
planning = multc99_precision_sample_size(
    design,
    "response",
    target_posterior_mean=0.4,
    width=0.25,
    coverage=0.9,
    search_cap=500,
)
```

Prior elicitation selects one mean-centered compound-event interval and solves
for a shared Dirichlet concentration. It follows the source's selected-event
alpha bracket 1e-7–10,000, with the Python concentration cap applied. An
unbracketed target raises; this is not a proof of global infeasibility.

Precision planning takes the first sample size meeting an equal-tailed Beta
interval-width target. The source rounds expected event counts toward the
posterior mean closer to **0.5**, preferring ceiling on a tie; Python retains
that declared rule and excludes impossible counts. For conditional events the
result counts observations in the conditioning subset, not total enrollment.
The search cap is explicit and independent of the design's enrollment cap.

## Replay, timing and randomized arms

`run_multc99_trial` accepts a full potential tape of `max_subjects` zero-based
category indices and returns counts/history through the first stop or cap.
Default interim looks occur after complete cohorts strictly before the cap;
an explicit increasing `look_sizes` vector can replace them.

`simulate_multc99` retains actual sample sizes, category counts, event/side
hits, sparse joint hit patterns, Monte Carlo errors and the native empirical
10/25/50/75/90% sample-size order statistics. It draws a full categorical
potential tape before drawing each schedule. The cap patient is counted.
To use native period-count monitoring, supply both `monitoring_period` and
`accrual_rate`, optionally `response_window`. Poisson counts determine looks;
the window shifts the first observation interval. Positive increments are
sampled directly instead of looping through empty periods. This does not
specify individual arrival times, pending-outcome imputation or endpoint delays.

```python
from mdanderson_stats import simulate_multc99_randomized

randomized = simulate_multc99_randomized(
    design,
    [[0.1, 0.1, 0.3, 0.5], [0.1, 0.1, 0.5, 0.3]],
    target_event="response",
    reassign=True,
    trials=100,
    seed=19,
)
print(randomized.selection_probability)  # None, arm 0, arm 1
```

For randomized workflows, `design.max_subjects` is a **total assignment-slot
budget**, not a per-arm budget. The two stages of the initial allocation are
balanced and shuffled independently. A global cohort triggers monitoring of
the arm receiving that slot, using its own observed counts. Stopped-arm future
slots are skipped or reassigned in balanced, shuffled order. Selection uses
the target event's correct conditional posterior mean among surviving arms;
ties are uniform. Efficacy targets maximize, adverse targets minimize; other
targets require explicit `maximize`. All-arm termination selects None.
`run_multc99_randomized_trial` supports supplied allocation and arm tapes.

## Editable boundaries, curve data and saved studies

`multc99_with_boundaries` creates an immutable design with explicit monotone
integer tables. Rows follow event order and columns follow conditioning count
0–max_subjects. Suffix limits from `min_subjects` onward are validated; early
lower limits and compound upper limits are regenerated consistently.
Conditional upper prefix limits remain the supplied, validated values.
Manual tables drive conduct and are preserved in captured inputs.

`multc99_probability_curve(design, event_name, counts, margins)` provides
probability/error arrays and `.write_csv(path)` for external plotting. It uses
an explicit margin vector instead of the native floating 0.005-step loop.

```python
from mdanderson_stats import multc99_study_report, replay_multc99_study

report = multc99_study_report(result, title="Multc99 response/toxicity study")
report.write_html("multc99.html")
report.write_inputs("multc99-inputs.json")
replayed = replay_multc99_study("multc99-inputs.json")
```

Reports capture every actual design, event, manual table, scenario, simulation
setting and seed. Single-arm and randomized studies use the same versioned
format. Writes are atomic, and JSON replay checks schema, duplicate fields,
nonfinite values and size limits. Exact RNG replay requires the recorded
NumPy version. Ordered scenarios can be run serially with explicitly recorded
seeds, saving a report/input pair for each; no automatic cutoff-tuning method
is inferred from the native manual adjustment recommendations.

## Validation, limits and licensing

137 focused checks cover 85 independent 45-digit probability integrals, eight
C boundary tables, three C prior-elicitation examples and nine C planning
sample sizes, plus replay, conditional counts, exact tape enumeration, Monte
Carlo checks, timing rescaling, reassignment, uniform ties, manual boundaries,
curve data and saved study files. The legacy integrator deviates by as much
as 1.29e-4 in these references; the Python probability checks use independent
high-precision values rather than encoding those discrepancies as truth.
See the [audit](../research/multc99-audit.md) and
[source record](multc99-source.json) for corrections and the completion boundary.

Limits are 2–50 elementary categories, 1–32 events, 1–10 historical components,
500 subjects, component concentration at most 1e6, 100,000 estimated comparison
calls during boundary construction, and 10,000 serial replicates subject to a
20-million work budget. Curves allow 1,000 margins; input files allow 4 MiB.
Period-count simulation requires rate×period in [1e-6, 1e6] and window/period
at most 1e9. Failed numerical separation or infeasible bounded searches raise.

The adapted workflow retains Multc99's **noncommercial source-use terms**;
commercial source use requires upstream written permission. Its original
[README notice](../notices/mdanderson-multc99-readme.txt) is preserved, re-encoded
from Windows-1252 to UTF-8 without changing the content. This component is not
an unrestricted MIT relicense. Original C files and binaries are not bundled
or required at runtime.
