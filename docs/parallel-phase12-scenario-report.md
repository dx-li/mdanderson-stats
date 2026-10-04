# Parallel Phase I/II named scenario reports

`ParallelPhase12Scenario` and `simulate_parallel_phase12_scenarios` run a small,
ordered set of named truth scenarios through the existing four-arm operating
characteristic simulator. Each scenario records its toxicity and response
probability vectors, replicate count, integer seed and optional caller-defined
optimal-arm set. The result keeps those inputs beside the operating
characteristics computed from them and can render or atomically save a plain
text report.

```python
from mdanderson_stats import (
    ParallelPhase12Scenario,
    simulate_parallel_phase12_scenarios,
)

report = simulate_parallel_phase12_scenarios(
    (
        ParallelPhase12Scenario(
            "candidate A",
            [0.04, 0.09, 0.16, 0.25],
            [0.10, 0.20, 0.35, 0.50],
            n_trials=4,
            seed=8505,
            optimal_arms=(3,),
        ),
        ParallelPhase12Scenario(
            "candidate B",
            [0.04, 0.09, 0.16, 0.25],
            [0.10, 0.25, 0.32, 0.36],
            n_trials=4,
            seed=8506,
            optimal_arms=(2, 3),
        ),
    )
)
print(report.report())
report.write_report("parallel-phase12-report.txt")
```

Each seed reproduces that scenario's existing serial OC call. The report
includes the supplied truth vectors, fixed design settings, selection and
no-selection probabilities, stopping probabilities, admissibility, enrollment,
toxicity and response summaries, and their Monte Carlo errors where defined.
The four-replicate example is only a workflow smoke example; increase
`n_trials` for stable operating-characteristic estimates.
`optimal_arms` is an explicit caller annotation; it does not create or infer a
source-defined success criterion.

`parallel_phase12_scenario_from_native_input` reads exactly eight
whitespace-separated probabilities from the archived C input layout: four
efficacy values first, then four toxicity values. Replicate count, seed and
label are not present in that file, so the caller must supply them explicitly.

The request is limited to 20 unique labels and one million worst-case
patient assignments across the whole batch. Every request is validated and
the aggregate work bound is checked before simulation starts. Scenarios then
run serially; the returned records contain compact summaries, not patient
histories. A failed scenario raises and returns no partially completed batch.

## Source scope

The archived C input file contains four response probabilities followed by
four toxicity probabilities. It does not configure the trial rules: cohort
size, maximum enrollment, beta priors, stopping cutoffs and allocation rules
are compiled constants. This Python batch accepts named truth scenarios and
reuses the already implemented source-rule simulator; it is a convenience for
comparing scenarios and producing a durable report, not a reproduction of the
native output/RNG. The separate six-dose calendar-time C++ application is
covered by the calendar OC API and its duration summaries; its native
posterior-kernel printout remains outside current parity.

See the [source crosswalk](../research/parallel-phase12-scenario-report-audit.md)
for the archived input/output fields and the explicit boundary between the C
program and the Python scenario workflow.
