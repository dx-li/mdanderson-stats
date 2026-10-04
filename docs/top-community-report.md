# Community TOP study reports

`mdanderson_stats.top_report` adds a bounded, reproducible Python workflow for
named operating-characteristic scenarios. It composes the existing TOP design
and calendar simulation APIs; it does not introduce another model or
calibration algorithm.

```python
from mdanderson_stats.top_binary import TOPBinaryDesign
from mdanderson_stats.top_report import TOPBinaryScenario, top_binary_report

design = TOPBinaryDesign(
    max_subjects=12,
    null_rate=0.20,
    cutoff_scale=0.86,
    gamma=0.95,
    looks=[4, 8, 12],
    suspension="table",
    timing_probabilities=[0.5, 0.3, 0.2],
)
report = top_binary_report(
    design,
    [
        TOPBinaryScenario(
            label="null",
            response_probability=0.20,
            window=6,
            accrual_rate=2,
            trials=20,
            seed=20261004,
            arrival="exponential",
            response_distribution="uniform",
        ),
        TOPBinaryScenario(
            label="promising",
            response_probability=0.40,
            window=6,
            accrual_rate=2,
            trials=20,
            seed=20261005,
        ),
    ],
)
report.write_html("top-binary-report.html")
```

The two-endpoint workflow takes joint truth probabilities in `(11, 10, 01,
00)` order. Use `mode="coprimary"` for the co-primary combination rule or
`mode="efficacy_toxicity"` for efficacy and toxicity. In the latter mode, the
first coordinate is efficacy and the second is toxicity, whose adverse tail is
handled by the existing design implementation.

```python
from mdanderson_stats.top_endpoints import TOPMultiEndpointDesign
from mdanderson_stats.top_report import TOPMultiEndpointScenario, top_multiendpoint_report

design = TOPMultiEndpointDesign(
    max_subjects=12,
    null_joint_probabilities=[0.12, 0.28, 0.18, 0.42],
    cutoff_scale=0.8,
    gamma=0.5,
    mode="efficacy_toxicity",
    windows=[4, 6],
    looks=[4, 8, 12],
    suspension="strict",
)
report = top_multiendpoint_report(
    design,
    [
        TOPMultiEndpointScenario(
            "reference",
            (0.12, 0.28, 0.18, 0.42),
            2,
            20,
            20261006,
            arrival="fixed",
            truth_timing_probabilities=(0.4, 0.35, 0.25),
        ),
    ],
)
report.write_html("top-joint-report.html")
```

Reports accept 1–10 uniquely named scenarios and at most 2,000,000
trial-patient-endpoint cells per call. Each scenario needs an explicit integer
seed in `[0, 2**64-1]`; scenario seeds are passed independently to NumPy's
generator. Action probabilities use the number of simulated trials as their
denominator, and their MCSE is the binomial Monte Carlo SE across independent
simulation replications. Enrollment and calendar means include the usual
sample-mean MCSE across replications. These summaries describe simulation
precision, not clinical uncertainty. Boundary tables are the exact boundaries
returned by the existing design objects.

The report is standalone escaped static HTML written atomically to an existing
directory. It records the design/prior, suspension and analysis timing,
scenario truth, arrival and outcome-timing assumptions, seeds, actions,
denominators, enrollment and timing summaries. The cached native application
capture advertises a protocol-template download after boundaries and simulation
are available; that template file and its output schema are not present in the
source cache. This Python report is therefore not a native template, protocol
document, or claim of native scheduling/RNG/report parity.
