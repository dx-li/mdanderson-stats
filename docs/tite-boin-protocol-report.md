# TITE-BOIN protocol simulation report

`run_tite_boin_protocol` builds a bounded, readable Python protocol report by
running the existing TITE-BOIN simulator for each supplied toxicity scenario.
Each scenario carries an explicit dose-level DLT probability vector. The report
captures the BOIN design, enrollment plan, assessment window and unit label,
accrual and timing settings, both analysis gates, trial count, seed convention,
and compact operating-characteristic summaries. It does not accept a detached
simulation result with caller-asserted metadata.

```python
from mdanderson_stats import (
    BOINDesign,
    TITEBOINProtocolRequest,
    TITEBOINScenario,
    run_tite_boin_protocol,
)

report = run_tite_boin_protocol(
    TITEBOINProtocolRequest(
        trial_name="Example dose-finding study",
        design=BOINDesign(target=0.3, early_stop_patients=6),
        scenarios=(
            TITEBOINScenario("Increasing toxicity", [0.05, 0.20, 0.40]),
            TITEBOINScenario("Lower toxicity", [0.03, 0.12, 0.24]),
        ),
        window=90,
        time_unit="days",
        accrual_rate=1 / 30,
        cohorts=4,
        cohort_size=2,
        start_dose=1,
        trials=100,
        seed=20261004,
    )
)
html_text = report.to_html()
print(html_text[:300])
report.write_html("tite-boin-protocol.html")
```

Scenarios run serially. A `SeedSequence` derives and records a 128-bit integer
seed for each scenario in request order; each existing simulation consumes its
recorded seed. Re-running the report with the same request reproduces those
streams. For Weibull and log-logistic event timing, the effective default
late-onset probability is 0.5 at each dose; it is unused for uniform event
timing. Under uniform event timing with no event-time trimester masses, event
times are uniform over the full window. Trimester masses are unused for
Weibull and log-logistic event generation. Analysis weights separately default to uniform conditional
time-to-DLT masses of `(1/3, 1/3, 1/3)`. The former controls simulated event
times; the latter controls interim STFT weights. The input unit label applies
to the assessment window, accrual rate, duration and suspension summaries; it
does not convert numeric values.

The report shows selection probabilities and their Monte Carlo standard errors,
mean enrollment and toxicity counts by dose, mean duration and suspension time,
and stop-reason frequencies. It does not contain patient-level outcomes or
choose a trial MTD from actual patient data. Final MTD selection remains a
separate complete-outcome operation through `BOINDesign.select_mtd`.

The flow summary follows the implementation's conduct order: safety and sticky
exclusions first; imputed ordinary BOIN transition and enabled 1/3 or 2/6
modification; completion suspension (except the observed-toxicity and 2/6
exceptions); minimum-follow-up suspension for an actual escalation; precision
stopping when the actual assignment stays; then continued enrollment until a
stop or the planned cap. After enrollment ends, outcomes are followed to
ascertainment before final MTD selection. The report documents this behavior;
it does not change the simulator.

Bounds are at most 20 scenarios, 1,000,000 trial-by-dose result cells per
scenario, and 2,000,000 planned patient replications in aggregate. The HTML is
escaped, limited to two million characters, and `write_html` atomically replaces
the requested UTF-8 file. This is a Python protocol summary, not the app's
HTML/Word/PDF template or an exact native scheduler or random-number claim.
