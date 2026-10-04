# TITE-Keyboard protocol and scenario report

`run_tite_keyboard_protocol` in `mdanderson_stats.tite_keyboard_protocol_report`
captures a validated `KeyboardDesign`, calendar settings, named dose-toxicity
scenarios, and an explicit seed. It runs the existing TITE-Keyboard calendar
simulator serially and returns an immutable compact report. `write_html` saves a
self-contained, escaped Python HTML report atomically.

```python
from mdanderson_stats import KeyboardDesign
from mdanderson_stats.tite_keyboard_protocol_report import (
    TITEKeyboardProtocolRequest,
    TITEKeyboardScenario,
    run_tite_keyboard_protocol,
)

report = run_tite_keyboard_protocol(
    TITEKeyboardProtocolRequest(
        trial_name="Dose escalation example",
        design=KeyboardDesign(target=0.30),
        scenarios=(
            TITEKeyboardScenario("below target", [0.05, 0.15, 0.25, 0.40]),
            TITEKeyboardScenario("above target", [0.10, 0.30, 0.45, 0.60]),
        ),
        window=90,
        accrual_rate=1 / 15,
        cohorts=6,
        cohort_size=3,
        trials=500,
        seed=20261004,
    )
)
report.write_html("tite-keyboard-protocol.html")
print(report.scenarios[0].selection_probability)
```

The report includes the captured target interval and posterior key, extra-safe
settings, the decision order, posterior effective-follow-up transition
brackets, enrolled-count safety cutoffs, each scenario's selection probabilities
and Monte Carlo standard errors, dose-level mean enrollment and DLT counts,
trial-duration and suspension-time summaries, and observed stop-reason
frequencies. It stores summaries rather than patient histories. The simulation
caps the report at 20 scenarios, 100 doses, 200 planned patients per trial,
100,000 trials, one million trial-by-dose cells per scenario, and two million
aggregate patient replications.

The bracket coordinates are effective non-DLT count. Each finite pair gives the
adjacent representable values immediately before and at the posterior transition;
`[0, 0]` means the transition already applies at zero and `[NaN, NaN]` means it
cannot occur within the table's maximum effective total. The safety table is
indexed by enrolled patients and observed DLT count. Posterior transitions do
not replace the safety, pending-fraction, two-ascertained-outcome escalation,
precision-stop, or final-selection rules. All time values must use one common
unit.

For uniform conditional event timing, omitted generation masses use the
simulator's native default path and are reported as equal thirds. Separate
analysis follow-up masses may be supplied for non-adaptive timing. Parametric
Weibull or log-logistic timing instead accepts a scalar or per-dose probability
of a DLT in the late half of the window. The adaptive timing sampler is an
explicit Python extension with its own work budget and diagnostic thresholds;
it cannot be combined with manually supplied analysis masses. Adaptive runs
record separate outcome and sampler child seeds and compact fit-count, work,
split-R-hat, and pending-weight MCSE summaries. Diagnostic thresholds are
empirical checks, not guarantees of convergence.

The saved file is a Python report format. It does not reproduce the app's native
HTML/Word files or exact Figure 1/Table 1 layout. The cached app page exposes an
Operating Characteristics pane but not its column definitions, so this report
does not claim native correct-selection, regret, or overdose estimand parity.
See the [source and coverage audit](../research/tite-keyboard-protocol-report-audit.md)
and the existing [TITE-Keyboard methods guide](tite-keyboard.md).
