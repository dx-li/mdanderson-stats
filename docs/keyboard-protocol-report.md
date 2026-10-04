# Keyboard protocol report

`keyboard_protocol_report` builds a static Python protocol summary from one
validated `KeyboardDesign`, an explicit cohort plan, per-scenario true DLT
probabilities, a simulation count, and an integer seed. It computes the
decision-cutoff table once and runs scenarios serially from one NumPy random
stream in their input order. Only compact scenario summaries are retained.

The report records the effective key, safety, extra-safe and precision settings;
the cutoff table; scenario toxicity probabilities; selection probabilities and
Monte Carlo standard errors; mean patients and DLTs by dose; and stop-reason
frequencies. It also reports the probability of allocating strictly more than
60% or 80% of planned enrollment above target when a scenario has at least one
dose whose true probability equals the target; otherwise these metrics are
marked unavailable. Selection index zero is “No MTD.” The report is a static Python
snapshot, not an editable analysis session or a reconstruction of the app's
HTML/Word protocol template. It does not include post-trial MTD selection,
which remains a separate `KeyboardDesign.select_mtd` calculation. The simulated
selection probabilities do include each simulated trial's MTD estimate.

```python
from mdanderson_stats.keyboard import KeyboardDesign
from mdanderson_stats.keyboard_report import keyboard_protocol_report

report = keyboard_protocol_report(
    KeyboardDesign(target=0.3, extra_safe=True),
    [[0.05, 0.15, 0.30, 0.45], [0.10, 0.25, 0.40, 0.60]],
    cohort_size=3,
    cohorts=8,
    start_dose=1,
    trials=500,
    seed=2026,
)
report.write_html("keyboard-protocol.html")
```

Inputs are bounded before scenario copying or simulation: at most 20 scenarios,
20 doses, 200 planned patients, one million trial-by-dose cells per scenario,
and two million aggregate simulated patient replications. Scenario order is
preserved. The same explicitly recorded seed initializes one generator shared
across the serial scenario runs; this documents Python reproducibility and does
not promise parity with R's random-number stream.

Allocation-risk probabilities use the trial-level sum of patients treated at
doses whose true toxicity probability exceeds target, compared strictly against
60% or 80% of planned enrollment. Their Monte Carlo standard errors use the
Bernoulli plug-in estimate. The planned denominator remains fixed even when a
trial stops early.

The report distinguishes ordinary key movement from safety elimination and
extra-safe stopping. Safety uses the uniform Beta(1,1) overdose prior and
requires at least three patients at a dose; elimination removes that dose and
all higher doses. Each simulated trial is selected using the weak Beta(.05,.05)
prior and inverse-variance weighted isotonic estimation. For the completed
real trial, final dose estimation remains a separate `select_mtd` call. The
optional precision stop is triggered by enrollment at the current dose and is
subordinate to safety.

## Source crosswalk

The cached Keyboard app guide says users first configure the trial, run
operating-characteristic simulations, and then download an HTML or Word
protocol template; it describes scenario entry/upload and post-trial MTD
selection as separate workflows. The cached app shell exposes the target,
cohort size/count, start dose, safety cutoff and optional extra-safe offset,
scenario vectors, simulation count and seed, plus a distinct trial-data MTD
panel. See `research/raw/Keyboard/Guide.pdf` and `research/raw/Keyboard/app.html`.

The statistical rules and interpretation in this report follow the local
implementation guide in `docs/keyboard.md`; simulation and selection use
`KeyboardDesign` and `simulate_keyboard`. The cached dynamic protocol template
contents and exact native HTML/Word layout are not available in the static app
snapshot, so this report claims workflow-level usability rather than native
template or byte-level parity.

The Keyboard 0.1.3 R source cached with the combination package defines
`get.oc.kb` allocation-risk summaries in
`research/raw/KeyboardComb/Keyboard/R/get.oc.kb.R` and labels them in
`R/summary_kb.R`. It gates them on an exact true-to-target dose and compares
allocation above target with 60% and 80% of planned enrollment. The Python
summary uses an explicit per-trial row sum, which also defines behavior for
multiple exact-target doses and multiple higher doses; it does not reproduce
the R routine's special single-higher-dose branch behavior for
non-monotone vectors. The cached R implementation uses strict greater-than
cutoffs even though its manual says “or more.”
