# Six-dose calendar duration summary audit

The pinned C++ output stores every simulated trial's enrollment-stop time,
including paths stopped before posterior analysis. `TrialDesign.cpp::InitSims`
sets seven zero-based sorted indices to
`n/40`, `n/20`, `n/4`, `n/2`, `n−n/4`, `n−n/20`, `n−n/40` using integer
division (lines 193–204). `TrialDesign::TallySim` stores the current stop time
for each simulation (lines 216–236). `TrialDesign::TallyScenario` sorts the
durations, scales days by `12/365`, computes the arithmetic mean and
`E[D²]−E[D]²`, and reads those indices (lines 250–269).

The member and console label call the second result “Std,” but the expression
is population variance in month-squared units; this implementation exposes it
as `duration_population_variance_months_squared`. Existing
`mean_enrollment_stop_time_days` and its MCSE remain unchanged. Native code
does not initialize an order statistic whose index equals/exceeds the number
of simulations. Python returns NaN for those positions, and returns the exact
indices alongside the values, rather than clipping or silently substituting a
different quantile.

Source: `research/raw/P12Xuelin/extracted/Phase12Xuelin/DFKernel/TrialDesign.cpp`,
`InitSims` (lines 193–204), `TallySim` (216–236), `TallyScenario` (250–269),
and `PrintResults` (951–967).
