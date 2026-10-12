# PRT saved simulation and operating characteristics

`simulate_prt` connects a saved simulation `PRTStudy` to the recovered PRT1.0
arrival, hazard, screening, interim and final controllers. `simulate_prt_batch`
runs its saved repetition count and exports JSON/CSV operating-characteristic
reports. Both require `profile="legacy_1_0"`; the legacy defects described below
are explicit compatibility choices. See the [installed example](../examples/prt_simulation.py).
Original labelled settings are imported with `import_prt_simulation`; saved
studies reopen with `read_prt_study`, and `dataclasses.replace` creates edits.

The generator uses exactly800 sorted uniform arrival times over
`[0,800*mean_interarrival]`, the recovered beta-shaped hazard calibration, and
the native shifted-rate/reused-variate inverse. It draws candidate-by-dose
potential delays up front. Independent NumPy streams separate generation from
posterior computation. This preserves the recovered distributional choices,
not the executable's uniform consumption order, Gibbs sampler, seeds or RNG.
Reports save the generation and inference initial states and complete tapes.
`run_prt_simulation_calendar` replays those tapes with the saved inference state.
All arrivals/delays must be finite and nonnegative; delays beyond the window
represent non-events. Integer-day interval endpoints are required.

Patients start at dose zero in cohorts of three; the final cohort can contain
one or two. Interim assessment occurs at the next candidate arrival. Suspension
declines that candidate and rechecks at the following arrival. Source screening
counts patients with any positive observed interval separately from fully
assessed patients. The native simulation controller is distinct from conduct:
it has a configured top dose and its own de-escalation/final selection rules.
The shared posterior analysis retains projection, proper-moment, Rhat, MCSE and
individual-chain agreement checks. Invalid native selections, unstable or failed
fits return unresolved holds, never recommended doses.

Two native calendar defects remain visible: exactly-window censoring maps to
interval zero, and the final clock considers only the last three enrolled
patients. Earlier patients can therefore remain pending. Native final screening
can also bypass inference and reach a selector using retained summaries; Python
returns `final_inference_skipped` instead of recommending from stale or
uninitialized values. Early safety stops retain the events observed at stopping;
future events are not added. Calendar time and duration are measured from time
zero in days, including the wait to the first generated arrival.

Each trial reports `selected`, `safety_stop`, `numerical_hold`,
`final_inference_skipped`, `arrival_exhausted`, `analysis_budget`, or `fit_budget`.
Only `selected` supplies a dose. All requested replicates remain in every batch
denominator. Selection, safety-stop and unresolved probabilities sum to one.
An unresolved replicate is not a resolved no-dose selection. Assignment/event
means and mean duration explicitly describe observations as of termination;
with unresolved runs they are incomplete operating characteristics. Status
counts and `complete` make this distinction machine-readable. Two runs in the
example verify the workflow, not the precision of a design's operating
characteristics or its clinical validity.

Bounds are800 arrivals,500 enrolled patients,10 doses/intervals,800 calendar
analyses and200 fits per trial; default analysis/fit limits are200/50. Per-fit
cell/work and likelihood caps also apply. A batch allows at most100 runs, one
billion permitted likelihood calls and two million retained-history cells.
Larger studies must be explicitly split without dropping unresolved runs.

Validation includes21 original-instruction fixture files,111 precision boundaries,
72 composed enrollment steps, independent posterior quadrature and an actual
quality-qualified two-stage posterior example. Calendar fixture composition
supplies known decisions to isolate timing; separate963 simulation-controller,
144 final-selector and208 screening cases validate those components. None of
these claims pathwise native posterior/RNG equality. The paper profile and its
separate explicit-calendar API remain documented in [PRT](prt.md) and
[paper calendar](prt-calendar.md).
