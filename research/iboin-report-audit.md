# iBOIN captured-input reporting checkpoint

October 8, 2026. Catalog entry 145 remains partial.

The main guide identified report generation as an unfinished workflow after
boundary, conduct, final-selection and serial-simulation implementations.
`iboin_report.py` now composes the existing validated simulator with complete
captured settings, exact trial seeds, versioned JSON replay and atomically saved
HTML. It adds no statistical estimator or inferred native default.

The report preserves the distinction between original and robust-effective
historical ESS, prior-independent safety, and explicit final-selection policies.
It reports all-trial selection/no-selection and stopping probabilities,
per-dose enrollment/grade-2/DLT means with MCSE, total enrollment and quantiles.
Single-repetition MCSE remains undefined. Portable inputs preserve uint64 seeds
as exact JSON integers and restore an explicit uint64 array before simulation;
automatic NumPy inference can otherwise promote mixed large/small integers to
float64 and lose seed bits.

`tests/test_iboin_report.py` checks exact input/result replay, independent
patient-level trial aggregation, custom weights/eligibility, robust ESS,
uint64 boundary seeds, escaped HTML, saved JSON replay, single-trial undefined
MCSE, caller-array independence, schema rejection and simulation resource
limits. Existing `test_iboin.py`, `test_iboin_trial.py` and
`test_iboin_final_simulation.py` remain the statistical/conduct reference checks.

No new native source or application execution was obtained for this checkpoint.
An October 8 request for the official final-selection PDF was denied with HTTP
403 before an origin response was received. The cloud network policy did not
allow `biostatistics.mdanderson.org`; its addition was saved in the environment
draft, preserving the package-manager presets. Saving the draft does not apply
the runtime policy. Recheck the source request after the setting is applied.
Native weighting, tie handling, final-prior linkage, candidate conventions and
native file/report formats remain unresolved. Reporting closes a documented
community-workflow gap; it does not promote catalog status or change the
88 implemented / 42 partial / 8 pending counts.

## Executed validation

On the pinned Python 3.13 environment, the baseline full suite completed with
33,154 passed in 842.37 seconds, with no failures or skips reported. That run
started before the new report tests were added. The final focused run, including
21 new report cases and the 20 existing iBOIN cases, completed with 41 passed in
2.01 seconds under `-W error`.

Repository-wide Ruff lint and formatting checks passed; mypy passed across all
677 source files. Source distribution and wheel builds succeeded. The guide's
100-trial example executed from the installed wheel in a separate environment,
saved HTML/JSON and replayed the inputs exactly. Catalog parsing confirmed 138
entries with unchanged status counts, and `git diff --check` passed.
