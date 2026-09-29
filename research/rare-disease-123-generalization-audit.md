# RareDisease 1+a+b extension audit

## Recovered scope

The local official help artifact
`research/raw/1plus2plus3/123Design_Dose_Input.pdf` states that the generalized
design assigns cohorts of sizes 1, `a`, and `b`, for a maximum of 1+a+b
patients per dose. The matching local `app.html` exposes `a` values 1, 2, 3
and `b` values 1 through 5; defaults are 2 and 3. The default generated
protocol documents only the 1+2+3 decision table. The help and protocol do not
provide generalized decision tables or specify how the table's “at least
three patients” eligibility threshold changes with `a` and `b`.

The Python extension therefore retains the existing 1+2+3 decision logic and
full-cohort simulation, with source-backed cumulative cohort states
`0, 1, 1+a, 1+a+b`. For nondefault `a` or `b`, callers must supply
`admissibility_min_patients`; for the source-backed 1+2+3 default it remains
3. The existing n>=threshold toxic/futility eligibility rule, adjacent-dose
movement, higher-dose exploration restriction, and assignment-to-full-dose
selection behavior are explicitly Python policy extensions to generalized
cohorts; no native 1+a+b decision-table parity is claimed. In particular, no
within-cohort staggered timing is modeled. The simulator remains a complete
cohort, outcomes-observed-before-next-decision design with its existing latent
normal endpoint correlation model.

## Validation

The focused checks cover all 15 UI-supported `(a,b)` choices for cumulative
counts and cohort increments, required explicit thresholds away from the
default, default decision-table compatibility, generalized full-dose
selection, simulation path conservation and deterministic endpoint extremes,
including outcome accounting for the maximum 1+3+5 cohort configuration. Both
RareDisease123 test files passed: 9 tests in 2.17 seconds with warnings treated
as errors. Peak RSS was 157,728,768 bytes (150.42 MiB), with zero process
swaps. Ruff check/format and targeted mypy passed.
