# BOIN combination accelerated-titration audit

## Source-backed behavior

The opt-in path in `simulate_boin_combination(..., titration=True)` implements
the accelerated-titration branch in the cached CRAN BOIN 2.7.2
`get.oc.comb()` source (`R/get.oc.comb.R`, SHA-256
`0bcd01f550cece9b7caae5b6070c259475cd6298f46d8d0d5b06f5b41a610814`). It
starts with one patient at the
requested starting cell. After a non-DLT, it moves one row or column upward;
when both moves are available the source chooses each with probability 1/2,
and at a grid edge it is forced along the remaining axis. A DLT ends the
prelude at that cell. Otherwise it ends after treating the upper-right cell.

The source then enters its ordinary `ncohort` loop. On the first standard
cohort, it tops the prelude endpoint up to `cohortsize` if fewer patients have
been treated there; subsequent standard cohorts enroll `cohortsize` patients.
Thus `cohorts` in the Python API continues to mean ordinary cohorts, and the
maximum total enrollment is
`cohorts * cohort_size + (rows - start_row) + (columns - start_column)` when
titration is active. The 1,000-patient limit includes those staircase visits.
For cohort size one, the CRAN simulator disables titration; Python records
`cohort_size_one` and runs ordinary one-patient cohorts.

The returned `titration_patients`, `titration_endpoint`, and
`titration_end_reason` preserve the prelude separately from later BOIN
allocation. For a request without titration, the reason is
`not_requested`, the endpoint is `(0, 0)`, and the titration count matrix is
zero. The simulator preflights a conservative working-state estimate of 64
bytes per trial-dose cell plus 768 bytes per trial and rejects estimates over
128 MiB; this is a bound on accounted arrays and result records, not a promise
about process RSS. NumPy random streams are not expected to match R's random
stream.

## Scope and differences

This implementation covers the cached CRAN simulation workflow only. The
app-level guide and wrapper source are not available in the local source
cache. Therefore the app's separately described titration cap, moderate-DLT
stopping, and optional 3+3 run-in are not inferred or claimed here. The Python
prelude has only the CRAN upper-right endpoint and first-DLT exit.

After the prelude, the existing Python simulator retains its established
complete-binomial cohort simulation, source-simulation movement and
convergence-stop behavior. R's internal random-number ordering is not a
compatibility promise. With `titration=False`, the existing simulation path
and its seeded results are unchanged.

## Independent integration checks

`tools/reference_boin_combination_titration.R` extracts and executes the
original cached R titration and first-cohort expressions with deterministic
outcome/direction tapes. Nine cases cover both free directions, forced edges,
initial/interior/upper-right DLTs, an upper-right start, and cohort-size-one
disabling. The fixture `tests/fixtures/boin-combination-titration.csv` agrees
exactly with the Python prelude counts, endpoint and first-cohort counts.

A separate comparison against the published `39a67b3` simulator reproduces
all existing result fields across 100 no-titration trials, including safety,
precision stopping, an upper-right start and single-patient cohorts. The
128 MiB working-state guard rejects oversized requests before changing an
explicit generator's state. This integration check took 1.612 seconds,
peaked at 121.55 MiB RSS and reported zero swaps. The seven focused simulation
checks also pass, with a 131.72 MiB peak and zero swaps; targeted Ruff and
mypy checks passed. No broad numerical suite was run locally.
