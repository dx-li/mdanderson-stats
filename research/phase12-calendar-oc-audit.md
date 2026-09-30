# Six-dose Phase I/II calendar OC audit

## Scope and provenance

The aggregate interface wraps the Python calendar simulator in
`parallel_phase12_calendar.py`; it does not introduce a new patient-level
conduct or posterior model. The primary cached C++ implementation is
`research/raw/P12Xuelin/extracted/Phase12Xuelin/Simulations/SimulationsMP.cpp`.
Its simulation cases and replication settings are around lines 108 and 238;
selection aggregation is around lines 64–68, and Cases 1–6/Test 1 are around
lines 324–402. Those source runs are evidence for source aggregation/reporting
scope, not a claim that Python reproduces the entire native simulation design.

## Summary definitions

Each completed trial contributes exactly one mutually exclusive selection
outcome: early selected dose, final selected dose, or no selection. Early and
final selection vectors preserve the calendar's raw returned decisions,
including its early-selection eligibility quirk. Optional optimal-set
probability is explicitly defined as selection of any dose in the caller's
set; neither the source nor this wrapper supplies an inferred optimal-dose
set.

Generated endpoint totals use all simulated patient truth. For observed
endpoint totals, each trial is snapshotted at its final analysis time when one
exists, otherwise at its stop time. Efficacy and toxicity denominators are
counted separately from the endpoint tally's observed/nonpending cells. Thus
observed response and toxicity rates are not divided by all enrolled patients
when follow-up remains pending. The per-dose rates pool records across trials;
the reported MCSE uses the delta-method trial-level ratio contributions
`(event_i - rate * observed_denominator_i)` and the usual sample-variance
correction. A zero pooled denominator yields an undefined rate and MCSE.

Means and event probabilities use completed trials as independent units. The
final-analysis-time mean is conditional on trials with a final fit, and its
denominator is returned explicitly. Enrollment-stop-time summaries use each
trial's stop time; they do not imply complete post-enrollment follow-up. The
wrapper retains aggregate vectors and a bounded seed vector, not
individual trial histories or posterior draws.

## Reproducibility and resource policies

Trial seed `i` is derived directly from the original master seed with
`SeedSequence(master_seed, spawn_key=(i,))`; deriving one seed never affects a
later seed. A returned trial seed replays `simulate_phase12_calendar` when
called with the identical scenario and settings. Calendar data and posterior
random generators remain separate. This isolates sampler consumption from
data-stream consumption, but does not force identical simulated outcomes when
posterior decisions lead to different treatment allocations.

The preflight upper-bounds patient assignments, calendar attempts, importance
component evaluations (`67 * integrations` per importance fit), mode
iterations, and configured MCMC chain-transition slots. The MCMC quantity is a
transition proxy; it is not a likelihood-evaluation count. The elliptical-
slice MCMC loop can make up to 1,000 shrink proposals per coordinate update;
this is distinct from the importance optimizer's mode-iteration cap. One-trial allocation is estimated conservatively from
record, attempt, analysis, fit, seed, time, and compact result storage. This is
not a guaranteed process-RSS ceiling because Python, BLAS, and other runtime
allocations are outside the result's accounting. All work is serial.

Work and allocation limits are Python safeguards. The archived C++ source's
published simulation scenarios are not fully reconstructed here, and no
native RNG, posterior fitting, or full OC-report parity is claimed.

## Focused validation

The wrapper was checked against independently replayed calendar trials from
its returned per-trial seeds. Tests compare selection partitions, assignment
and generated endpoint totals, and observed endpoint counts at the same
per-trial analysis cutoffs. A separate replay check reconstructs the pooled
response-rate MCSE from per-trial residuals. The importance-backend test
exercises actual posterior fits; an over-budget request is rejected before
simulation. The documented 20-trial MCMC example executes successfully.

On the focused run, the new OC tests plus the existing calendar and importance
calendar tests passed (10 tests, 1.95 seconds, peak RSS 143,605,760 bytes,
zero swaps). Targeted Ruff checks/format and mypy for the new module passed.
