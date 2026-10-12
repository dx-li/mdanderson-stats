# PRT69 guide workflow closure

The recovered PRT1.0 Python workflow now includes original conduct/simulation
settings import, saved JSON/edit/reopen, posterior analysis, separate conduct and
simulation controllers, generated timing, complete calendar paths, batch
operating characteristics and JSON/CSV reports. The final acceptance manifest
is `prt-simulation-validation-2026-10-12.json`:76,279 full tests,5,187 final-wheel
PRT tests,9,347 broader regressions,28 extras and all four installed examples pass.
Catalog69 is implemented for this declared scope; current totals are103/31/4.

| Guide contract | Python workflow and evidence |
|---|---|
| Conduct settings and three-column patient history | `import_prt_conduct`, `PRTPatient`, `PRTStudy`, strict saved JSON;88 native count cases |
| Conduct posterior/decision output | `analyze_prt_conduct`, explicit paper/native profiles, per-chain reanalysis, quality holds and versioned JSON/CSV; native1368 controller and17 dispatch cases |
| Simulation settings, truth, accrual, hazard and repetitions | `import_prt_simulation`, immutable study;6 full800-arrival tapes,24 calibrations and288 event transforms |
| Cohort enrollment/suspension/resumption and final partial cohort | `run_prt_simulation_calendar`;18 source paths with72 composed steps, supplied controller decisions isolate bookkeeping |
| As-of follow-up and final clock |64 native observation cases,3 final clocks,90 calendar-control states and111 precision boundaries; actual CRT53-bit precision independently recovered |
| Interim/final inference and decisions | Shared independently validated posterior inference;963 native simulation decisions,144 final selectors and208 screening states; conduct rules are not substituted |
| Selected proportion, per-dose patients/events, no-dose stop and time | `simulate_prt_batch` JSON/CSV; all requested trials in each denominator, unresolved probability separate, assignment/event/time summaries explicitly as-of termination |
| End-to-end and saved replay | Real two-stage posterior example selects dose0, assigns3/3, duration110days, maxRhat1.000354 and riskMCSE0.000238; two generated pilot trials complete and replay exactly from saved tapes/inference state |

The two generated pilot trials both select dose0. They average3patients per dose,
0/1observed events per dose, and163.09235458149664days. This tiny deterministic
example validates composition and report arithmetic; it is not an OC precision
claim, design calibration, reproduction of the guide's30-run table, or a clinical
validation. The original diffuse-prior six-patient conduct example remains on
hold when covariance projections or separate-chain decisions are invalid.

## Explicit compatibility and safety boundaries

Native max-min/lower-floor projection, raw patient-mean predictive products,
complemented decision cutoffs, shifted hazard inverse, uniform-order-statistic
arrivals, exactly-window censor defect and last-three final clock require the
explicit `legacy_1_0` profile. Paper arithmetic remains separate. The Python
elliptical-slice sampler and split NumPy streams are independent; no native Gibbs
RNG, seed, exact GUI or file-format identity is claimed.

Native screening4674–47fe jumps directly to97e1 when it skips fitting. At final
enrollment that reaches the selector without a new posterior. Python marks
`final_inference_skipped` unresolved rather than using stale/uninitialized values.
Invalid projection/moments/endpoints/native selections, inadequate Rhat/MCSE or
chain agreement, finite-arrival exhaustion and computation budgets also remain
explicit unresolved outcomes. The batch does not discard them or relabel them
as clinical safety stops. Early stops do not count future toxicity events.

Scope closure means the advertised workflows have defined, tested outputs,
including honest invalid/uncertain outcomes. It does not certify the historical
algorithm's scientific correctness or its suitability for a clinical study.
Original binary assets remain ignored/reference-only. Licensing restrictions and
notices are unchanged; no publication, deployment, merge or outreach follows.
