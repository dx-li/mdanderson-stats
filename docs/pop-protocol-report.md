# PoP protocol simulation report

`run_pop_protocol` computes bounded, reproducible operating characteristics for
named, nondecreasing dose-toxicity scenarios. It captures the design, cohort
plan, titration and early-stop options, risk cutoff, and seed, then runs the
existing `simulate_pop` implementation for every scenario. The report stores
only compact summaries and a complete integer-count boundary table.

```python
from mdanderson_stats.pop_design import PoPDesign
from mdanderson_stats.pop_protocol_report import PoPReportScenario, run_pop_protocol

report = run_pop_protocol(
    PoPDesign(target=0.25),
    (
        PoPReportScenario("target at dose 2", [0.10, 0.25, 0.45]),
        PoPReportScenario("no exact target", [0.08, 0.20, 0.40]),
    ),
    total_patients=18,
    cohort_size=3,
    trials=100,
    start_dose=1,
    titration=True,
    earlyterm=True,
    risk_cutoff=0.8,
    seed=175,
)
html_text = report.to_html()
print(html_text[:300])
```

The scenario seed list is derived from `SeedSequence.spawn` in input order, and
the simulations run serially. Each scenario's seed is retained in the report so
that its existing simulation can be replayed independently. The true MTD is the
first dose minimizing absolute distance from target, matching the package's
`which.min` convention when two doses tie. Risk under and over are the existing
simulator's strict comparison against `risk_cutoff * planned_patients`; they
remain defined even when no truth probability equals target.

The report includes selection probability and Bernoulli MCSE (no selection,
then doses), mean treated/toxic counts by dose and in total, all-dose exclusion stop
probability and MCSE, and the source-defined under/over allocation risks and
MCSEs. It records the risk cutoff actually passed to Python. The cached PoP R
wrapper exposes that option but fails to forward it to its inner function,
which therefore uses its default 0.8; this report does not reproduce that
wrapper omission.

For the customary risk cutoff of 0.8, at most one of the under/over events can
occur. At custom cutoffs below 0.5, the native R routine uses an `if/else if`
and records at most one direction when both thresholds are crossed, whereas
the current Python simulator computes the two directional flags independently.
The report faithfully exposes the Python simulation values and does not claim
native parity for that custom-cutoff corner.

Boundary rows cover every integer patient count from 1 through the planned
maximum, independent of the cohort multiple. Strict predictive-Bayes-factor
comparisons are used. Impossible actions use `-1` for a maximum DLT count and
`n+1` for a minimum DLT count. Simulation results and report inputs obey the
existing 100,000 trial-dose-cell and 100,000 trial-patient limits per
scenario, with a two-million aggregate planned-patient-replication limit.

Final selection follows the executable selector: Beta(0.05,0.05) posterior
means are fitted by inverse-variance-weighted increasing isotonic regression,
then perturbed by `1e-10` times treated-dose rank. It chooses the closest
admissible estimate and the higher dose on an exact remaining distance tie.
The separate safety rule uses a Beta(1,1) posterior tail above 0.95 after the
configured minimum patient count, and excludes that dose and all higher doses.

The cached package manual's `plot.pop` documentation mentions 95% toxicity
credible intervals, but its executable selector returns the MTD and isotonic
estimates only, and its plot draws those estimates and the target line. The
report does not infer an interval rule. The app HTML contains conditional
`plus3` output hooks but no corresponding input control; comparison behavior is
unresolved in the inspected cache, so this report makes no claim about it. The output is an independent
Python report, with separate [selection plots and portable scenario inputs](pop-community-workflow.md).
Native protocol templates and RNG parity are not reproduced.
