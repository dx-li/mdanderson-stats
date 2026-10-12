# Current mTPI desktop workflow

The version 2.1 (2012) download for catalog 72 declares two spreadsheet tasks:
monitoring tables and scenario simulations. Python now connects their inputs,
controllers, final estimators, operating characteristics and reopenable reports.
Original Excel, XLL, R, manuscript and guide files are external reference assets;
none is distributed here. See [recovery evidence](../research/mtpi-desktop-recovery-2026-10-12.md).

```python
from mdanderson_stats import (
    MTPIDesktopStudy,
    MTPIDesktopTableStudy,
    mtpi_desktop_table_report,
    simulate_mtpi_desktop,
    mtpi_desktop_simulation_report,
    read_mtpi_desktop_report,
)

monitoring = mtpi_desktop_table_report(MTPIDesktopTableStudy())
study = MTPIDesktopStudy(
    target=0.3,
    epsilon_lower=0.05,
    epsilon_upper=0.05,
    maximum_patients=30,
    cohort_size=3,
    starting_dose=1,
    scenarios=((0.1, 0.2, 0.3, 0.45), (0.25, 0.4, 0.55, 0.7)),
    trials=2000,
)
result = simulate_mtpi_desktop(study, seed=72)
report = mtpi_desktop_simulation_report(result)
reopened = read_mtpi_desktop_report(report)
assert reopened.study == study and reopened.seed == 72
```

`read_mtpi_desktop_workbook(bytes, sheet_name=...)` imports one explicitly selected
original worksheet. It reads numeric active inputs, rejects formulas with possibly
stale cached values, and never executes VBA or loads the XLL. The importer bounds
compressed/uncompressed sizes, member counts and XML. Monitoring inputs require no
invented scenario grid. `study.to_json()` and `read_mtpi_desktop_project()` save and
reopen versioned scientific inputs. `read_mtpi_desktop_report()` additionally
recovers the profile and seed; use that function to reproduce a simulation report.

## Scientific and compatibility profiles

The default `safe` profile uses the existing paper-facing `MTPIDesign`: uniform
allocation/safety prior, persistent exclusions, explicit numerical ties and final
isotonic selection among admissible tried doses. It clips the final cohort to the
planned maximum. The name distinguishes safeguards from legacy defects; it is
not a claim of clinical validation. Existing TPI/mTPI APIs are unchanged.

Explicit `profile="desktop_legacy"` preserves the recovered XLL controller:

- Beta(1,1) allocation; Beta(.5,.5) lowest-dose safety and top-dose underdosing stops.
- Next-dose overdose shape `Beta(y_next+.5, n_next+.5)`, omitting toxicity subtraction.
- A carried next-dose indicator at the top and an exclusion marker that may be overwritten.
- Strict UPM winners, excessive-underdosing stop above .8, overdose cutoff above .95.
- Complete final cohorts, possibly exceeding a nondivisible planned cap.
- Beta(.005,.005) inverse-variance isotonic final estimates, including the lower
  untried prefix, with zero-based dose offsets `i*1e-10` from the second dose and
  higher-dose preference on exact distance ties.

`select_mtpi_desktop_legacy_mtd()` exposes that final estimator separately. The
previous `select_mtpi_legacy_mtd()` retains the R script's different one-based
perturbations and first-tie selection. History-dependent exclusion and no-selection
flags are explicit inputs; terminal counts cannot reconstruct them.

Starting at the highest dose leaves the native indicator undefined; the legacy
profile raises instead of inventing a scientific value. Safe mode supports that
start. Actual legacy enrollment must remain within the package's 200-patient
bound. `replay_mtpi_desktop_trial()` accepts a supplied uniform tape, retains
cohort sizes, dose path, counts, stop reason and history, and flags assignments
already unsafe under the paper's uniform-prior posterior. Scenario indices are
zero-based; dose and flagged-cohort labels are one-based.

Native monitoring tables have a separate VBA contract: overdose overrides to DU;
otherwise a strict UPM winner determines E/S/D and exact ties remain blank. Safe
tables preserve existing paper API tie handling. These tables do not replace
trial-level safety decisions.

## Simulation, reports and numerical evidence

Every supplied scenario is executed in order. Selection bins are doses 1..D,
then no selection (unlike `simulate_mtpi`, whose no-selection bin is first).
Reports include truth, selection and binomial MCSE, mean allocation and sample
MCSE, mean total enrollment, pooled DLT/enrollment, distinct safety/underdosing/
capacity probabilities, and unsafe-assignment diagnostics. Enrollment MCSE is
undefined for a single trial. A zero plug-in binomial MCSE is not proof that an
unobserved event is impossible.

Legacy seed 2 uses the original MT19937 uint32 midpoint stream and advances by
actual enrollment across all trials and scenarios. Other legacy seeds are a
Python extension. Safe mode uses NumPy's default generator. Bounds are 100
scenarios, 100 doses, 100,000 trials per scenario, 2 million trial-dose cells and
10 million worst-case uniforms; joint work is rejected before allocation.

Portable references in `tools/reference_mtpi_desktop.py` verify original asset
hashes and regenerate six fixtures. They establish:

- 2,496 original XLL decision cases, with SciPy substituted only for the beta CDF.
- 432 complete unchanged-R trial replays, including whole-cohort cap overflow.
- 2,000 original native RNG values through multiple state regenerations.
- Original final-ranking segments, including an exact-distance tie.
- 495 cached monitoring cells and all 20,000 trials from ten cached scenarios:
  mean allocation and pooled toxicity match exactly; selection differences are
  at most 1.25e-16 from native no-selection subtraction.

Cached workbook evidence is not a fresh Excel execution. The native probes do
not establish full Windows initialization or native beta-library identity.
Original R helpers and separate exact-rational PAVA checks support the estimator;
the native ranking probe takes supplied fitted means. Source compatibility,
mathematical correctness and clinical validity remain separate claims.

The original 2007 TPI multiplier tuning and informative-prior loss recalibration
are separate [TPI](tpi.md)/[prior-sensitivity](mtpi-prior-sensitivity.md) extensions,
not editable inputs of the current spreadsheet. Their previously documented
limits remain explicit; this workflow does not manufacture missing procedures.
