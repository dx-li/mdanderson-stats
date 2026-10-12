# mTPI archive final-estimator recovery — October 12, 2026

The official TPI_V2.1PID439.zip was retrieved again with normal HTTPS verification.
Archive SHA-256:
`4e87f1a816bbf1ffb1a7817ac2a87d3de019090e78d4c24c07ec6c8b44059c8c`.
The included `mTPI-middle.R` SHA-256 is
`2629c28a1bf607329b125d7fca6715d8d1bfd0615162ebc4d091b2c603f5f860`.
Original R, Office files and binaries remain in ignored research storage.

The final block establishes Beta(.005,.005) estimation priors, inverse-variance
PAVA weights, a prefix through min(maxdose,toxdose-1), candidate interval
mindose:tdose, and offsets i*1e-10 for i>=2. The source comment calls the weights
posterior variances; its executed expression is their reciprocal. Untried lower
doses are included in the fit. There is no additional final overdose filter.

`mtpi_legacy_selection.py` independently implements these mathematical operations
using SciPy isotonic regression. No original R function, source text or XLL is
included in the distribution. `MTPIDesign` and simulation behavior are unchanged.
The dedicated result separates unperturbed fitted means from selection scores.
The caller supplies native historical exclusion/stop state explicitly.

## References and validation

```
python tools/reference_mtpi_legacy_selection.py path/to/mTPI-middle.R --rscript Rscript
uv run pytest -W error tests/test_mtpi_legacy_selection.py tests/test_mtpi.py \
  tests/test_mtpi_prior_sensitivity.py tests/test_mtpi_isotonic_posterior.py
```

Reference execution used R 4.5.0 in a local, research-only runtime.
The runner pins the original file hash, extracts and executes its unchanged
helpers and final-estimation statements against ten synthetic terminal states.
It does not run the original trial generator or substitute a Python selector.
The original files are needed only for regeneration, not ordinary tests.
The committed fixture compares scores at rtol 3e-14 and atol 1e-16 and checks
selected doses/eligibility exactly. A separate Fraction-based max-min formula
checks all 60 configurations for n=(2,3,4), without PAVA, NumPy or SciPy in the
reference calculation. Focused validation passes 38 tests.

## Unresolved native controller contracts

The archive's trial script differs from both its monitoring-table script and
the paper-facing Python controller:

- Allocation uses Beta(1,1), lowest-dose stopping uses Beta(.5,.5), and final
  estimation uses Beta(.005,.005). Monitoring-table safety uses Beta(1,1).
- The next-dose overdose test uses .5+n as its beta shape, omitting toxicity
  subtraction. It updates toxdose before the actual allocation action.
- At the top dose, posterior Pr(p<pT)>.8 stops with no selection for underdosing.
- The next-dose indicator may be unset when starting at the top dose and is
  carried from prior iterations at that dose.
- Correction (2026-10-12): strict UPM maxima are mutually exclusive and the
  outer dose branches are nested. The earlier claim of multiple increments in
  one cohort was unsupported. Executed XLL and unchanged R decision references
  agree on 2,496 cases; see the desktop recovery audit.
- The 2:tdose perturbation loop incorrectly indexes downward for tdose=1.
  Python corrects this and handles empty candidate ranges explicitly.

The ReadMe identifies the Excel XLL as its simulation implementation. R-script
parity does not establish XLL parity. Scenario-based tuning, native report
workflows, and informative-prior loss recalibration remain separate gaps.
No catalog promotion is justified by this bounded recovery.

## Subsequent current-desktop integration

The [desktop recovery](mtpi-desktop-recovery-2026-10-12.md) now resolves the
current application's controller, XLL differences, seeded cached scenario
outputs and saved-input/report workflow. The above bounded estimator checkpoint
is historical; it is not the final current-application coverage statement.
Original2007 TPI tuning and prior-specific recalibration remain separately
qualified extensions, not controls of the current version2.1 spreadsheet.
