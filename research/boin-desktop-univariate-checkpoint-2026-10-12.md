# BOIN desktop single-agent and TITE integration

Catalog99 remains partial; counts remain **98 implemented /34 partial /6 pending**.
This checkpoint supersedes the prior combination checkpoint's unintegrated
single/TITE draft notes. Final acceptance results are recorded below once complete.

## Closed scope

- Saved UseDefaultBounds now gates custom numeric bounds consistently in conduct,
  selection and simulation. Inactive numeric values do not override defaults.
- Complete single-agent simulator matches192 original wrapper cases, including
  desktop titration endpoint min(cap+1,doses), convergence and extra-safe gating.
- Desktop TITE calendar matches2304 original traces (256 base plus2048 extended),
  including entry times, waits, counts, selections and reported duration. Safe
  replay removes the source future-outcome pre-stop and backward clock. The
  explicit legacy profile retains those defects and labels them in reports.
- Assessment units, monthly accrual, pending fraction, event distributions and
  informative analysis weights connect to saved inputs. STFT84 original IL cases
  agree; pending follow-up is always days. Simulation normalizes prior weights;
  the calculator uses valid raw probabilities. These are separate contracts.
- Rolling Six matches238 complete original traces and explicitly rejects two
  native over-enrollment loops. The source driver imposes a deterministic host
  patient budget; its guard is not a native error claim. Safe replay keeps
  exclusions sticky. Actual decision time and legacy reported duration remain
  separate, and unsafe legacy assignments are visible.
- Saved 3+3 and Rolling Six comparison flags now invoke recorded independent
  streams after joint request preflight. Twelve original3+3 cases validate
  fixed-selection expansion: planned cap, never shrink, never revise selection.
  Failed comparator replications raise; they are never omitted from summaries.
- English/Chinese single/TITE methods, method-specific simulation diagnostics and
  a combined reopenable report are integrated and publicly exported. Complete-
  outcome tables explicitly do not replace pending TITE decisions. Proprietary
  templates and original prose are not copied.
-480 original adverse single-MTD edited histories:383 source matches and97
  deliberate safe exclusions of native unsafe assignments. This is evidence for
  mathematical safety divergence, not blanket source-equivalence. No unsafe
  assignment occurs in the safe results.

## Precise remaining contracts

1. Recover and validate the active `get.boundary.tite.R` pending-state decision
   table, including strict-boundary rounding, pending gates, informative-prior
   interpretation and saved report/flowchart options. The source is available;
   it is actionable work, not an external blocker. Its old README describes a
   separate clinical-conduct website, so do not assume that absent website
   routine is an advertised desktop patient controller.
2. Audit standalone final-selection and calculator report workflows and complete
   the declared desktop capability crosswalk before catalog promotion. Native
   layout, RNG identity and floating-point identity are separate qualifications.
3. After99 closure, continue dependency-ordered recoverable partial applications.
   Six missing fitted-model contracts, mixed-license distribution decisions and
   Forbidden PR creation remain external/parent issues, not blockers to this work.

## Provenance and safety boundary

The hash-pinned original1.1.0 archive remains external and ignored. Original R,
DLLs, installer, default projects and templates are not distributed. Synthetic
fixtures and independently authored drivers are retained. Existing mixed-license
notices and commercial restrictions remain unchanged. Source agreement,
mathematical correctness and clinical validity are separate claims.

## Next contract now recovered

Managed `MainForm.GetBoundaryTITE` RVA0x1c1e4 always requests `stft=1`, uses the
window in months, honors UseDefaultBounds, and passes the pending percentage and
prior flag. `GenerateBoundaryTITEDecisionTableForDocument` paginates those returned
rows; pagination/template appearance adds no new statistical method.
`get.boundary.tite.R` MD5 is19803f88bc78c49cc9fe7104c0258c94.

The external-source driver `reference_boin_desktop_tite_table.R` now retains
**7,124 rows from64 requests**, covering ordinary and symmetric custom upper
bounds, target.25/.3, cohort1/3, pending.5/.65, uniform/informative priors and
extra-safe off/on. An independent temporary mathematical probe
`/tmp/boin-pending-table-probe.py` matched **61,916 evaluations** at endpoints,
thresholds and either side of thresholds. This is not yet an integrated public
pending-table API. The source rounds STFT thresholds to two decimals and uses
strict integer de-escalation in this table, which differs at exact boundaries
from the simulation's inclusive decision rule. The table ignores the simulation
run-in flag. These distinctions must remain visible, with correct unrounded
mathematics separate from explicit source-table compatibility.

The final report crosswalk still needs explicit scenario-level total enrollment,
toxicity-stop percentage, native single-MTD truth summaries, and waterfall
contour-relative allocation/selection summaries. Existing reports retain the
underlying arrays and per-dose summaries, but that does not close these named
outputs. The original waterfall defines its reference contour by each row's
nearest dose to target, requiring probability <= target+.05, then classifies
geometric regions. It is not simply `p == target` or `p > target`. Empty-contour,
transposed-grid and nonmonotone truth behavior require explicit tests/profile
choices. Original single-MTD metrics instead use exact `p.true == target`.

`MainForm.GenerateOCtable` RVA0x185bc additionally reports a conventional-design
duration approximation: ceil(mean enrollment / cohort size) times
((cohort size-.5)/monthly accrual + assessment window in months). Its displayed
label says CRM or mTPI although the help mentions CRM or non-TITE BOIN. This is
an approximation, not a simulated conventional calendar. IL capture is
`/tmp/boin-oc-table-il.txt`. Full protocol output must bind the methods/table and
operating characteristics to the same saved settings. Standalone single/TITE
final-selection reports need the available posterior means, marginal isotonic
intervals and overdose probabilities; selection alone is not the entire report.

Additional source report evidence is preserved for the next integration step:

- `single-reports.csv`:960 active standalone final-report executions. All means,
  weighted isotonic marginal limits and isotonic overdose probabilities agree
  with the installed independent core (native single-agent limits are rounded
  to two decimals; native TITE managed outputs retain unrounded limits). Maximum
  mean difference4.996e-16 and overdose-tail difference1.111e-15.960 installed
  assertions pass. These transforms are not simultaneous credible bands.
- `simulation-summaries.csv`:192 original complete-wrapper executions retain
  console and returned summaries separately. In the gradient single-MTD case,
  console correct-selection is0% while returned closest-target selection is100%.
  The managed single-MTD wrapper consumes console output; the returned fields
  must not silently substitute for that UI contract. The all-zero waterfall
  scenario selects a row-wise reference contour in the first column, so its
  geometric “above” category must not be mislabeled toxic overdosing.
-13 independent R fixture files regenerate byte-identically. Native originals,
  source text, console prose and executable resources are not redistributed.

## Accepted checkpoint evidence

**43,912 full-suite tests passed in879.81seconds**, with both extras and warnings
as errors. Production code was unchanged during that run. The481 adverse-history
and960 final-report numerical tests added while it ran passed separately against
the installed wheel: the full-run total does not include those1,441 new tests.

The fresh external wheel passed8,518 desktop tests in17.42seconds, plus those
1,441 additional tests,21 plot tests,28 image tests, seven retained review
regressions, and all four original defaults. The saved-default checks now include
single/TITE simulations, enabled comparators, reports and bilingual methods.
The installed quality run passed40,000 trials across all five original scenarios,
two independent500-trial batches, single-agent and all three TITE event-time
families; maximum selection MCSE was0.0156773. These are precision/invariant
checks, not clinical validation. The [manifest](boin-desktop-univariate-validation-2026-10-12.json)
and [quality record](boin-desktop-univariate-quality-2026-10-12.json) retain exact scope.
Lint/format(1,849 files), mypy(737 source files) and all13 R regeneration checks
passed. Final audited artifact hashes are stored in the preservation commit
message to avoid a self-referential sdist manifest.
