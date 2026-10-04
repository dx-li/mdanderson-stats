# Report workflow completion audit

This batch closes advertised input and reporting workflows around existing
validated calculations. It adds no statistical model and does not claim exact
native terminal or document formatting. The same functional Python standard
already used for [BP1CI](../docs/bp1ci.md) applies: explicit arguments replace
menus, results retain effective inputs, and saved reports expose the calculations.

## TTEConduct, catalog 63

Sections 3–4 of the cached TTEConduct 2.0 guide (November 6, 2006) describe prior
parameter entry and a stopping table that echoes seven inputs. File Save writes
HTML; File Open displays saved output. The guide does not specify an editable
project format. [Source provenance and numerical references](../docs/tteconduct-source.md)
identify the inspected guide and independent R checks.

The existing design, monitoring, boundary-table and inverse-gamma prior-solver
APIs cover the calculations. [The report factory](../docs/tteconduct-report.md)
computes its table from the same validated design, records quadrature tolerance
and search cap, and writes a self-contained HTML snapshot. It distinguishes
zero boundaries from boundaries unresolved within the cap. The strict posterior
futility comparison and separate maximum-patient rule are stated explicitly.

Python retains continuous caller-unit roots and an explicit cap. Native rounded
days and the ten-year-per-patient ceiling are documented presentation conventions,
not additional probability calculations. Operating-characteristic simulation
belongs to the separate One Arm Time to Event Simulator entry.

## CID2BP, catalog 38

The cached `CID2BP_V1/source/all.f` describes repeated comparisons in lines
38–43, input/change-confidence/solve choices in `indata`, both success/failure
and success/trial input in `sample`, nine method choices, and cumulative result
output in lines 341–362. The redirected `TestIn`/`TestOut` witnesses are interactive
reference runs, not a specified structured batch-file format.
[Source records](../docs/cid2bp-sources.json) preserve provenance.

[The session API](../docs/cid2bp-session.md) normalizes both count-entry modes,
retains ordered comparisons and repeated methods, and records each confidence
level, rate estimate, difference and interval. Both requested `auto` and its
resolved method remain visible. Whole-session structural, count and resource
checks precede serial interval calculations, including the existing Peskun grid
limit. Reports validate their precision setting before opening an output file.
The nine numerical algorithms and their documented corrections are unchanged.

## CONFINT, catalog 64: source crosswalk

The cached `CONFINT/confint-2.0/SOURCE/confint.f90` main menu names eight families.
Each `do_*_mod.f90` module echoes inputs and appends answers to the report file
selected at startup. The source defines a repeated calculation log, not editable
session restoration. [The method guide](../docs/confint.md) and its linked source
records document the existing calculations and intentional numerical corrections.

| Source family or properties table | Existing calculation coverage |
| --- | --- |
| Normal mean, normal SD, pooled two-mean difference | Normal assurance, sample size and SD limits; repeated assurance values form the source SD-limit tables |
| One binomial | Width assurance, event-probability limits, attainable length and sample size; median length uses assurance 0.5 |
| Two binomial proportions | Wald-width assurance, balanced sample size and event-probability limits; exchanging groups supplies the other held-fixed limit table |
| Poisson | Width assurance, rate limit, length and earliest exposure; repeated assurance values form properties tables |
| Exponential hazard and mean-survival intervals | Mixture assurance, bracketed length/design inversions and hazard ranges; repeated follow-up values form accrual-duration tables |

No additional mathematical menu option or properties-table statistic was found
in these source routines. Reports must preserve the distinction between total
interval width, confidence and assurance; population SD and standard error;
pooled mean differences and Wald binomial differences; and hazard versus mean
survival. Survival outputs must retain omitted-mass, search-bound and clipping
diagnostics rather than reducing every answer to a single unlabeled number.

The [immutable calculation log](../docs/confint-session.md) covers all 18 named
APIs. Each call returns a new log and the original numerical result; later
changes to a returned array cannot alter the stored report. Default settings,
interval targets and complete result fields are retained. Requests are scalar
apart from explicit bound pairs and a fixed-event table of at most 129 counts.
Logs have at most 100 calculations and two megabytes of rendered text. Existing
kernel limits remain in force, and a report is rendered before atomic saving.

With these input/output workflows completed, entries 38, 63 and 64 move to
implemented. Their documented numerical corrections and native-format
differences remain explicit. This does not change the status of other programs.

## Validation

The integrated TTEConduct, CID2BP and CONFINT changes, together with the
[ASYPOW vector inversion](asypow-vector-inversion-audit.md), passed 38 focused
tests, including existing numerical-reference cases, with warnings treated as
errors. Scoped Ruff and mypy checks passed. The sequential validation batch took
6.386 seconds, peaked at 173.31 MiB child-process RSS and recorded zero process swaps.
These measurements describe this local workload, not whole-system memory usage.

The full local suite was not run for this workflow batch. No dependencies or
CI configuration were added. The original kernels retain their independent
mathematical and native-reference validation described in their method guides.
