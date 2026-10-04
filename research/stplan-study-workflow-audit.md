# STPLAN saved-study workflow boundary

The STPLAN methods manual enumerates the forward methods and describes inverse
planning as solving a parameter-specific power equation; the original program
also exposes per-method parameter ranges and directional branches. The Python
package already contains 25 active forward procedures plus the archived
matched-pairs option in its 26-entry registry, and a bounded
`stplan_solve` implementation. What was missing was an application-facing way
to save several named calculations with their actual fixed inputs and replay
the reported result.

`STPLANStudySpecification` adds named scalar forward-power cases and bounded
inverse cases over those existing APIs. It records effective defaulted forward
inputs, target and achieved power, computed values, explicit bounds, allocation
weights, integer/search controls, and caller-selected solver limits. The schema
rejects unsupported methods and fields. Group vectors are capped at 100 values;
studies are capped at 20 cases and 1 MiB; combined inverse evaluations and
conservative estimated terms are budgeted across the study before execution.
Structural, schema, and resource preflight occurs before the first calculation.
Method-specific scientific domains are still checked by the existing numerical
functions during execution, so a later domain failure can follow an earlier
completed case. A failed run raises without returning a combined result; it does
not retry failed calculations.

The report is a Python community convention. No native save/session schema or
HTML report contract was established. Bounds and root branch remain explicit
caller choices. The study does not add power calculations, infer automatic
native bounds, or allocate fractional K-group totals to integers. The cached
source coverage boundary is documented separately in
[`stplan-coverage-boundary-audit.md`](stplan-coverage-boundary-audit.md).

Other source-defined operations that do not fit this small saved-case contract
remain available via their existing interfaces: exact critical-region
significance planning, joint historical-control allocation, and matched-pairs
no-pilot initial-size estimates.

## Validation

Three focused workflow checks cover deterministic JSON replay, immutable input
snapshots, invalid later schemas before solving, escaped HTML and atomic output.
Four independent R fixtures validate continuous normal sample size, normal
significance, exact-binomial integer attainment and fractional K-group allocation
(`tests/fixtures/stplan-planning-r.csv`). The significance case also guards
against accidentally restoring a computed parameter's default into fixed inputs.
Integer results retain the adjacent candidate and its power.

Root integration passed the three checks, Ruff, formatting and mypy in 2.747
seconds, with 155.31 MiB peak child-process RSS and zero swaps. The full local
suite was not run; the underlying planning formulas were unchanged.
