# STPLAN saved-study workflow boundary

The STPLAN methods manual enumerates the forward methods and describes inverse
planning as solving a parameter-specific power equation; the original program
also exposes per-method parameter ranges and directional branches. The Python
package already contains the 26 forward method registry entries and a bounded
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
completed case; the workflow does not claim a transaction or silently retry.

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
