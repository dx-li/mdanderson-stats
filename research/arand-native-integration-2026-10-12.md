# ARAND native-profile integration checkpoint

Recovered remote parent: `e74b1ef82904a8d0587442abd17d53242e0cf183` on
`codex/community-completion-2026-10-12`. No reset or forced update occurred.
No AGENTS.md or repository `.agents/skills` were present in this environment.
Repository contribution instructions and the continuation/roadmap/license records
were read before implementation.

## Scope closed

The [native workflow guide](../docs/arand-native-workflow.md) documents the new
single-endpoint supplied-calendar controller and the complete explicitly defined
Python study workflow: validated inputs → controller/posterior replay → fixed-delay
binary or censored exponential scenario generation → repeated simulation →
selection/enrollment/duration summary → saved JSON and self-contained HTML report.
All-arm ranking reuses the independently tested beta/inverse-gamma mathematics.
Original-order monitoring, status transitions, trial/arm limits, proportional
allocation floors, group and whole selection remain separate from the preexisting
required-policy controller. Native behaviors that can stop pending trials, reopen
closed arms or allocate during forced run-in are explicit compatibility rules.

Production additions: `arand_native.py`, `arand_native_trial.py`,
`arand_native_study.py`, with root exports. No existing statistical implementation
was replaced. Original artifacts remain ignored and excluded from distributions.
The new notice documents source-informed authorship and does not assert a new
permission, clean-room status, unrestricted bundle or clinical validation.

## Independent evidence

- Official archive downloaded directly; SHA-256 still
  `ff9c4d490a8876e1b3197d6fcb809bd8d001761be13f083a667e0e7db34fc877`.
  Locally extracted library hash matches the existing reference harness.
- All **98** original-instruction fixtures across **13** families regenerate with
  identical parsed JSON values. This is fixture durability, not 98 full Python
  integrations. Dormant predictive and latent multi-endpoint/group paths remain
  outside the advertised single-endpoint implementation.
- All six original integrated controller outcomes agree with Python: statuses,
  stopping/allocation, analysis times, completion time and final winner. Component
  fixtures check floors, monitoring thresholds, observations/censoring, status
  transitions, constraints and both final selectors.
- `tools/arand_native_reference/verify_python.py` executes **900** deterministic
  original-instruction comparisons: 300 status transitions plus 600 whole/group
  selectors. Random seed 123 covers exact cutoffs, permanent closures, action
  overrides and deliberately inconsistent supplied posterior scores. This checks
  controller logic, not posterior inference.
- Exact beta polynomial integration gives 5/6 after a success/failure history;
  independent reciprocal-Gamma ordering gives 11/16 for a censor/event history.
  Seeded mean/median trials agree. A 512-replicate one-arm simulation agrees within
  six analytical MCSEs with Poisson(.5) enrollment and .75-month expected stopping
  time, including exponential candidate overshoot beyond the .25-month horizon.

## Additional report-source recovery

Original `PrintSimulationSummary` at RVA0x60a14 contains a duration-vector guard
at IL offsets927–949: percentile reads execute only for at least 100 values.
The zero-based indices then use `int(ceil(n*p - 1) + 1)` for p=.025/.975, not
NumPy type7. A new bounded original-IL slice runner preserves 12 fixtures in
`fixtures/arand-native/report-index-supplement.jsonl`. It supplies sorted vector
memory and Math.Ceiling, executes the original branch/arithmetic/read instructions,
and fails on out-of-range reads. Below 100 it records no read and untouched zero
locals; it does **not** claim what the complete GUI renders in those cases.
At 100 the indices are 3 and 98; at 1000 they are 25 and 975. Enrollment-array sections
also contain the same arithmetic, but their complete output branches have not
been executed by this new slice probe. The existing98-fixture manifest remains
unchanged; these12 probes are a separately scoped supplement.

`AccumulateSimulationStatistics` RVA0x62894 records final status per group,
selected count and early-selected count. `SimulationDirector.SetSeed` RVA0x2c60
passes base seed plus scenario index through the scenario vtable. This is a
source-path observation, not a completed source-reference fixture for report
aggregation or wrapper RNG. Do not relabel the current Python streams or report
as original-identical on this evidence alone.

## Precise remaining application contracts

1. Original managed saved-study XML mapping, including explicit versus omitted
   values and enabled/disabled constraint flags. Current Python rules enable
   their declared limits and expose explicit defaults; they are not a general
   native-XML loader. Original ARand.Design.dll includes separate booleans for
   trial/arm/group maxima/minima that must be mapped without guessing.
2. Native exponential-batch construction versus reset (ceil(a) versus int(a)),
   including initialization and scenario reset. Current supplied monitor states
   reproduce transitions without assuming how an omitted initial state was built.
3. End-to-end original report aggregation, final-status/drop labels, sorted
   percentile selection, small-replicate output branches, and report dispatch
   through the exposed noncovariate wrapper. Bounded percentile evidence above
   narrows this contract but does not close it.
4. Wrapper scenario and replicate seeding/reset policy. NumPy streams are an
   explicitly supported independent alternative; no original random-sequence or
   native posterior-integrator identity is claimed or required for that profile.
5. Original persisted nonempty patient histories and GUI file round trips remain
   distinct from Python JSON input round trips.

Coverage stays **93 implemented /39 partial /6 pending**. ARAND #62 stays partial.
No approximate model was supplied for the six externally blocked applications.
The mixed-license distribution decision and previous PR Forbidden remain with
the parent; no bypass, merge, registry release, deployment, settings change or
third-party contact was attempted. No verified task URL was exposed.

## Validation

Final-code full regression and artifact evidence are appended below when complete.

### Missing XML closure-action default narrowed

Inspection of original constructor RVA0x19704 shows status cells at offsets 116
and 120 initialized to2. XMLPopulate RVA0x6e05c reads `min_prob_action` and
`max_prob_action` with empty-string defaults, then calls SetAction. The original
SetAction method at RVA0x6f5a4 maps case-sensitive `open`→0, `tmp_close`→1,
`close_early`→2 and `close`→3; empty or unrecognized strings retain the cell's
existing value. A second bounded original-IL runner,
`arand_action_reference.py`, preserves eight probes in
`fixtures/arand-native/action-supplement.jsonl`, including missing, unknown and
uppercase strings. Thus a fresh native stopping-rule object's absent action
attributes retain 2; a reused object's value is state-dependent. This resolves
that narrow missing-action question, not the full saved-study loader, enabled
flags, desktop serialization/default precedence or nonempty patient histories.
The current explicit Python profile continues to use its documented supported
actions and does not silently assume those XML defaults.

A further 300 deterministic original-instruction comparisons (seed 314,
`verify_constraints.py`) agree on ordered arm/group constraints, including
conflicting maxima/minima, closed-arm reopening and exact start/duration
boundaries. Total supplementary differential comparisons: **1,200**.

### Allocation-population review correction

Review of original AdaptiveRandomizer.GetTreatment RVA0x6bcbc confirms that an
ordinary allocation samples the eligible-arm map, the single-eligible-arm branch
selects that arm directly, and ForceEqualNumberOfPatients RVA0x6c1ac visits all
arms. Replay now retains `assignment_arms` separately from its full-length
allocation display vector. An exactly zero supplied uniform cannot select an
excluded arm in ordinary allocation, but can select a zero-deficit first arm in
the forced population under the original `<=` rule. Both cases have regressions.
The first full-suite attempt was deliberately interrupted for this correction
and is not reported as a pass. A new final-code run supersedes it.

Seven additional original GetTreat instruction probes now confirm the `<=`
inverse-CDF boundary itself, with supplied uniform/vector services and no RNG
identity claim. Exact0 with weights(0,.5,.5) chooses index0; the next representable
positive uniform chooses index1. The caller's eligibility/population dispatch is
established by the separate source path above. All 27 supplementary action,
percentile and assignment probes are hash-recorded in
`fixtures/arand-native/supplement-manifest.json`, separate from the original 98.

### Interim executor status requested by parent

Environment ready; exact e74b1ef parent recovered. Implementation preservation
commit `3732fde7c6d4b69653da379509aa862722cd1e03` is pushed and remote-verified.
No executor, repository-access or tool blocker prevents continued work. Corrected
code passes 89 affected tests and 120 isolated-wheel tests including plot/image
extras; the guide example, lint, types, build and provenance checks pass. The
restarted full regression is still running and is not yet counted as a pass.
Do not duplicate ARAND implementation while that verification is active. The
final validation section below will supersede this interim status.

## Final validation and preserved outcome

The corrected, frozen production code and test set passed **34,179 tests in
818.71 seconds**, with warnings treated as errors, under Python 3.13.5. This
supersedes the interrupted earlier attempt. The complete ARAND affected set
passed **89 tests in 3.66 seconds**. No statistical implementation or test changed
after the successful full run began; subsequent edits were research evidence,
provenance and continuation documentation.

A fresh isolated wheel environment outside the checkout with both `[plot,image]`
extras passed **120 tests in 4.93 seconds**: the 89 affected ARAND tests plus 31
existing BLiP/plot and independent TIFF-reference tests. The installed native
workflow guide example completed, including simulation, seed replay, JSON
round trip and HTML report. Imports resolve to the isolated site-packages tree.
Installed versions: NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, Pillow 12.3.0.
This is affected-workflow/example validation, not an installed example audit of
all 138 applications.

Ruff lint and formatting pass; mypy passes for 707 source files. Wheel and sdist
build successfully. All 711 packaged Python/JSON source/data files match the
checkout bytes; licenses/notices are present, and no original EXE/DLL/MSI/ZIP or
raw research assets are included. All 27 new root exports exist exactly once.
Changed-guide links and all 138 catalog/status IDs agree. The scoped provenance
manifest records source hashes and relative imports. No explicitly restricted
seed module appears in the new modules' relative-import closure, but that audit
excludes eager root/external/dynamic imports and is not legal clearance; the
complete package retains its mixed-license restrictions.

The original 98 fixtures regenerate identically, the 27 supplementary probes are
hash recorded, and 1,200 original-instruction differential comparisons pass.
Original posterior/RNG/whole-GUI identity and clinical validity remain separate,
unclaimed properties. No coverage status was promoted. No head-specific hosted
CI is claimed; no PR permission-denial workaround was attempted.

Implementation is preserved in `3732fde7c6d4b69653da379509aa862722cd1e03`, with
additional assignment evidence in `78653d44`. Both were pushed normally to the
existing feature branch; the final validation/provenance commit follows them.
The parent can continue from the current remote head using the five precise
remaining ARAND contracts above. No executor/tool blocker remains, and no
verified task URL was exposed by this environment.

## Continuing native desktop integration (after 37a1f7d)

This is an intermediate preservation checkpoint; application #62 is still partial.
The independent constraint gates now control trial/arm/group minima/maxima and
trial duration. The original native paths passed 1,000 randomized ordered-limit
comparisons and 500 randomized minimum-gate comparisons. Portable fixtures cover
all 256 combinations of eight flags. Scalar defaults preserve existing explicit
Python rules; omitted native XML flags must still be mapped to false by the
future native XML reader.

Per-arm lower/upper/final/futility probability cutoffs now flow through the
controller and Python JSON studies. Original per-arm native instruction paths
passed 1,800 status/whole/group-selector comparisons; 60 portable cases are
retained. All four native action codes (including open) are explicit. A zero
per-arm futility cutoff disables that arm's futility check; per-arm *parameter*
thresholds still require posterior/study integration.

`arand_native_batch_monitor` reproduces original construction and replication
reset. Six original-instruction fixtures establish `ceil(a)` versus `int(a)`
initial thresholds, with the same precomputed exponential increments. The reset
option is explicit; empty schedules, zero increments, overflow and excessive
bounds fail safely instead of inheriting undefined native operations.

An isolated, locally extracted Mono runtime successfully executed the original
managed assemblies, without installing system packages or running original
installers. `tools/arand_native_reference/managed_studies.py` hash-pins all three
original assemblies, compiles a synthetic-input probe, checks original IsValid,
and executes original Serialize/Deserialize round trips for binary and TTE
studies with histories. Four XML outputs and their manifest are retained under
`research/fixtures/arand-native/managed-studies`. Only wall-clock current_time is
normalized. The managed getters/validator rewrite history metadata, outcome time
and group capitalization; these fixtures are source evidence, not yet a Python
native-file importer or proof of valid clinical histories. The original mixed-mode
native engine has not been loaded into Mono, and no such identity is claimed.

Targeted portable contract tests: 328 passed. Existing 74 native workflow tests
passed before the batch helper; final combined/artifact/regression validation
will follow. Remaining work: native XML mapping and safe history semantics,
per-arm parameter thresholds/delays, wrapper scenario reset/seeding and full
native report aggregation/dispatch. Batch construction and enable-flag primitive
contracts above are closed; the complete app is not promoted.

### Original simulation XML and further report recovery

The strict `read_arand_native_xml` reader maps the original managed Binary/TTE
single-group simulation schema into runnable studies, preserving the entire
source text and a `has_patient_history` flag. It supports per-arm probability and
parameter thresholds, binary response delays and enabled limit mapping. Only
explicit Months units are accepted. Simulation starts fresh; this is **not** a
clinical-history recommendation/import/editor. The four managed serializer
fixtures round-trip through the Python document and run binary/TTE simulations.
A separate independent beta/inverse-gamma CDF check validates vector thresholds.

The original metadata truncates long instantiated method names. The base
noncovariate report candidate is RVA0x57b68, reached through base constructor
0x17b48's vtable field0x040006f6. Its arithmetic/control layout matches the earlier
covariate report, apart from class offsets and metadata token substitutions.
However, wrapper XMLPopulate0x23fa8/IL410 **replaces** that vtable with field
0x04000524. Its slot12 is native x86 **0x18d3e0**, not the base report candidate.
Therefore the 12 base-report percentile probes do not establish the advertised
wrapper's final report. `verify_report_dispatch.py` now records the complete
base/derived boundary; full derived report execution remains open. An interim
name-based interpretation was corrected before committing this evidence.

Accumulation slot28 is shared. `arand_accumulate_reference.py` executes the full
original noncovariate accumulation body at0x59ac0 using final-record services and
counter sinks; 12 fixtures cover all status families, absent/present winners,
early selection and completion. It records per-arm enrollment/final status,
selected/selected-early counters, duration moments and whether any final status
is1 or2. Integrating those fields into a separately named native-report profile
remains next work. Existing Python reports keep their declared conventions.

Native history investigation: original managed patient setters/validation emit
zero outcome times in the synthetic histories. A pending TTE query with arrival7,
record time0 and now8 reaches an unsupported native path in the bounded x86
harness (unmapped fetch), so no negative-exposure or exception-result fixture is
claimed. The importer retains those records without silently using them as
clinical data. Resolve original history setter/getter/date conversion and that
native path before enabling history recommendations.

Current checks: lint/format, mypy708 modules, wheel/sdist build, 712 packaged
Python/JSON files byte-identical to checkout, notices retained and original
binary/raw exclusions pass. Fresh installed wheel/extras checks:464 tests pass;
installed guide plus binary/TTE XML simulation/report examples pass. Initial
isolated-test attempts lacked copied test fixtures/helpers, and the old example
asserted its previous environment path; these harness issues were corrected.
Full final-code regression is running. Two earlier follow-on regressions were
stopped because implementation was still changing; neither is counted as a pass.
No application promotion, release, merge, PR bypass, or license change.
