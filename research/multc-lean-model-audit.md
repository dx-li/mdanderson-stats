# Multc Lean managed model and functional workflow audit

On October 9, 2026, inspection of the locally recovered official Multc Lean 2.1
assembly and installed configuration resolved model conversion, reset defaults,
saved input order and the desktop's exact/Monte Carlo composition. The Python
native-model study workflow now completes catalog entry #12. The bundled
snapshot is **91 implemented / 41 partial / 6 pending**, not full-site coverage.

## Source and executable evidence

The original installer archive and numerical DLL hashes are recorded in the
[duration source record](../docs/multc-legacy-duration-source.json). Additional
research inputs are `MultcLeanDesktop.exe` (SHA-256
`a9d8b5b8e3e0f6d2d30e93c550cda8253c00b0777dd6e5a3a77322eba553c69d`)
and its installed `.config` (SHA-256
`4c6fdc133739a4dd935212df4392e3194e13bd0eb7e56a88a50ec82ce1b4c58d`).
No original executable/configuration/template or inspection runtime is shipped.
The original license restricts redistribution of the original program.

`tools/reference_multc_lean_model.py` checksum-verifies those files and uses
research-only `dnfile==0.18.0` and `dncil==1.0.2` to decode managed CIL bodies.
An authored interpreter executes original instruction order, stack/local
operations, branches, indexing and field assignments, limited to 100,000
instructions per call. Unknown instructions/services fail explicitly.

| Original method RVA | Contract inspected or executed |
| --- | --- |
| `.ctor` `0x2508` | Inspected adjacent constant/field assignment: repetition default 10,000; minimum/cohort 1 |
| `InitializeFormVariables` `0x8268` | Inspected initialization/restoration path |
| `RestoreModelFromFile` `0x863c` | Executed original parser/field-selection control |
| `SaveModelToFile` `0x8ad4` | Executed original writer/order control |
| `ResetModelParametersInControl` `0xa860` | Executed original config-to-field/control selection |
| `RunScenarios` `0xd378` | Inspected exact-first, optional-simulation composition and timing guard |
| `RunDependentScenario` `0xd92c` / `RunDependentSimulation` `0xdaa0` | Inspected managed/native parameter field transfers |
| `SaveAs_HTML` `0xef88` | Inspected input/results report workflow |

The fixture contains eight **synthetic** profiles, original-control-produced
model text and restored fields, with zero through three scenarios, beta/fixed
histories, fractional shapes/margins and alternate minimum/cohort settings.
Nine Python exports, including a 17-digit experimental shape, additionally
pass through the original restore control. Reference generation never imports
the production codec. The optional codec check is separate from generation.

CLR objects, file streams, clock/version strings, string split/concatenation,
number conversion/formatting and UI controls are substituted services. They are
not original CLR execution. Writer formatting uses an external 15-digit general
formatter; timestamps are controlled. Python export uses 17 digits. References
verify schema, order, branch/field control and default selection, **not**
byte-identical .NET formatting, GUI exceptions, OS integration, application
execution or native floating-point parsing. Exception handlers are not interpreted;
these references cover valid-input normal paths. The original numerical DLL is
not called by the managed interpreter.

## Recovered input and runner contracts

A file starts with `%` comment headers, two six-field endpoint rows, seven
controls, a scenario count and six fields per scenario. The response/toxicity
rows retain historical beta shapes and a fixed constant even when inactive;
the flag selects which historical model is active. Scenario categories are
both, response only, toxicity only, neither, followed by mean interarrival and
response window. The source writer emits single ASCII spaces; its numeric rows
use LF and header/count `WriteLine` uses CRLF. Names, seeds, repetitions and
result data are absent. The Python reader documents the canonical writer profile,
accepts BOM/CR/LF/CRLF and rejects extra data, negative counts and malformed input.
It raises rather than performing the desktop's error-triggered default reset.

Reset/config response values are historical `(30,70)`, experimental `(0.6,1.4)`,
inactive constant 0.3; toxicity values are `(25,75)`, `(0.5,1.5)`, inactive constant
0.25. Both historical models are beta, both cutoffs 0.95, margins zero, cap 30,
minimum/cohort 1. Potential-boundary output is off by default. The published
tutorial's toxicity history `(20,60)` is a different example. The constructor
sets 10,000 repetitions independently of the file schema.

Each timing input can be zero through 10,000. Simulation runs only when **both**
are positive. `RunScenarios` always obtains exact dependent-scenario OCs first;
when enabled it copies **only average duration and average balk count** from
the simulation into those results. Average enrollment/response/toxicity and
sample-size probabilities remain exact. The Python report follows that
scientific separation, while additionally exposing complete simulation summaries
and MCSEs. Explicit captured seeds replace the source's seed-zero clock default.
The native UI sum tolerance is 1e-10; Python uses the existing stricter 1e-12
core tolerance and rejects rather than silently altering probabilities.

## Functional coverage review

The user guide's model, stopping-boundary, scenario-input, scenario-output and
miscellaneous sections define the following advertised functions. Completion
requires the usable composed workflow, not only the recovered file schema.

| Advertised function | Python coverage and independent evidence |
| --- | --- |
| Fixed/beta history, experimental beta priors, margins and cutoffs | `MultcLeanEndpointInput` and `to_design`; tutorial and independent shifted-beta R integration |
| Continuous/cohort monitoring, minimum and prior screen | Monitoring APIs; original compact-boundary/control references establish mandatory prior precedence and strict ties |
| Full/optional potential boundaries | `stopping_bounds`, `potential_boundaries`, protocol and optional report table; tutorial and R/reference control checks |
| Associated/independent scenario entry | Owned four-category `MultcLeanScenario` or explicit marginal-independent constructor; imported models checked against full R `4^6` path enumeration |
| Exact enrollment/outcome means and stopping PMF/CDF | Original mathematical recursion independently validated against R paths; exact results retained when timing is enabled |
| Optional average duration and balks | Recovered x86 duration kernel, boundary conversion and original study aggregation; controlled draws, PCG64 batch and replay |
| Model save/restore and installed example | Native schema/default selection verified through managed control; explicit atomic read/write APIs replace automatic GUI-exit persistence |
| Saved scenario output | Atomic HTML with actual inputs, bounds, exact results, optional duration/balk MCSEs and seeds; JSON records replay inputs |
| Cumulative/density stopping plots and PNG export | Exact `plot(cumulative=True/False)` Figure with normal `savefig`; tests distinguish CDF from PMF after early stop |
| Editable statistical protocol example | Authored Markdown using actual priors/rules/cohort bounds and captured inputs; no original Word template redistributed |
| Parameter Solver / Inequality Calculator tools | Existing validated `solve_distribution_moments`, `solve_distribution_quantiles`, `compare_beta_difference` APIs |

All identified advertised calculations and research workflows are implemented.
Entry #12 is promoted from partial under the repository's functional coverage
definition. The general observation-aware calendar study is also retained with
its explicit separate timing policy; it is not silently changed to latent-count
legacy behavior.

The independently implemented posterior integrator remains validated against
R and published tutorial boundaries rather than the original DLL integrator.
Substituted native predicates establish control, not integrator parity. Original
CLR/GUI/clock RNG, automatic desktop lifecycle, clipboard/menu interactions,
Word-template layout, timestamps/report bytes and random streams are explicit
compatibility differences. Python adds strict errors, bounded work/storage,
immutable inputs, Monte Carlo errors, captured seeds and a proper point mass at
zero on prior rejection (the native wrapper leaves that PMF empty). Those choices
must accompany reproducibility claims. No full Windows application or original
random-stream execution is claimed.

## Validation and reproduction

The 65 added workflow tests cover source model order/defaults, inactive history,
17-digit preservation, native endings/BOM, independent R OCs after import,
all four optional-timing combinations, exact/MC separation, default 10,000
replicates on prior rejection, study/trial seed replay, HTML escaping, actual
CDF/PMF plotted values, atomic files, malformed text/JSON, float64 representability and aggregate checks
before RNG creation. Numerical and x86 references remain those in the linked
[duration](multc-native-duration-audit.md) and
[boundary/aggregation](multc-native-boundaries-audit.md) audits. Complete regression,
packaging and installed-guide results are recorded in the
[continuation checkpoint](source-recovery-2026-10-09.md).

With research-only packages installed separately from runtime dependencies:

```bash
PYTHONPATH=/workspace/.tools/source-inspection/python \
  .venv/bin/python tools/reference_multc_lean_model.py \
  --exe research/raw/source-recovery-2026-10-08/multc-lean-native/MultcLeanDesktop.exe \
  --config research/raw/source-recovery-2026-10-08/multc-lean-native/MultcLeanDesktop.exe.config \
  --check --check-codec
uv run --locked --extra plot pytest tests/test_multc_lean_model.py -q -W error
```
