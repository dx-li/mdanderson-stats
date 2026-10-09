# October 8 source recovery and continuation record

The [October 9 continuation](source-recovery-2026-10-09.md) now resolves the
compact-boundary lead below and records a fresh 33,726-test full-suite pass.
Earlier validation counts and open leads on this page describe their original
checkpoints; the linked continuation is the current state.

The prepared environment uses pinned Python 3.13, locked uv dependencies and
the plotting extra. Reusable setup/start instructions and research domains
are saved in the environment draft. Publication is a separate user action;
saving does not establish runtime access to the newly listed publication hosts.

## Catalog scope

The bundled catalog has **138 entries: 90 implemented, 42 partial, 6 pending**.
The desktop/online `GetAll?IsOnline=false/true` endpoints and catalog detail/
download-metadata routes returned HTTP 500. Direct archives and PDFs still
worked. Counts describe the bundled September snapshot with audited
implementation updates; a fresh complete site inventory is not established.
Full functional Python coverage remains the goal, not a completed claim.

## Implemented or extended in this continuation

| Program | Result | Evidence and remaining scope |
| --- | --- | --- |
| FLECS90 #43 | Complete translator/file/CLI workflows; moved to implemented. | 144 C outputs and compiled Fortran references; [audit](flecs90-audit.md). |
| Multc99 #3 | Complete general outcomes, mixtures, calibration, single/randomized trials, boundary editing, curves and report replay; moved to implemented. | C/high-precision references, native defects corrected and upstream terms retained; [audit](multc99-audit.md). |
| iBOIN #145 | HTML/JSON reports and exact-seed replay. | Final isotonic weights/ties/defaults still unresolved; [audit](iboin-report-audit.md). |
| RMC-COMPASS #174 | Approximate displayed-model predictions, risk groups and parameter-rounding envelopes; moved to partial. | 55 fresh probabilities and 11 medians/groups. Exact fit/uncertainty/raw-input contract open; [audit](rmc-compass-audit.md). |
| WFMM #70 | Native inverse-gamma mapping and energy compression with actual-energy diagnostics. | Eight prior cases, five full Haar transforms and 46 native compression/bypass/error runs. MOM/profile estimator, proposals, other extended transforms/pass filtering/files open; [prior audit](wfmm-native-prior-audit.md); [compression audit](wfmm-native-compression-audit.md). |
| Multc Lean #12 | Native duration compatibility replay with explicit compact bounds and external variates. | 34 original x86 instruction references resolve clipping, shared follow-up, balks and latent-count scheduling. Native boundary conversion/full workflow still open; [audit](multc-native-duration-audit.md). |
| SYNERGY #18 | Four parametric models with fitting, inference and figures. | 138 original-kernel predictions and nine R fits. Separate semiparametric bootstrap interval still open; [audit](synergy-parametric-audit.md). |

The full warnings-as-errors suite passed **33,541 tests** in 819.41 seconds
after repairing older reference tests' unclosed CSV handles. The subsequently
added SYNERGY tests passed all **36** new method/figure checks; the 80-test
combined SYNERGY/WFMM-prior check also passed. No tests were skipped.
Final packaging checks passed: Ruff, formatting (2,376 files), mypy (689
source files), wheel/sdist builds, and all 15 Python examples in the six
continuation guides from an isolated wheel installation. All 689 Python
modules, catalog and notices match the wheel byte for byte; native research
archives and local runtimes are excluded. The complete 423-test new-workflow
run passed in 12.10 seconds, and final collection contains 33,577 tests.
Matplotlib's writable configuration directory is saved as
`MPLCONFIGDIR=/workspace/.cache/matplotlib`, avoiding the read-only home cache.

## Further recovered native bundles

Original archives, executables/libraries, parameter tables and example data
remain ignored local research inputs and are excluded from distributions.

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| [Multc Lean](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcLean_V2.1.0_WithFX4.0.zip) | 63,625,138 | `be4ddfcaa7f442a1fa6895e31e379146fda4cff23775df291ff1b597b10d5120` |
| [bCRM](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/bCRM/bCRM_V1.1.3.zip) | 18,691,148 | `a05cb3358711d85a59baf7924ae339f76d782697938e09493e0134adbee8af55` |
| [PerfectMatch](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/PerfectMatch/PerfectMatchv2.1.zip) | 23,858,118 | `e688fce76c0a9b938aa3d6041f3e27e0d15e141742d3542e15cd1e6d19218b91` |

Multc Lean's MSI supplies `MultcLeanCalcs.dll`, the guide/tutorial and
`License.rtf`. The calculation DLL is a mixed native/managed Windows x86
assembly; numerical methods are native machine code, not portable CIL.
Metadata locates `RunResponseAndToxicityScenario` at RVA `0x7ee0`,
`RunDependentResponseToxicityScenario` at `0x8540`, and
`RunDependentDurationSimulation` at `0x75b0`, image base `0x10000000`.
The single-trial duration kernel at RVA `0x64d0` is now verified through
bounded x86 emulation with external variates/vector services substituted. It
establishes clipping, shared follow-up, balked arrivals and latent-count
scheduling. Full Windows execution and RNG parity are not claimed. The next
compact-boundary control routine is located at RVA `0x7480`; it still needs
executable references before posterior/cohort conversion is claimed.
The bundled license prohibits redistribution of the original program.

bCRM contains a Windows x86 executable, example installer and Java runtime
installer, with no mathematical source listing. Recovering its binary does
not resolve the joint likelihood/prior, transition precedence, positive-gap
futility rule or final logistic estimator.

PerfectMatch supplies its Windows executable, compiled/PDF/Word help, chip
annotation/probe tables and three energy files (HG-U95Av2/HG-U133A/MG-U74Av2).
Each energy file has two ordered 16-dinucleotide blocks, two ordered
24-position blocks and two scalar tails. Exact role mapping in the native
reader still needs confirmation before a compatibility API is claimed.
Recovered help says `ScalingFactor` and `Absent genes` are improperly computed
and should be ignored. It still does not define exact `err_T`, cross-PM
aggregation or the affinity subset rule. These statistics must not be
fabricated; see [the guide](../docs/perfectmatch.md).

## Six pending calculators

| Program | Recovered evidence | Essential missing evidence |
| --- | --- | --- |
| ComPAS #140 | Public app shell and official flowchart. | Hierarchical likelihood/model-selection priors, borrowing and full conduct rules; an independent-arm model is insufficient. |
| Bayesian Survival #146 | Public input/output interface. | Fitted/posterior objects and conditional prediction/uncertainty contract. |
| BLESS #149 | Official instruction and variable PDFs now downloaded. | General/disease-specific Cox coefficients/interactions and reference baseline survival. |
| CNSRISK #161 | Public interface and reduced-model publication lead. | Exact reduced competing-risk coefficients and baseline cumulative incidence. |
| KCOMPASS #169 | Public interface and linked 2026 paper. | Exact Weibull shape/intercept/coding and uncertainty. |
| MDSDPSS #170 | Public dynamic-score interface. | Exact longitudinal/dynamic fit and prediction/uncertainty rules. |

BLESS [instructions](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BLESS/Instructions_for_BLESS_online_calculator.pdf)
and [variable definitions](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BLESS/BLESS_Variable_definitions.pdf)
specify days, consistent NLR units, fluid LDH/protein units and clinical
indicators, but supply no fit or baseline curves. Publication-domain proxy
denials and prior supplement failures are recorded in
[remaining source leads](remaining-source-leads.md). The PMC article itself is now readable, but its directly linked appendix
returns download/browser challenges, including in the trusted HTTPS browser.
Elsevier and Europe PMC requests still returned HTTP 403 in the recheck.
No replacement clinical constants are inferred.

## Next evidence-driven work

The [42-entry priority review](statistical-coverage-priority-2026-10-04.md)
lists remaining scientific contracts. Start with recovered native bundles or
primary supplements instead of repeating blocked reader routes. Portable
file/report improvements can be useful, but do not promote an entry by
substituting guessed coefficients, baselines, priors, calibration or stopping
precedence. The remaining clinical models require exact fitted evidence.

## Further continuation checks

The new WFMM compression and Multc Lean legacy-duration workflows pass **149**
focused tests (79 and 70 respectively) with warnings treated as errors. The
original-program reference generators independently regenerate all 46 WFMM
runs and 34 controlled-variate x86 duration cases with `--check`. The preceding
33,541-test full run plus the 36 SYNERGY checks remain the full baseline; these
149 are additional checks, not a claim of another whole-suite run. Coverage
counts remain 90 implemented / 42 partial / 6 pending, with the newly resolved
subcontracts and remaining scope recorded explicitly.

The affected Multc/WFMM regression run passed all **375 tests** in 12.95 seconds
with warnings as errors. Ruff, formatting (2,385 files), mypy (691 source
files), wheel/sdist builds and installed-wheel validation pass. All 691
Python modules, catalog and notices match the wheel byte for byte; all nine
Python examples in the updated WFMM and legacy-duration guides execute from
the isolated wheel. Complete collection now contains **33,726 tests**.

Further Multc Lean source location: RVA `0x7480` constructs a compact boundary
vector and calls adverse-probability predicate RVA `0x6340`; wrapper RVA
`0x5fe0` uses the complemented toxicity parameters supplied by RVA `0x59b0`.
The prior-only predicate is called before the enrollment loop. The loop steps
by cohort size and clamps to minimum enrollment; it also appends cap display
placeholders. These instruction observations are a lead for controlled native
references, not validation of a posterior-to-vector compatibility API. Check
the vector's cap placeholder semantics before deriving bounds for replay.
