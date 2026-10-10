# October 10 source recovery and implementation checkpoint

The preserved 138-entry catalog now has **93 implemented / 39 partial / 6 pending**.
CI of Interaction Index #65 and Bayes Factor TTE #89 complete their recovered
functional workflows. The user's target is at least 100 implemented and no
pending software; that target remains unmet. Seven further completions are
needed for 100. Counts describe the preserved catalog, not a freshly verified
inventory of every current site tool.

## Completed workflows

The [interaction-index audit](interaction-index-workflow-audit.md) records the
recovered fixed-ray ratio, residual-df pooling, exact three-drug scenario
constants, two simulation workflows, retained QQ diagnostics and saved outputs.
An original native fixed-ray variance defect is corrected rather than copied.
The independent unchanged original functions and corrected base-R covariance
calculation have separate committed synthetic references.

The [BayesFactorTTE audit](bayes-factor-tte-native-control-audit.md) records original
managed arrival truncation, enrollment checks, strict event/look ties, terminal
timing and integer-day boundary search. Native-scheduled replay/simulation,
original sorted-count quantiles and saved full-precision studies are now usable.
Original controller execution substitutes only declared runtime/RNG/BF services;
independent mathematical integration remains separate evidence. Native int32
overflow, the one-patient empty loop and capped/incomplete boundary tables are
handled explicitly in Python.

## Newly recovered official archives

All original files stay under ignored `research/raw/source-recovery-2026-10-10`.
The URLs use `https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/`
plus the path below. No original binaries or source files are redistributed.

| Entry | Official path | SHA-256 |
| --- | --- | --- |
| EffTox #2 | `EffTox/EffTox_V5.2.3.0.zip` | `84fc159c00bb825d0e28c18d095a77a98a8fd97e17c8b2451f9e03e3484b5c47` |
| ToxFinder #14 | `ToxFinder/ToxFinder_V1.1.1.zip` | `eddd41346542f3ee398a07eef8873a3687c3f3a9fa93e2a960ed217d01276d70` |
| Adaptive Randomization #62 | `ARAND/AdaptiveRandomization_V5.2.2.zip` | `ff9c4d490a8876e1b3197d6fcb809bd8d001761be13f083a667e0e7db34fc877` |
| CI of Interaction Index #65 | `CIInteractionIndex/CIInteractionIndex_V2.0.1source.zip` | `e69f045d6c15595cc67e5055c94cc8ad119230e07e334a2def11708a0f1712f1` |
| TPI #72 | `TPI/TPI_V2.1PID439.zip` | `4e87f1a816bbf1ffb1a7817ac2a87d3de019090e78d4c24c07ec6c8b44059c8c` |
| Proportional Density #78 | `PropDen/PropDen_V1.00.zip` | `04f6c1de72fe8a08349bb18c797c1b816ea4ae53ad06479c3e486cc529c2a7e9` |
| BMA-CRM #81 | `BMACRM/BMA-CRMSimulator_V2.2.4.zip` | `1f8a00c8766bdcfbb569ec4990c29c91b026ab54215a6b1436bc90b6361a79c3` |
| Bayes Factor TTE #89 | `BayesFactorTTE/BayesFactorTTE_V1.1_NoFX4.0.zip` | `7fa329b68f7ac60887571087b38820048c5bdd0ce7a95ca9780ce1df5ef89c5a` |

The metadata BOIN #99 archive name still returned 404 at its official prefix.
PropDen contains only parameter estimation and model checking R files: no
unequal-censoring bootstrap null generator was found. Availability alone does
not complete either workflow.

## Six pending software contracts

Public Shiny interfaces were inspected through normal OS-verified HTTPS using
the existing proxy and an XHR polling transport. No certificate verification
was disabled. Additional global CA trust changes were rejected by automatic
review and were not made. The usable HTTPS transport leaves that rejected
operation unnecessary.

| Entry | New observation | Exact artifact still needed |
| --- | --- | --- |
| ComPAS #140 | PID 1000 v1.0 public shell and paper DOI 10.1002/sim.8026 | Adaptive-shrinkage joint model, priors, posterior/decision/simulation contract or author code |
| BayesianSurvival #146 | PID 1029 v1.0.6.0 covariates and conditional five-year-survivor UI | Fitted oropharyngeal regression/posterior and prediction/uncertainty model |
| BLESS #149 | PID 1049 v1.0.2.0; two linked instructions PDFs now recovered | Four fitted coefficient vectors and baseline survival tables/eAppendix |
| CNSRISK #161 | PID 1138 v1.0.0.0; default rounded 2/5/10-year risks .02/.06/.10 | Exact competing-risk fit and baselines |
| K-COMPASS #169 | PID 1178 v1.4.2.0; illustrative 52-month STFS, interval 37–77 | Six-variable fit, transformations, Weibull intercept/shape and uncertainty |
| MDS-DPSS #170 | PID 1183 v1.0.0; time categories, clinical/cytogenetic/mutation inputs | Dynamic fitted score, calibration and output contract |

No generic statistical calculator, rounded-curve fit or guessed prior is used
to change pending status. Public availability does not reveal server-side models.

## Concrete next candidates toward 100

These are source leads, not promises of seven automatic promotions. Each must
resolve its remaining scientific contract and pass an independent reference.

1. EffTox #2: extracted `Efftox2Calculations.dll` hash
   `7570fa52e5c03c4818ae8c4f17913bd75a2f036ebac844aa6c86e2ce3a183fd0`.
   Managed metadata identifies `ContourObjective.SetTargets` RVA `0xab84`,
   objective `0xb144`, `ContourParameterSolver.Solve` `0xec94` and
   `g_CalcContourParams` export `0x124a0`. Static decoding now identifies the weighted-SSE/penalty objective; see
   the [updated legacy audit](efftox-legacy-contour-audit.md). Execute objective/
   solver references and separately resolve trinary calibration; decoding
   addresses/formulas is not numerical validation.
2. Adaptive Randomization #62: `ARand.Calcs.dll` hash
   `a11d60157236bdf2e3395de39909dbb029b22bae52bcee5dd6d8ee4eebb3a6e3`.
   Named Director/AdaptiveRandomizer/StoppingRule and simulation method bodies
   can resolve allocation floors, trigger precedence and timing. The installed
   native model is mixed C++/CLI; validate decoded bodies before execution.
3. TPI #72: the archive is mTPI, not the original TPI program. Its R simulation
   final selection uses Beta(.005,.005), inverse posterior-variance PAVA weights
   and dose-index `1e-10` perturbations. This differs from the current Python
   uniform-prior/equal-weight default. Its next-dose safety expression also omits
   the toxicity subtraction in one Beta argument; record/correct defects explicitly.
   Native final-selection recovery and usable tuning/reporting remain open.
4. BMA-CRM #81: `BMA-CRMKernel.dll` hash
   `a170b501674307aee638b03915c5fb61337f7008a3a65c06b505098903e30768`.
   Resolve legacy prior/skeleton calibration from the extracted model and guides.
5. CRM Suite #132: the same recovered desktop kernel identifies trial decisions
   at `0x316b0`, DA decisions at `0x334f0` and simulation at `0x2dfa0`.
   Resolve wait/raw-rate precedence and suspension scheduling; compare source
   versions before applying desktop rules to the online successor.
6. BMA-CRM online #133: verify the online contract against its own documentation
   and recovered prior/skeleton controls; a desktop filename alone is insufficient.
7. ToxFinder #14: the recovered Wise installer contains the .WISE payload.
   A plain 7z listing exposes that section, but has not recovered Java code.
   Extract and audit the exact stage-two information criterion before promotion.

The complete [39-entry remaining review](statistical-coverage-priority-2026-10-04.md)
continues to track the other partial contracts. No authors were contacted.

## Validation and delivery

The combined affected interaction/BayesFactorTTE suite passes 73 tests with
warnings treated as errors. Ruff and formatting pass; mypy covers 696 source
modules. Full-suite, isolated-wheel and final delivery results are appended
below after completion. Research-only R/dnfile/dncil are not runtime dependencies.

Final verification: the full warnings-as-errors suite passes **33,975 tests in
771.84 seconds** under Python 3.13.5. It began before the final bounded-day-tape
preflight/Boolean guard and its two additional cases. The final affected run
passes **73 tests in 2.67 seconds**; current collection is **33,977**. A second
whole-suite run is not claimed. Both original-reference regeneration checks pass.

Ruff, formatting (2,409 files), mypy (696 modules), wheel/sdist build and isolated
wheel byte checks pass. All nine Python examples in seven affected guides run
from the installed wheel. Catalog/status tables and all 39 partial contracts
agree; guide links, licenses/notices and original-artifact exclusions are checked.
The source-recovery dependencies remain research-only and the lockfile is unchanged.

Delivery uses `codex/native-software-coverage`, based on verified main `c45e678`.
The prepared PR describes the complete branch, including FLECS90, Multc99,
Multc Lean, WFMM/SYNERGY recovery and this checkpoint. GitHub API PR creation was
previously Forbidden; publication and the latest API result are recorded after
attempting delivery. No merge to main/master is implied by a feature-branch push.
