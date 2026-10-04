# Statistical coverage priority review

This bounded review originally covered 67 partial-program entries. TTEConduct, CID2BP and CONFINT subsequently completed their report workflows, leaving 64 partial entries in this review. It distinguishes unresolved statistical contracts from cases where no additional calculation was identified and the remaining work is input, reporting, or native-application parity. The classifications alone do not promote catalog statuses, estimate completion percentages, or claim complete site coverage. References point to the inspected local guides and audits; no new native execution or live application inspection was performed.

## A. Statistical or source-contract uncertainty remains

| Program | Unresolved contract | Evidence |
| --- | --- | --- |
| EffTox (#2) | Legacy contour-fitting objective and trinary calibration/native behavior remain unverified. Existing binary/trinary mathematical APIs do not resolve these source choices. | [legacy contour audit](efftox-legacy-contour-audit.md); [trinary calibration audit](efftox-trinary-calibration-audit.md) |
| Multc99 (#3) | Broader multiple-event scope was identified, but the archive/source was not recovered; only the fixed Phase-IIa contract is validated. | [Multc source records](../docs/multc-sources.json) |
| ToxFinder (#14) | Stage-2 information criterion is unresolved; the guide’s second-derivative substitute does not establish the native Eq. 12 rule. | [ToxFinder guide](../docs/toxfinder.md) |
| bCRM (#15) | Joint two-outcome likelihood/association prior, two-stage conduct, and post-trial four-parameter logistic fit lack a recovered contract. | [bCRM guide](../docs/bcrm.md); [source records](../docs/bcrm-sources.json) |
| SYNERGY (#18) | The four 2007 parametric response surfaces and fitting procedures remain unavailable; the inspected PMC, publisher, and archive routes were exhausted. | [response-surface audit](synergy-response-surface-audit.md) |
| STPLAN (#41) | Native inverse bounds and integer allocation of proportional group totals are unspecified. | [planning guide](../docs/stplan-planning.md) |
| Adaptive Randomization (#62) | Explicit Python controller policies exist; native scheduler/control ordering, floors and RNG conventions remain unspecified. | [controller audit](arand-controller-audit.md); [calendar guide](../docs/arand-calendar.md) |
| CI of Interaction Index (#65) | Pooled measurement-error inference is not recovered; the existing guide identifies plots/case-study fixtures as native workflow rather than a new calculation contract. | [interaction guide](../docs/interaction-index.md) |
| Bayesian Chi-Square TTE (#66) | Rounded observations, censored diagnostics, native fitting/priors, fallback priors, Rychlik rank/trim convention, sorting, and reports remain uncertain. No BIC/DIC output contract was established. | [Bayesian chi-square guide](../docs/bayesian-chi-square.md) |
| PRT (#69) | The published covariance-weighted isotonic projection can materially leave [0,1]; original executable behavior is unresolved. | [PRT guide](../docs/prt.md) |
| WFMM (#70) | Native prior/proposal defaults, `delta_omega` mapping, other transforms, compression and files remain unresolved. | [WFMM audit](wfmm-audit.md); [prediction guide](../docs/wfmm-prediction.md) |
| TPI (#72) | Scenario-based tuning remains underspecified and unimplemented; this is distinct from the available TPI/mTPI rules and informative priors. | [TPI guide](../docs/tpi.md) |
| Dose Schedule Finder (#75) | Automatic calibration, synthetic low-grade episode generation and within-patient adaptation are listed as open, but the inspected contract does not give complete procedures. | [Dose Schedule Finder guide](../docs/dose-schedule.md) |
| U2OET (#77) | Explicit models, fitting, calibration and trial workflows exist; native prior interpretation, calibration and full operating-characteristic validation are unresolved. | [U2OET guide](../docs/u2oet.md); [source records](../docs/u2oet-sources.json) |
| Proportional density (#78) | Unequal-censoring treatment-effect null calibration is mentioned, but null generation and restricted-fit procedure are absent. | [full-bootstrap audit](proportional-density-full-bootstrap-audit.md) |
| BMA CRM (#81) | Native prior files, automatic skeleton calibration and conduct conventions remain. | [BMA CRM guide](../docs/bmacrm.md) |
| CiBolus (#86) | Mathematical model and supplied joint-truth simulation are covered, but calendar and pending-outcome conduct remain source-underspecified. | [CiBolus guide](../docs/cibolus.md) |
| UAROET (#92) | Native prior files, pseudo-trial/ESS calibration, delayed-outcome simulation and reporting remain unresolved. | [UAROET sources](../docs/uaroet-sources.json); [UAROET guide](../docs/uaroet.md) |
| BOIN desktop (#99) | Desktop-specific combination titration/run-in and integration remain uncertain; installer/help rules were not numerically inspected. | [desktop guide](../docs/boin-desktop.md) |
| MTADF (#114) | No uncovered calculation was identified, but strict-paper versus inclusive-app cutoff behavior, fixed calibration and hidden settings remain ambiguous. | [MTADF guide](../docs/mtadf.md) |
| KeyboardComb (#121) | Key 5 says randomize among two while its candidate set contains three; do not infer the rule. | [Keyboard combination guide](../docs/keyboard-combination.md); [source audit](../docs/keyboard-combination-source.md) |
| BOINComb (#128) | App-specific titration caps, moderate-toxicity stopping and 3+3 run-in lack a complete inspected contract; do not infer wrapper rules from `next.comb`. Standard/waterfall decisions and titration are already implemented. | [BOIN combination guide](../docs/boin-combination.md); [source guide](../docs/boin-combination-source.md) |
| BARPO (#130) | The DBCD desired-target vector construction is unknown; explicit vectors exist, but simultaneous-floor parity and reporting remain unresolved. | [BARPO guide](../docs/barpo.md); [trial guide](../docs/barpo-trials.md); [source guide](../docs/barpo-source.md) |
| BMA-CRM calibration (#133) | Automatic skeleton calibration remains blocked by unresolved displayed Q, regression intercept and candidate-selection conventions. | [CRM model-selection guide](../docs/crm-model-selection.md); [source records](../docs/bmacrm-sources.json) |
| TITE-Keyboard (#135) | The paper gives an adaptive pending-patient timing weight and effective non-DLT count update, but the inspected contract does not establish the joint posterior update for shared timing parameters: using only observed DLT times ignores interim follow-up truncation and pending-patient survival contributions. The app exposes uniform/piecewise timing, so this paper-method extension does not establish app parity. | [TITE-Keyboard guide](../docs/tite-keyboard.md); [source records](../docs/tite-keyboard-sources.json) |
| Platform BARPO (#137) | Complete-outcome platform conduct is covered, but no sufficiently complete delayed-response controller contract was found. | [BARPO trials](../docs/plbarpo-trials.md); [BARPO control](../docs/plbarpo-control.md) |
| Phase2Delay (#141) | Native priors, sampler, calibration and continuous-monitoring rules are unknown. | [Phase2Delay guide](../docs/phase2delay.md) |
| U-BOIN (#142) | Delayed immune-outcome imputation remains pending. | [U-BOIN guide](../docs/uboin.md) |
| BFMonitor (#143) | ESS-to-shape calibration rule is unspecified. | [BFMonitor guide](../docs/bfmonitor.md) |
| iBOIN (#145) | Isotonic weights, tie handling and final-selection defaults are unknown. | [iBOIN guide](../docs/iboin.md); [selection audit](iboin-selection-audit.md) |
| BOIN12 (#148) | Run-in precedence is unspecified; multilevel behavior is under development. | [two-stage audit](boin12-two-stage-audit.md) |
| Two-arm BOP2 / rBOP2 (#150) | Binary monitoring and explicit-grid calibration are implemented, but native automatic-grid and allocation-rounding conventions are unspecified. Paired-endpoint decision rules are unresolved; cached help also gives inconsistent Dirichlet formulas for endpoint marginals. | [rBOP2 binary guide](../docs/rbop2-binary.md); [calibration guide](../docs/rbop2-calibration.md); [source audit](../docs/rbop2-binary-source.md) |
| TITE-BOIN12 (#152) | Cited supplement sections S7/S2 were unavailable, leaving their rules unresolved. | [TITE-BOIN12 guide](../docs/tite-boin12.md) |
| BaCIS (#153) | Published fixed cutoff conflicts with reported operating characteristics; no automatic calibration procedure was recovered. Explicit-cutoff simulation remains usable. | [BaCIS simulation guide](../docs/bacis-simulation.md) |
| BayesESS (#154) | Unknown-mean variance ESS has conflicting Hessian signs, an unspecified prior-df adjustment, and native scale/argument inconsistencies. | [variance ESS audit](normal-variance-ess-audit.md) |
| MERIT (#160) | Source does not resolve how previously stopped arms enter later isotonic pooling. The Python policy remains explicit rather than a claimed native rule. | [interim audit](merit-interim-search-audit.md) |
| BARD (#165) | Stage-2 timing, eligibility and quota rules remain unknown. | [integrated audit](bard-integrated-audit.md) |
| SurvivalContour (#166) | Recovered helpers provide model-based confidence-limit surfaces, but mapping app counting-process fields to Python interval-likelihood fitting remains unresolved. This is a contract boundary, not evidence that every optional forest-library feature is an app requirement. | [source coverage audit](survival-contour-coverage-audit.md) |
| MDS-HOPE (#171) | Cytogenetics handling, standardization and baseline are absent from recovered sources. | [source status](mds-hope-source-status.md) |
| Rare Disease 123 (#172) | Cohort staggering, generalized threshold and prior remain unknown. | [generalization audit](rare-disease-123-generalization-audit.md) |
| Success calibration (#173) | Native optimizer, search and rounding conventions remain unspecified; explicit Python calibration is separate from native parity. | [success calibration guide](../docs/success-calibration.md) |

## B. No additional calculation identified; remaining work is input, output, or application parity

| Program | Remaining work seen in the inspected evidence | Evidence |
| --- | --- | --- |
| PerfectMatch (#7) | PDNN fitting/expression, normalization, per-probeset correlations and the five-quantile chip summary are covered. Native parameter/data/output/rescaling, unspecified QC metrics and display workflows remain. | [PerfectMatch guide](../docs/perfectmatch.md); [quantile profile](../docs/perfectmatch-quantile-profile.md) |
| Multc Lean (#12) | Statistical rules/calendar are covered; native input, report and timing conventions remain. | [Multc guide](../docs/multc.md) |
| ASYPOW (#33) | LR/SMO families and source-defined vector sample-size inversions are covered; remaining items are interactive/native reporting workflows. | [ASYPOW guide](../docs/asypow.md); [vector inversion audit](asypow-vector-inversion-audit.md) |
| Parallel phase I/II (#85) | C-design and six-dose calendar workflow are implemented; native configurable input, report and full parity remain. | [parallel phase guide](../docs/parallel-phase12.md) |
| Bayes Factor TTE (#89) | Monitoring, boundaries, calendar replay and simulation are covered; native integer-day/text/HTML reports and timing are not established. | [Bayes Factor survival guide](../docs/bayes-factor-survival.md) |
| Pinnacle (#95) | Peak-detection algorithm is covered; unsupported TIFF encodings, project ingestion, interactive editing and reports remain. | [Pinnacle guide](../docs/pinnacle.md) |
| OneArmTTE (#98) | Monitoring and calendar simulation are covered; desktop scenario editor, interactive HTML/report UI and native RNG parity remain. | [OneArmTTE guide](../docs/one-arm-tte.md) |
| BOP2 online (#112) | Documented binary, ordinal/multiple, joint, survival, calendar, calibration and sample-size workflows are covered; protocol/report formats, animation and native optimizer equivalence remain. | [BOP2 sources](../docs/bop2-sources.json); [binary guide](../docs/bop2-binary.md); [survival guide](../docs/bop2-survival.md) |
| BOIN (#120) | Single-agent conduct, safety, final MTD, simulation, accelerated titration, 3+3 comparison, boundary inversion and protocol text are covered; animation and HTML/Word reports remain. | [BOIN guide](../docs/boin.md) |
| Keyboard (#127) | Posterior keys, conduct, safety, integer tables, isotonic MTD and batch simulation are covered; integrated protocol/report output remains. | [Keyboard guide](../docs/keyboard.md) |
| TITE-BOIN (#129) | Imputation, ordinary/informative weights, interim/safety gates, optional 3+3 modifications and calendar simulation are covered; flowcharts and integrated protocols/reports remain. | [TITE-BOIN guide](../docs/tite-boin.md) |
| CRM Suite (#132) | CRM/BMA/DA posterior, pending look-ahead, calendar, cohort simulation and OC are implemented; native files/reports and older conduct differences remain. | [CRM conduct](../docs/crm-conduct.md); [CRM simulation](../docs/crm-simulation.md) |
| TOP (#134) | Joint monitoring, calendar simulation, finite-grid optimization and holdout validation exist; native optimizer grids/report/app-version parity remain. | [TOP guide](../docs/top-endpoints.md); [calibration guide](../docs/top-endpoints-calibration.md) |
| BOP2 desktop (#144) | Desktop product has been retired to the online application; remaining differences are product/input/presentation boundaries. | [desktop guide](../docs/bop2-desktop.md) |
| aPCoA (#147) | Remaining items are file, formula-display and styling workflows; no separate missing calculation was identified. | [aPCoA guide](../docs/apcoa.md) |
| IPDfromKM (#151) | Digitization and native graphics workflow remain; these are input/presentation features. | [IPDfromKM guide](../docs/ipdfromkm.md) |
| BOP2-DC (#156) | Documented advertised workflows are covered; optional extensions are not automatically missing app methods. | [remaining-methods audit](bop2-dc-remaining-methods.md) |
| CondiS (#157) | Eight refinement learners are implemented. The separate vignette example uses target-derived inputs and pre-split imputation, so it does not establish future-subject prediction behavior. | [workflow audit](condis-workflow-audit.md) |
| BCHM (#158) | No additional advertised mathematical workflow was identified; file/report workflows and direct JAGS parity remain. | [BCHM guide](../docs/bchm.md) |
| EasyCellType (#159) | Versioned gene-ID mapping and plot workflow remain. | [EasyCellType guide](../docs/easycelltype.md) |
| BFBOIN (#162) | Report output and explicit calendar/RNG conventions remain. | [BFBOIN guide](../docs/bf-boin.md) |
| DCT (#164) | Continuous/binary planning formulas are covered; native report and rounding behavior remain. | [DCT guide](../docs/dct-normal.md) |
| PoP (#175) | Source-defined boundaries, selection and operating characteristics are implemented; HTML/Word/report parity remains. | [PoP guide](../docs/pop-design.md) |

The classifications above are review findings, not status changes. A source-contract uncertainty should be resolved before adding a purported native method; input and presentation work can still be useful community functionality without being described as statistical coverage.

## Completed after this review

- TTEConduct (#63): the static HTML report echoes design inputs and boundaries, saves atomically and reopens for viewing. Continuous caller-unit values and an explicit search cap are documented substitutions for native rounded-day presentation. See [report guide](../docs/tteconduct-report.md).
- CID2BP (#38): bounded repeated comparisons support both count-entry modes, changing confidence/methods and cumulative reports for all nine numerical options. See [session guide](../docs/cid2bp-session.md).

- CONFINT (#64): the repeated-calculation log records all eight source menu families, defaults and complete diagnostics in readable saved reports. See [report guide](../docs/confint-session.md) and [source crosswalk](report-workflow-completion-audit.md).

These three entries retain their independent numerical reference checks. Focused workflow checks cover their saved reports and validation boundaries; exact terminal or HTML formatting is not the completion criterion.
