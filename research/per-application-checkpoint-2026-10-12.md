# Per-application continuation checkpoint — October 12, 2026

Catalog membership was rechecked against both official API lists: 80 desktop +
58 online entries, with identical response hashes to the preserved inventory.
No entry was added or dropped. See [membership evidence](catalog-membership-2026-10-12.json).

Current counts are **103 implemented / 31 partial / 4 pending**. ARAND62 now
meets the guide-defined simulation workflow; see its scope crosswalk below. This continuation
adds the mTPI R final estimator, recovered EffTox contour fitter, CRM desktop
study/controller workflow, ToxFinder native allocation/simulation workflow, and
the complete ARAND guide-defined simulation workflow. ARAND, desktop CRM81/132 and ToxFinder14 are promoted
for their declared completed workflows. BLESS149 now moves from pending to partial
for its recovered published-grid workflow; its deployed daily fit remains unavailable.
It does not re-certify every historical numerical reference or installed example.
The validation report identifies exactly which checks were rerun.

“Implemented” means the cataloged recovered workflow, subject to its guide’s
numerical and compatibility limits. It does not mean native GUI/RNG/file identity,
clinical validity, or an unrestricted license. “Partial” entries below retain the specific statistical, source-contract or
workflow gaps listed in their rows. License scope is tracked separately in the
[community licensing path](community-license-path-2026-10-12.md).

| ID | Application | Status | Remaining boundary / evidence |
| ---: | --- | --- | --- |
| 2 | EffTox | implemented | Current5.2.3 saved-prior/calibration, bounded pending conduct, patient edits/deletion, simulation and reports are integrated. Source references total738 cases, including81 complete native posterior-integrand cases; negative-slope summary inconsistency is an explicit legacy option. All-scenario/batch, original saved-file and qualified precision acceptance are complete; [declared current-binary scope](efftox-current-scope-2026-10-12.md). Historical trinary/legacy scopes remain separate. [Current contracts](efftox-current-workflow-contracts-2026-10-12.md); [recovered fitter](../docs/efftox-native-contour.md); [trinary audit](efftox-trinary-calibration-audit.md). |
| 3 | Multc99 | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 6 | ParameterSolver | implemented | Declared recovered workflow implemented; [Guide](../docs/parameter-solver.md) retains scope and compatibility limits. |
| 7 | PerfectMatch | partial | The paper's array-mean-500 scaling now joins PDNN fitting, conditional expression, normalization, correlations and chip quantiles. Several manual QC outputs and the native correlation convention remain undefined; array parameters and native file workflows are also unresolved. [PerfectMatch guide](../docs/perfectmatch.md); [source records](../docs/perfectmatch-sources.json) |
| 9 | Inequality Calculator | implemented | Declared recovered workflow implemented; [Guide](../docs/inequality-calculator.md) retains scope and compatibility limits. |
| 10 | Predictive Probabilities | implemented | Declared recovered workflow implemented; [Guide](../docs/predictive-binary.md) retains scope and compatibility limits. |
| 12 | Multc Lean | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 14 | ToxFinder | implemented | Physician-input calibration, TMML, native allocation, patient CSV edit/reopen, simulation and diagnostic reports are complete under explicit profiles. Final refit is the study default; legacy final posterior is opt-in. [Scope/evidence](toxfinder-saved-study-2026-10-12.md); [guide](../docs/toxfinder-saved-study.md). |
| 15 | bCRM | partial | High-level stage controls and zero-gap futility are documented. Joint two-outcome likelihood/association prior, exact transition rules, positive-gap futility cutoff and post-trial four-parameter logistic estimator remain unresolved. [bCRM guide](../docs/bcrm.md); [source records](../docs/bcrm-sources.json) |
| 16 | ACCFLF | implemented | Declared recovered workflow implemented; [Guide](../docs/accflf.md) retains scope and compatibility limits. |
| 18 | SYNERGY | partial | The four original parametric models now provide native-validated fits, uncertainty and plots. The separate semiparametric bootstrap interval convention remains unresolved. [parametric audit](synergy-parametric-audit.md); [bootstrap audit](synergy-wild-bootstrap-audit.md) |
| 19 | MULTINOMPOW | implemented | Declared recovered workflow implemented; [Guide](../docs/multinomial-power.md) retains scope and compatibility limits. |
| 20 | TDTASP | implemented | Declared recovered workflow implemented; [Guide](../docs/tdtasp.md) retains scope and compatibility limits. |
| 21 | CDFLIB90 | implemented | Declared recovered workflow implemented; [Guide](../docs/cdflib90.md) retains scope and compatibility limits. |
| 22 | ONESAMPLE | implemented | Declared recovered workflow implemented; [Guide](../docs/onesample.md) retains scope and compatibility limits. |
| 23 | STATTAB | implemented | Declared recovered workflow implemented; [Guide](../docs/stattab-research.md) retains scope and compatibility limits. |
| 24 | KSBIN2 | implemented | Declared recovered workflow implemented; [Guide](../docs/ksbin2.md) retains scope and compatibility limits. |
| 25 | KSBIN1 | implemented | Declared recovered workflow implemented; [Guide](../docs/ksbin1.md) retains scope and compatibility limits. |
| 26 | SPPCR | implemented | Declared recovered workflow implemented; [Guide](../docs/sppcr-research.md) retains scope and compatibility limits. |
| 27 | RANDLIB | implemented | Declared recovered workflow implemented; [Guide](../docs/randlib.md) retains scope and compatibility limits. |
| 28 | EXPSURV | implemented | Declared recovered workflow implemented; [Guide](../docs/expsurv.md) retains scope and compatibility limits. |
| 29 | RANLIST | implemented | Declared recovered workflow implemented; [Guide](../docs/ranlist.md) retains scope and compatibility limits. |
| 30 | CTA | implemented | Declared recovered workflow implemented; [Guide](../docs/cta.md) retains scope and compatibility limits. |
| 31 | DRDIST | implemented | Declared recovered workflow implemented; [Guide](../docs/drdist.md) retains scope and compatibility limits. |
| 32 | EVENTCHART | implemented | Declared recovered workflow implemented; [Guide](../docs/eventchart.md) retains scope and compatibility limits. |
| 33 | ASYPOW | implemented | Declared recovered workflow implemented; [Guide](../docs/asypow.md) retains scope and compatibility limits. |
| 35 | BERDS | implemented | Declared recovered workflow implemented; [Guide](../docs/berds.md) retains scope and compatibility limits. |
| 36 | BLIP | implemented | Declared recovered workflow implemented; [Guide](../docs/blip.md) retains scope and compatibility limits. |
| 38 | CID2BP | implemented | Declared recovered workflow implemented; [Guide](../docs/cid2bp.md) retains scope and compatibility limits. |
| 39 | CUMINC | implemented | Declared recovered workflow implemented; [Guide](../docs/cuminc.md) retains scope and compatibility limits. |
| 40 | CUMNOR | implemented | Declared recovered workflow implemented; [Guide](../docs/numerical-methods.md) retains scope and compatibility limits. |
| 41 | STPLAN | implemented | Declared recovered workflow implemented; [Guide](../docs/stplan.md) retains scope and compatibility limits. |
| 42 | EXTSIG | implemented | Declared recovered workflow implemented; [Guide](../docs/extsig.md) retains scope and compatibility limits. |
| 43 | FLECS90 | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 44 | GOFCHI | implemented | Declared recovered workflow implemented; [Guide](../docs/numerical-methods.md) retains scope and compatibility limits. |
| 46 | INVMF | implemented | Declared recovered workflow implemented; [Guide](../docs/numerical-methods.md) retains scope and compatibility limits. |
| 47 | KSB1CI | implemented | Declared recovered workflow implemented; [Guide](../docs/ksb1ci.md) retains scope and compatibility limits. |
| 48 | KWRANGE | implemented | Declared recovered workflow implemented; [Guide](../docs/range-tests.md) retains scope and compatibility limits. |
| 49 | MUHAZ | implemented | Declared recovered workflow implemented; [Guide](../docs/muhaz.md) retains scope and compatibility limits. |
| 50 | MULTI | implemented | Declared recovered workflow implemented; [Guide](../docs/multi-session.md) retains scope and compatibility limits. |
| 53 | RANGE2 | implemented | Declared recovered workflow implemented; [Guide](../docs/range-tests.md) retains scope and compatibility limits. |
| 54 | SEQBIN | implemented | Declared recovered workflow implemented; [Guide](../docs/seqbin-coverage.md) retains scope and compatibility limits. |
| 55 | SINGLE | implemented | Declared recovered workflow implemented; [Guide](../docs/single-coverage.md) retains scope and compatibility limits. |
| 56 | SOGS | implemented | Declared recovered workflow implemented; [Guide](../docs/sogs.md) retains scope and compatibility limits. |
| 57 | SORTF90 | implemented | Declared recovered workflow implemented; [Guide](../docs/sortf90.md) retains scope and compatibility limits. |
| 58 | STUKEL | implemented | Declared recovered workflow implemented; [Guide](../docs/stukel.md) retains scope and compatibility limits. |
| 59 | SURVAN | implemented | Declared recovered workflow implemented; [Guide](../docs/survan.md) retains scope and compatibility limits. |
| 60 | TRAX | implemented | Declared recovered workflow implemented; [Guide](../docs/trax.md) retains scope and compatibility limits. |
| 61 | WINDOWS | implemented | Declared recovered workflow implemented; [Guide](../docs/windows.md) retains scope and compatibility limits. |
| 62 | Adaptive Randomization | implemented | Guide-defined Binary/TTE simulation, original simulation XML/Python studies, native monitoring/stopping, independently seeded scenarios and all required per-arm reports are complete. Original RNG identity, GUI styling and latent clinical-history conduct are explicit boundaries. [Scope/evidence crosswalk](arand-simulation-scope-2026-10-12.md); [native workflow](../docs/arand-native-workflow.md); [integration checkpoint](arand-native-integration-2026-10-12.md) |
| 63 | TTEConduct | implemented | Declared recovered workflow implemented; [Guide](../docs/tteconduct.md) retains scope and compatibility limits. |
| 64 | CONFINT | implemented | Declared recovered workflow implemented; [Guide](../docs/confint.md) retains scope and compatibility limits. |
| 65 | CI of Interaction Index | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 66 | Bayesian Chi Square TTE fit | partial | Rounded interval-likelihood fitting now covers all seven families with explicit proper Gaussian parameter priors and paired randomized CDF diagnostics. Censored diagnostics, native fitting/priors, fallback priors, Rychlik rank/trim convention, sorting, and reports remain uncertain. No BIC/DIC output contract was established. [Bayesian chi-square guide](../docs/bayesian-chi-square.md); [rounded-time guide](../docs/rounded-tte-bayesian-gof.md) |
| 67 | ANOVA DDP | implemented | Declared recovered workflow implemented; [Guide](../docs/anovaddp.md) retains scope and compatibility limits. |
| 69 | PRT | implemented | Original saved conduct/simulation settings, separate paper/native controllers, generated calendar, numerical holds, saved replay and unconditional batch reports are integrated.21 native fixture files,72 composed enrollment steps and qualified real posterior trials support the [scope crosswalk](prt-simulation-complete-checkpoint-2026-10-12.md). Native RNG/GUI identity and clinical validity are separate claims. |
| 70 | WFMM | partial | Native MOM/profile variance and proposal initialization, other extended transforms/boundaries, pass filtering and files remain unresolved. The inverse-gamma `delta_omega` mapping and energy compression are now native-verified, with five periodic Haar transforms checked. [native-prior audit](wfmm-native-prior-audit.md); [WFMM audit](wfmm-audit.md); [prediction guide](../docs/wfmm-prediction.md) |
| 72 | Toxicity Probability Intervals | implemented | Current version2.1 monitoring and all-scenario simulation/report workflow is integrated with bounded XLSM/JSON input, paper-default and explicit legacy controller/final ranking, native uniform replay and captured profile/seed. Original2007 TPI tuning and informative-prior recalibration remain separate extension limits. [Acceptance](mtpi-desktop-complete-checkpoint-2026-10-12.md). |
| 74 | BP1CI | implemented | Declared recovered workflow implemented; [Guide](../docs/bp1ci.md) retains scope and compatibility limits. |
| 75 | Dose Schedule Finder | implemented | Current2.2.0 guide-defined calibration, saved studies, actual-history conduct, generated low-grade simulation, all-scenario reports and distribution/trial-detail export are integrated. Safe and explicit legacy rules remain separate; independent posterior integration and native fixtures validate distinct claims. [Acceptance](dose-schedule-desktop-complete-checkpoint-2026-10-12.md); [guide](../docs/dose-schedule-desktop.md). |
| 77 | U2OET | partial | Explicit models, fitting, calibration and trial workflows exist; native prior interpretation, calibration and full operating-characteristic validation are unresolved. [U2OET guide](../docs/u2oet.md); [source records](../docs/u2oet-sources.json) |
| 78 | Proportional density | partial | Unequal-censoring treatment-effect null calibration is mentioned, but null generation and restricted-fit procedure are absent. [full-bootstrap audit](proportional-density-full-bootstrap-audit.md) |
| 80 | ESS Regresssion | implemented | Declared recovered workflow implemented; [Guide](../docs/regression-ess.md) retains scope and compatibility limits. |
| 81 | BMA CRM | implemented | Saved studies, explicit-date patient conduct, original cohort replay, simulation and captured reports are complete under independent inference. [Scope](crm-guide-scope-2026-10-12.md); [validation](crm-conduct-validation-2026-10-12.json). Native RNG/GUI identity and clinical validity are not claimed. |
| 82 | Response and Survival | implemented | Declared recovered workflow implemented; [Guide](../docs/response-survival.md) retains scope and compatibility limits. |
| 84 | Phase II Predictive Probability | implemented | Declared recovered workflow implemented; [Guide](../docs/phase2-predictive.md) retains scope and compatibility limits. |
| 85 | Parallel phase I and II design | implemented | Declared recovered workflow implemented; [Guide](../docs/parallel-phase12.md) retains scope and compatibility limits. |
| 86 | CiBolus | partial | Mathematical model and supplied joint-truth simulation are covered, but calendar and pending-outcome conduct remain source-underspecified. [CiBolus guide](../docs/cibolus.md) |
| 87 | Misclib | implemented | Declared recovered workflow implemented; [Guide](../docs/misclib.md) retains scope and compatibility limits. |
| 89 | Bayes Factor TTE | implemented | Declared recovered workflow implemented; [Guide](../docs/bayes-factor-survival.md) retains scope and compatibility limits. |
| 90 | BlockARAND | implemented | Declared recovered workflow implemented; [Guide](../docs/blockarand.md) retains scope and compatibility limits. |
| 92 | UAROET | implemented | Current1.8 saved study, calibration/ESS, continuous precision, explicit safe/legacy conduct, delayed calendar trials, compatible job aggregation and portable reports/histograms are complete within declared contracts. [Acceptance](uaroet-desktop-complete-checkpoint-2026-10-12.md); [guide](../docs/uaroet-desktop-study.md). |
| 94 | Bayes Factor Binary | implemented | Declared recovered workflow implemented; [Guide](../docs/bayes-factor-binary.md) retains scope and compatibility limits. |
| 95 | Pinnacle | implemented | Declared recovered workflow implemented; [Guide](../docs/pinnacle.md) retains scope and compatibility limits. |
| 96 | Beta Binomial Distribution Demo | implemented | Declared recovered workflow implemented; [Guide](../docs/beta-binomial-demo.md) retains scope and compatibility limits. |
| 97 | CATBUB Design | implemented | Declared recovered workflow implemented; [Guide](../docs/catbub.md) retains scope and compatibility limits. |
| 98 | One Arm Time to Event Simulator | implemented | Declared recovered workflow implemented; [Guide](../docs/one-arm-tte.md) retains scope and compatibility limits. |
| 99 | BOIN Design Desktop Program | implemented | Current1.1.0 four-mode saved design, conduct, simulation/comparators, pending and score tables, STFT/WSTFT, standalone final analyses and captured-input bilingual methods/results are complete under documented safe/legacy profiles. Source compatibility, numerical identity and clinical validity remain separate. [Complete crosswalk/evidence](boin-desktop-complete-checkpoint-2026-10-12.md). |
| 100 | Bayesian Toxicity Monitoring | implemented | Declared recovered workflow implemented; [Guide](../docs/bayesian-monitoring.md) retains scope and compatibility limits. |
| 101 | Bayesian Update for a Beta-Binomial Distribution | implemented | Declared recovered workflow implemented; [Guide](../docs/beta-updating.md) retains scope and compatibility limits. |
| 102 | Bayesian Update for Two Beta-Binomial Distributions | implemented | Declared recovered workflow implemented; [Guide](../docs/beta-updating.md) retains scope and compatibility limits. |
| 103 | Bayesian Update for a Normal Distribution with Known and Unknown Variance | implemented | Declared recovered workflow implemented; [Guide](../docs/normal-updating.md) retains scope and compatibility limits. |
| 104 | Parameter Estimation for A Diagnostic Test | implemented | Declared recovered workflow implemented; [Guide](../docs/diagnostic-and-roc.md) retains scope and compatibility limits. |
| 105 | Varying Cut-Point and Parameter Estimation of the ROC Curve Analysis | implemented | Declared recovered workflow implemented; [Guide](../docs/diagnostic-and-roc.md) retains scope and compatibility limits. |
| 106 | Bayesian Hierarchical Model - Binomial Data | implemented | Declared recovered workflow implemented; [Guide](../docs/hierarchical-binomial.md) retains scope and compatibility limits. |
| 107 | Bayesian Hierarchical Model - Normal Data | implemented | Declared recovered workflow implemented; [Guide](../docs/hierarchical-normal.md) retains scope and compatibility limits. |
| 109 | Bayesian Efficacy Monitoring Via Posterior Probability | implemented | Declared recovered workflow implemented; [Guide](../docs/bayesian-monitoring.md) retains scope and compatibility limits. |
| 110 | Bayesian Efficacy Monitoring Via Predictive Probability | implemented | Declared recovered workflow implemented; [Guide](../docs/bayesian-monitoring.md) retains scope and compatibility limits. |
| 112 | BOP2: Bayesian Optimal Phase II Design with Simple and Complex Endpoints | implemented | Declared recovered workflow implemented; [Guide](../docs/bop2-binary.md) retains scope and compatibility limits. |
| 113 | Simon's Two-Stage Design | implemented | Declared recovered workflow implemented; [Guide](../docs/simon-two-stage.md) retains scope and compatibility limits. |
| 114 | A Phase I/II Design to Identify Optimal Biological Dose for Molecularly Targeted Agents | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 119 | Sample Size Calculation - Continuous Endpoint | implemented | Declared recovered workflow implemented; [Guide](../docs/continuous-sample-size.md) retains scope and compatibility limits. |
| 120 | Bayesian Optimal Interval (BOIN) Design for Phase I Clinical Trials | implemented | Declared recovered workflow implemented; [Guide](../docs/boin.md) retains scope and compatibility limits. |
| 121 | KeyboardComb: the Keyboard Design for Drug Combination Trials | partial | Key 5 says randomize among two while its candidate set contains three; do not infer the rule. [Keyboard combination guide](../docs/keyboard-combination.md); [source audit](../docs/keyboard-combination-source.md) |
| 127 | Keyboard: a novel Bayesian toxicity probability interval design for phase I clinical trial | implemented | Declared recovered workflow implemented; [Guide](../docs/keyboard.md) retains scope and compatibility limits. |
| 128 | Bayesian Optimal Interval Design (BOIN) for Drug Combination Trials | partial | App-specific titration caps, moderate-toxicity stopping and 3+3 run-in lack a complete inspected contract; do not infer wrapper rules from `next.comb`. Standard/waterfall decisions and titration are already implemented. [BOIN combination guide](../docs/boin-combination.md); [source guide](../docs/boin-combination-source.md) |
| 129 | Time-to-event Bayesian Optimal Interval (TITE-BOIN) Design for Phase I Clinical Trials | partial | Conduct, timing, calendar simulation and saved protocol reports are covered. The paper defines MTD-relative summaries and regret from failure to de-escalate when two of the first three patients at a dose have DLT, but the continuous-accrual evaluation timing remains unresolved. The report preserves full selection probabilities and per-dose allocation summaries. [TITE-BOIN guide](../docs/tite-boin.md); [report crosswalk](tite-boin-protocol-report-audit.md) |
| 130 | Bayesian Adaptive Randomization and Efficacy Monitoring with Posterior Probability | partial | The DBCD desired-target vector construction is unknown; explicit vectors exist, but simultaneous-floor parity and reporting remain unresolved. [BARPO guide](../docs/barpo.md); [trial guide](../docs/barpo-trials.md); [source guide](../docs/barpo-source.md) |
| 132 | CRM Suite | implemented | Saved studies, explicit-date patient conduct, original cohort replay, simulation and captured reports are complete under independent inference. [Scope](crm-guide-scope-2026-10-12.md); [validation](crm-conduct-validation-2026-10-12.json). Native RNG/GUI identity and clinical validity are not claimed. |
| 133 | Bayesian Model Averaging Continuous Reassessment Method | partial | Automatic skeleton calibration remains blocked by unresolved displayed Q, regression intercept and candidate-selection conventions. [CRM model-selection guide](../docs/crm-model-selection.md); [source records](../docs/bmacrm-sources.json) |
| 134 | TOP: Time-to-Event Bayesian Optimal Phase II Trial Design | implemented | Declared recovered workflow implemented; [Guide](../docs/top-endpoints.md) retains scope and compatibility limits. |
| 135 | Time-to-Event Keyboard Design for Phase I Clinical Trials | implemented | Declared recovered workflow implemented; [Guide](../docs/tite-keyboard.md) retains scope and compatibility limits. |
| 137 | Platform Design of Bayesian Adaptive Randomization with Posterior Probability | partial | Complete-outcome platform conduct is covered, but no sufficiently complete delayed-response controller contract was found. [BARPO trials](../docs/plbarpo-trials.md); [BARPO control](../docs/plbarpo-control.md) |
| 138 | Sample Size Calculation - Time-to-event Endpoint | implemented | Declared recovered workflow implemented; [Guide](../docs/survival-sample-size.md) retains scope and compatibility limits. |
| 139 | Sample Size Calculation - Binary Endpoint | implemented | Declared recovered workflow implemented; [Guide](../docs/binary-sample-size.md) retains scope and compatibility limits. |
| 140 | ComPAS | partial | Recovered five-model hierarchy, explicit groups/model priors, evidence/MAP inference, strict decisions, numerical holds and saved reports. Original defaults and dynamic platform retention/replacement/final/allocation contracts remain unresolved. [Guide](../docs/compas.md); [source and scope](compas-published-core-2026-10-12.md). |
| 141 | Bayesian Phase 2 Design with Delayed Outcomes | partial | Native priors, sampler, calibration and continuous-monitoring rules are unknown. [Phase2Delay guide](../docs/phase2delay.md) |
| 142 | Find optimal biological dose (OBD) for targeted and immune therapies | partial | Multiple-imputation evaluation and stage-II conduct now accept supplied predictive probabilities. The scaled-logistic prediction fit and standardization contract remain unresolved. [U-BOIN guide](../docs/uboin.md); [imputation audit](uboin-imputation-evaluation-audit.md) |
| 143 | Single arm phase II monitoring using Bayes factor with iMOM prior for binary outcome | partial | ESS-to-shape calibration rule is unspecified. [BFMonitor guide](../docs/bfmonitor.md) |
| 144 | BOP2 Desktop - Bayesian Optimal Phase II Design | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 145 | Bayesian Optimal Interval Design with Informative Prior (iBOIN) for Phase I Clinical Trials | partial | Captured-input HTML reports and exact-seed JSON replay are implemented. Native isotonic weights, tie handling and final-selection defaults remain unknown. [iBOIN guide](../docs/iboin.md); [selection audit](iboin-selection-audit.md) |
| 146 | Bayesian Survival Function Posterior Estimates | pending | Fitted oropharyngeal regression/posterior, prediction and uncertainty model. [Evidence](source-recovery-2026-10-10.md#six-pending-software-contracts). |
| 147 | aPCoA: Covariate Adjusted Principal Coordinates Analysis | implemented | Declared recovered workflow implemented; [Guide](../docs/apcoa.md) retains scope and compatibility limits. |
| 148 | BOIN12: Bayesian Optimal Interval Phase I/II Trial Design for Utility-Based Dose Finding | partial | Run-in precedence is unspecified; multilevel behavior is under development. [two-stage audit](boin12-two-stage-audit.md) |
| 149 | BLESS | partial | Four published rounded models, exact seven-day grid, saved inputs and audited reports. Deployed daily baseline, full-precision fit and additional uncertainty contracts remain unknown. [Guide](../docs/bless.md); [recovery](bless-published-recovery-2026-10-12.md). |
| 150 | Two-arm BOP2: Bayesian Optimal Phase II two-arm Design | partial | Binary monitoring and explicit-grid calibration are implemented, but native automatic-grid and allocation-rounding conventions are unspecified. Paired-endpoint decision rules are unresolved; cached help also gives inconsistent Dirichlet formulas for endpoint marginals. [rBOP2 binary guide](../docs/rbop2-binary.md); [calibration guide](../docs/rbop2-calibration.md); [source audit](../docs/rbop2-binary-source.md) |
| 151 | IPDfromKM: Reconstruct Individual Patient Data (IPD) From Kaplan-Meier Survival Curve | implemented | Declared recovered workflow implemented; [Guide](../docs/ipdfromkm.md) retains scope and compatibility limits. |
| 152 | TITE-BOIN12: extension of BOIN12 for late-onset toxicity and efficacy | partial | Cited supplement sections S7/S2 were unavailable, leaving their rules unresolved. [TITE-BOIN12 guide](../docs/tite-boin12.md) |
| 153 | Bayesian hierarchical classification and information sharing for clinical trials with subgroups and binary outcomes | partial | Published fixed cutoff conflicts with reported operating characteristics; no automatic calibration procedure was recovered. Explicit-cutoff simulation remains usable. [BaCIS simulation guide](../docs/bacis-simulation.md) |
| 154 | Bayesian Effective Sample Size Calculator | partial | Unknown-mean variance ESS has conflicting Hessian signs, an unspecified prior-df adjustment, and native scale/argument inconsistencies. [variance ESS audit](normal-variance-ess-audit.md) |
| 156 | BOP2 design with decision making on dual criteria | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 157 | CondiS: Imputation of Censored Lifetimes for Machine Learning-Based Survival Analysis | partial | Base imputation, all eight refiners and descriptive survival comparison are covered. The prediction example includes a target-derived input and performs imputation before splitting; it does not define a future-subject prediction or valid evaluation contract. Reactive app summary and prediction behavior were not recovered. [workflow audit](condis-workflow-audit.md) |
| 158 | Bayesian Cluster Hierarchical Model for Subgroup Borrowing | implemented | Declared recovered workflow implemented; [Guide](../docs/bchm.md) retains scope and compatibility limits. |
| 159 | EasyCellType: Automatic annotation tool designed for Sing-cell RNA sequencing data | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 160 | MERIT: Multiple-dose Randomized Phase II Trial Design for Dose Optimization and Sample Size Determination | partial | Source does not resolve how previously stopped arms enter later isotonic pooling. The Python policy remains explicit rather than a claimed native rule. [interim audit](merit-interim-search-audit.md) |
| 161 | Risk of CNS Metastasis in Clinically Localized Melanoma | pending | Exact competing-risk model and baseline functions. [Evidence](source-recovery-2026-10-10.md#six-pending-software-contracts). |
| 162 | Backfill Bayesian Optimal Interval (BF-BOIN) Design for Phase I Clinical Trials | implemented | Declared recovered workflow implemented; [Guide](../docs/bf-boin.md) retains scope and compatibility limits. |
| 164 | Sample Size Determination for Decentralized Clinical Trials (DCTs) | implemented | Declared recovered workflow implemented; [Guide](../docs/dct-normal.md) retains scope and compatibility limits. |
| 165 | Dose Optimization with Backfill and Adaptive Randomization (BARD) | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |
| 166 | SurvivalContour: Show Survival Prediction in Contour Plot | partial | The wrapper advertises interval-censored PH but the inspected mets implementation interprets the response as counting-process Cox data. A justified baseline-survival confidence surface for the genuine interval likelihood remains absent; support-identification bounds cannot substitute for sampling confidence intervals. [source coverage audit](survival-contour-coverage-audit.md) |
| 168 | ROSE: randomized optimal selection design for dose optimization | implemented | Declared recovered workflow implemented; [Guide](../docs/rose.md) retains scope and compatibility limits. |
| 169 | K-COMPASS: Estimate systemic-therapy free survival following MDT for oligometastatic clear cell RCC | pending | Six-variable fit, transformations, Weibull intercept/shape and uncertainty. [Evidence](source-recovery-2026-10-10.md#six-pending-software-contracts). |
| 170 | MDS-DPSS: An Interactive Dynamic Prognostic Scoring System Tool for MDS | pending | Dynamic fitted score, calibration and output contract. [Evidence](source-recovery-2026-10-10.md#six-pending-software-contracts). |
| 171 | MDS-HOPE: An interactive risk assessment tool for patients treated with HMA | partial | Cytogenetics handling, standardization and baseline are absent from recovered sources. [source status](mds-hope-source-status.md) |
| 172 | 1+2+3: to find the optimal biological dose for rare diseases | partial | Cohort staggering, generalized threshold and prior remain unknown. [generalization audit](rare-disease-123-generalization-audit.md) |
| 173 | Calibration of Bayesian Success Criteria for Clinical Trials | implemented | Declared recovered workflow implemented; [Guide](../docs/success-calibration.md) retains scope and compatibility limits. |
| 174 | RMC-COMPASS: Renal Medullary Carcinoma - Clinical and Outcomes Model for Prognostic Assessment and Survival Stratification | partial | Rounded public coefficients now support approximate point predictions and explicit rounding envelopes. Exact parameters/bounds, uncertainty, raw-input rules and output workflows remain unavailable. [Guide](../docs/rmc-compass.md); [live-model audit](rmc-compass-audit.md) |
| 175 | Posterior Predictive Design for Phase I Clinical Trials | implemented | Declared recovered workflow implemented; [Catalog](../src/mdanderson_stats/catalog.json) retains scope and compatibility limits. |

WFMM70 initialization update: bounded iterative MOM and explicit native EB are
implemented with independent original-executable references (166 MOM systems,
20 EB calls);4 proposal and4 bound matrices resolve native scaling/fallbacks.
Default MOM uses complete trace equations and absolute convergence. Legacy
profile retains omitted residual-column cross terms, zero-component omission
and signed stopping. Native EB reports signed stops separately from convergence.
Automatic conditional optimization/bounds, transforms and complete saved
MCMC/postprocessing/prediction integration remain open. Entry70 stays partial.
