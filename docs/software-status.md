# Software coverage

This page summarizes the 138 desktop and online entries in the bundled
[source catalog](../src/mdanderson_stats/catalog.json). The catalog retains
implemented features, validation evidence and source provenance for each entry.

| Status | Entries | Meaning |
| --- | ---: | --- |
| Implemented | 91 | The cataloged scope is implemented and has recorded numerical validation. |
| Partial | 41 | Documented Python methods are available; some methods or workflows remain open. |
| Pending | 6 | No implementation is yet recorded. |

These are software-entry counts, not a percentage of remaining engineering work.
Implemented entries provide usable Python workflows and recorded validation;
identical native interfaces, file bytes and random streams are not required.
A partial entry may already cover every advertised statistical endpoint while
input/output work or source-contract uncertainties remain. Read each guide for
supported inputs, numerical limits and differences from native software. The
project is independent of MD Anderson.

Original contributions use the [MIT License](../LICENSE.md). Source references
and adapted components have separate licensing histories. See
[third-party notices](../THIRD_PARTY_NOTICES.md) before redistribution;
some legacy adaptations retain commercial-use restrictions.

## Implemented

| Program | Source entry | Python documentation |
| --- | --- | --- |
| CONFINT | [desktop #64](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/64) | [Guide](confint.md) |
| CID2BP | [desktop #38](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/38) | [Guide](cid2bp.md) |
| STPLAN | [desktop #41](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/41) | [Guide](stplan.md) |
| TTEConduct | [desktop #63](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/63) | [Guide](tteconduct.md) |
| ACCFLF | [desktop #16](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/16) | See catalog feature and validation notes |
| ANOVA DDP | [desktop #67](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/67) | [Guide](anovaddp.md) |
| Bayes Factor Binary | [desktop #94](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/94) | [Guide](bayes-factor-binary.md) |
| Bayesian Efficacy Monitoring Via Posterior Probability | [online #109](https://biostatistics.mdanderson.org/shinyapps/BEMPO/) | [Guide](bayesian-monitoring.md) |
| Bayesian Efficacy Monitoring Via Predictive Probability | [online #110](https://biostatistics.mdanderson.org/shinyapps/BEMPR/) | [Guide](bayesian-monitoring.md) |
| Bayesian Hierarchical Model - Binomial Data | [online #106](https://biostatistics.mdanderson.org/shinyapps/BHM-BLN/) | [Guide](hierarchical-binomial.md) |
| Bayesian Hierarchical Model - Normal Data | [online #107](https://biostatistics.mdanderson.org/shinyapps/BHM-NN/) | [Guide](hierarchical-normal.md) |
| Bayesian Toxicity Monitoring | [online #100](https://biostatistics.mdanderson.org/shinyapps/BTOX/) | [Guide](bayesian-monitoring.md) |
| Bayesian Update for a Beta-Binomial Distribution | [online #101](https://biostatistics.mdanderson.org/shinyapps/BU1BB/) | [Guide](beta-updating.md) |
| Bayesian Update for a Normal Distribution with Known and Unknown Variance | [online #103](https://biostatistics.mdanderson.org/shinyapps/BNORM/) | [Guide](normal-updating.md) |
| Bayesian Update for Two Beta-Binomial Distributions | [online #102](https://biostatistics.mdanderson.org/shinyapps/BU2BB/) | [Guide](beta-updating.md) |
| BERDS | [desktop #35](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/35) | [Guide](berds.md) |
| Beta Binomial Distribution Demo | [desktop #96](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/96) | See catalog feature and validation notes |
| BLIP | [desktop #36](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/36) | [Guide](blip.md) |
| BlockARAND | [desktop #90](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/90) | See catalog feature and validation notes |
| BP1CI | [desktop #74](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/74) | [Guide](bp1ci.md) |
| CATBUB Design | [desktop #97](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/97) | [Guide](catbub.md) |
| CDFLIB90 | [desktop #21](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/21) | [Guide](cdflib90.md) |
| CTA | [desktop #30](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/30) | [Guide](cta.md) |
| CUMINC | [desktop #39](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/39) | [Guide](cuminc.md) |
| CUMNOR | [desktop #40](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/40) | [Guide](numerical-methods.md) |
| DRDIST | [desktop #31](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/31) | [Guide](drdist.md) |
| ESS Regresssion | [desktop #80](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/80) | [Guide](regression-ess.md) |
| EVENTCHART | [desktop #32](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/32) | [Guide](eventchart.md) |
| EXPSURV | [desktop #28](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/28) | [Guide](expsurv.md) |
| EXTSIG | [desktop #42](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/42) | [Guide](extsig.md) |
| GOFCHI | [desktop #44](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/44) | [Guide](numerical-methods.md) |
| Inequality Calculator | [desktop #9](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/9) | [Guide](inequality-calculator.md) |
| INVMF | [desktop #46](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/46) | [Guide](numerical-methods.md) |
| KSB1CI | [desktop #47](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/47) | [Guide](ksb1ci.md) |
| KSBIN1 | [desktop #25](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/25) | [Guide](ksbin1.md) |
| KSBIN2 | [desktop #24](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/24) | [Guide](ksbin2.md) |
| KWRANGE | [desktop #48](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/48) | [Guide](range-tests.md) |
| Misclib | [desktop #87](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/87) | See catalog feature and validation notes |
| MUHAZ | [desktop #49](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/49) | [Guide](muhaz.md) |
| MULTI | [desktop #50](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/50) | [Guide](multi-session.md) |
| MULTINOMPOW | [desktop #19](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/19) | [Guide](multinomial-power.md) |
| ONESAMPLE | [desktop #22](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/22) | [Guide](onesample.md) |
| Parameter Estimation for A Diagnostic Test | [online #104](https://biostatistics.mdanderson.org/shinyapps/DIAG/) | [Guide](diagnostic-and-roc.md) |
| ParameterSolver | [desktop #6](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/6) | [Guide](parameter-solver.md) |
| Phase II Predictive Probability | [desktop #84](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/84) | [Guide](phase2-predictive.md) |
| Predictive Probabilities | [desktop #10](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/10) | [Guide](predictive-binary.md) |
| RANDLIB | [desktop #27](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/27) | [Guide](randlib.md) |
| RANGE2 | [desktop #53](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/53) | [Guide](range-tests.md) |
| RANLIST | [desktop #29](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/29) | [Guide](ranlist.md) |
| Response and Survival | [desktop #82](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/82) | See catalog feature and validation notes |
| ROSE: randomized optimal selection design for dose optimization | [online #168](https://biostatistics.mdanderson.org/shinyapps/ROSE) | See catalog feature and validation notes |
| Sample Size Calculation - Binary Endpoint | [online #139](https://biostatistics.mdanderson.org/shinyapps/Nbinary) | [Guide](binary-sample-size.md) |
| Sample Size Calculation - Continuous Endpoint | [online #119](https://biostatistics.mdanderson.org/shinyapps/Nnormal/) | [Guide](continuous-sample-size.md) |
| Sample Size Calculation - Time-to-event Endpoint | [online #138](https://biostatistics.mdanderson.org/shinyapps/Nsurvival) | [Guide](survival-sample-size.md) |
| SEQBIN | [desktop #54](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/54) | [Guide](seqbin-coverage.md) |
| Simon's Two-Stage Design | [online #113](https://biostatistics.mdanderson.org/shinyapps/Simon2S/) | [Guide](simon-two-stage.md) |
| SINGLE | [desktop #55](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/55) | [Guide](single-coverage.md) |
| SOGS | [desktop #56](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/56) | [Guide](sogs.md) |
| SORTF90 | [desktop #57](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/57) | See catalog feature and validation notes |
| SPPCR | [desktop #26](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/26) | [Guide](sppcr-research.md) |
| STATTAB | [desktop #23](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/23) | [Guide](stattab-research.md) |
| STUKEL | [desktop #58](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/58) | [Guide](stukel.md) |
| SURVAN | [desktop #59](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/59) | See catalog feature and validation notes |
| TDTASP | [desktop #20](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/20) | [Guide](tdtasp.md) |
| TRAX | [desktop #60](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/60) | See catalog feature and validation notes |
| Varying Cut-Point and Parameter Estimation of the ROC Curve Analysis | [online #105](https://biostatistics.mdanderson.org/shinyapps/DTROC/) | [Guide](diagnostic-and-roc.md) |
| WINDOWS | [desktop #61](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/61) | [Guide](windows.md) |
| ASYPOW | [desktop #33](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/33) | [Guide](asypow.md) |
| Sample Size Determination for Decentralized Clinical Trials (DCTs) | [online #164](https://biostatistics.mdanderson.org/shinyapps/DCTs) | [Guide](dct-normal.md) |
| Keyboard: a novel Bayesian toxicity probability interval design for phase I clinical trial | [online #127](https://biostatistics.mdanderson.org/shinyapps/Keyboard) | [Guide](keyboard.md) |
| One Arm Time to Event Simulator | [desktop #98](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/98) | [Guide](one-arm-tte.md) · [Reports](one-arm-tte-report.md) |
| Bayesian Optimal Interval (BOIN) Design for Phase I Clinical Trials | [online #120](https://biostatistics.mdanderson.org/shinyapps/BOIN/) | [Guide](boin.md) |
| Pinnacle | [desktop #95](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/95) | [Guide](pinnacle.md) |
| EasyCellType: Automatic annotation tool designed for Sing-cell RNA sequencing data | [online #159](https://biostatistics.mdanderson.org/shinyapps/EasyCellType/) | [Fisher guide](easycelltype.md), [ranked scores](easycelltype-gsea.md), [bundled references](easycelltype-builtin-reference.md) |
| Bayesian Cluster Hierarchical Model for Subgroup Borrowing | [online #158](https://biostatistics.mdanderson.org/shinyapps/BCHM/) | [Guide](bchm.md), [input and reports](bchm-scenarios.md) |
| BOP2 Desktop - Bayesian Optimal Phase II Design | [desktop #144](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/144) | [Successor workflow](bop2-desktop.md) |
| BOP2: Bayesian Optimal Phase II Design with Simple and Complex Endpoints | [online #112](https://biostatistics.mdanderson.org/shinyapps/BOP2) | [Guide](bop2-binary.md) |
| Posterior Predictive Design for Phase I Clinical Trials | [online #175](https://biostatistics.mdanderson.org/shinyapps/PoPdesign/) | [Design, plots and saved inputs](pop-community-workflow.md) |
| aPCoA: Covariate Adjusted Principal Coordinates Analysis | [online #147](https://biostatistics.mdanderson.org/shinyapps/aPCoA) | [Guide](apcoa.md) |
| BOP2 design with decision making on dual criteria | [online #156](https://biostatistics.mdanderson.org/shinyapps/BOP2-DC) | [Methods](bop2-dc.md) and [reports](bop2-dc-community-report.md) |
| TOP: Time-to-Event Bayesian Optimal Phase II Trial Design | [online #134](https://biostatistics.mdanderson.org/shinyapps/TOP2) | [Guide](top-endpoints.md), [study reports](top-community-report.md) |
| IPDfromKM: Reconstruct Individual Patient Data (IPD) From Kaplan-Meier Survival Curve | [online #151](https://biostatistics.mdanderson.org/shinyapps/IPDfromKM) | [Guide](ipdfromkm.md), [reconstruction diagnostics](ipdfromkm-diagnostics.md), [image workflow](ipdfromkm-digitization.md) |
| Calibration of Bayesian Success Criteria for Clinical Trials | [online #173](https://www.trialdesign.org/one-page-shell.html#BayesianCalibration) | [Guide](success-calibration.md), [automatic two-arm calibration](success-two-arm-automatic.md) |
| Parallel phase I and II design | [desktop #85](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/85) | [Guide](parallel-phase12.md), [importance calendar](parallel-phase12-importance-calendar.md), [six-dose simulation](phase12-calendar-oc.md), [probability summary](parallel-phase12-probability-summary.md) |
| Time-to-Event Keyboard Design for Phase I Clinical Trials | [online #135](https://biostatistics.mdanderson.org/shinyapps/TITE-KEYBOARD) | [Guide](tite-keyboard.md) |
| Dose Optimization with Backfill and Adaptive Randomization (BARD) | [online #165](https://biostatistics.mdanderson.org/shinyapps/BARD) | [Stage two](bard.md), [BF-BLRM model](bard-blrm.md), [BF-BOIN trials](bard-bf-boin-trial.md), [BF-BLRM trials](bard-blrm-stochastic.md), [BF-BOIN OC](bard-bf-boin-simulation.md), [BF-BLRM OC](bard-blrm-simulation.md), [Saved studies](bard-study.md), [Reports](bard-report.md) |
| Backfill Bayesian Optimal Interval (BF-BOIN) Design for Phase I Clinical Trials | [online #162](https://biostatistics.mdanderson.org/shinyapps/BF-BOIN/) | See catalog feature and validation notes |
| A Phase I/II Design to Identify Optimal Biological Dose for Molecularly Targeted Agents | [online #114](https://biostatistics.mdanderson.org/shinyapps/MTADF/) | [Isotonic and logistic workflows](mtadf.md), including recovered author decisions, replay and simulation |
| FLECS90 | [desktop #43](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/43) | [Translator guide](flecs90.md) |
| Multc Lean | [desktop #12](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/12) | [Native saved studies](multc-lean-model.md); [Monitoring](multc.md) |
| Multc99 | [desktop #3](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/3) | [General study workflows](multc99.md) |

## Partially implemented

| Program | Source entry | Python documentation |
| --- | --- | --- |
| 1+2+3: to find the optimal biological dose for rare diseases | [online #172](https://biostatistics.mdanderson.org/shinyapps/1plus2plus3) | [Guide](rare-disease-123.md) |
| Adaptive Randomization | [desktop #62](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/62) | [Posterior](arand.md), [calendar](arand-calendar.md), [simulation](arand-simulation.md) |
| Bayes Factor TTE | [desktop #89](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/89) | [Guide](bayes-factor-survival.md) |
| Bayesian Adaptive Randomization and Efficacy Monitoring with Posterior Probability | [online #130](https://biostatistics.mdanderson.org/shinyapps/BARPO/) | [Guide](barpo-reference.md) |
| Bayesian Chi Square TTE fit | [desktop #66](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/66) | [Guide](bayesian-chi-square.md) |
| Bayesian Effective Sample Size Calculator | [online #154](https://biostatistics.mdanderson.org/shinyapps/BayesESS) | [Guide](conjugate-ess.md) |
| Bayesian hierarchical classification and information sharing for clinical trials with subgroups and binary outcomes | [online #153](https://biostatistics.mdanderson.org/shinyapps/BaCIS) | [Guide](bacis.md) |
| Bayesian Model Averaging Continuous Reassessment Method | [online #133](https://biostatistics.mdanderson.org/shinyapps/BMACRM) | See catalog feature and validation notes |
| Bayesian Optimal Interval Design (BOIN) for Drug Combination Trials | [online #128](https://biostatistics.mdanderson.org/shinyapps/BOINComb/) | [Guide](boin-combination-source.md) |
| Bayesian Optimal Interval Design with Informative Prior (iBOIN) for Phase I Clinical Trials | [online #145](https://biostatistics.mdanderson.org/shinyapps/iBOIN) | [Guide](iboin.md), [saved reports and replay](iboin-report.md) |
| Bayesian Phase 2 Design with Delayed Outcomes | [online #141](https://biostatistics.mdanderson.org/shinyapps/Phase2Delay) | See catalog feature and validation notes |
| bCRM | [desktop #15](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/15) | See catalog feature and validation notes |
| BMA CRM | [desktop #81](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/81) | See catalog feature and validation notes |
| BOIN Design Desktop Program | [desktop #99](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/99) | See catalog feature and validation notes |
| BOIN12: Bayesian Optimal Interval Phase I/II Trial Design for Utility-Based Dose Finding | [online #148](https://biostatistics.mdanderson.org/shinyapps/BOIN12) | [Guide](boin12.md) |
| CI of Interaction Index | [desktop #65](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/65) | [Interaction indices](interaction-index.md); [parametric models](synergy-parametric.md); [semiparametric method](synergy-surface.md) |
| CiBolus | [desktop #86](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/86) | [Guide](cibolus.md), [Prior calibration](cibolus-calibration.md) |
| CondiS: Imputation of Censored Lifetimes for Machine Learning-Based Survival Analysis | [online #157](https://biostatistics.mdanderson.org/shinyapps/CondiS/) | [Guide](condis.md), [curve comparison](condis-survival-comparison.md) |
| CRM Suite | [desktop #132](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/132) | See catalog feature and validation notes |
| Dose Schedule Finder | [desktop #75](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/75) | [Model](dose-schedule.md), [calendar trials](dose-schedule-trials.md) |
| EffTox | [desktop #2](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/2) | [Guide](efftox.md), [trinary calibration](efftox-trinary-calibration.md) |
| Find optimal biological dose (OBD) for targeted and immune therapies | [online #142](https://biostatistics.mdanderson.org/shinyapps/UBOIN) | [Guide](uboin.md) |
| KeyboardComb: the Keyboard Design for Drug Combination Trials | [online #121](https://biostatistics.mdanderson.org/shinyapps/KeyboardComb/) | See catalog feature and validation notes |
| MDS-HOPE: An interactive risk assessment tool for patients treated with HMA | [online #171](https://biostatistics.mdanderson.org/shinyapps/MDS-HOPE/) | [Guide](mds-hope.md) |
| MERIT: Multiple-dose Randomized Phase II Trial Design for Dose Optimization and Sample Size Determination | [online #160](https://biostatistics.mdanderson.org/shinyapps/MERIT/) | [Guide](merit.md) |
| PerfectMatch | [desktop #7](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/7) | [Guide](perfectmatch.md) |
| Platform Design of Bayesian Adaptive Randomization with Posterior Probability | [online #137](https://biostatistics.mdanderson.org/shinyapps/PLBARPO/) | [Control monitoring](plbarpo-control.md), [active allocation](plbarpo-allocation.md), [no-control trials](plbarpo-trials.md), [control trials](plbarpo-control-trials.md) |
| Proportional density | [desktop #78](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/78) | [Guide](proportional-density.md) |
| PRT | [desktop #69](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/69) | See catalog feature and validation notes |
| Single arm phase II monitoring using Bayes factor with iMOM prior for binary outcome | [online #143](https://biostatistics.mdanderson.org/shinyapps/BFMonitor) | [Guide](bfmonitor.md) |
| SurvivalContour: Show Survival Prediction in Contour Plot | [online #166](https://biostatistics.mdanderson.org/shinyapps/survivalContour/) | [Guide](survival-contour.md), [forest rank splits](random-survival-forest-logrankscore.md), [Brier splits](random-survival-forest-brier.md) |
| SYNERGY | [desktop #18](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/18) | [Interaction indices](interaction-index.md); [parametric models](synergy-parametric.md); [semiparametric method](synergy-surface.md) |
| Time-to-event Bayesian Optimal Interval (TITE-BOIN) Design for Phase I Clinical Trials | [online #129](https://biostatistics.mdanderson.org/shinyapps/TITE-BOIN/) | [Guide](tite-boin.md) |
| TITE-BOIN12: extension of BOIN12 for late-onset toxicity and efficacy | [online #152](https://biostatistics.mdanderson.org/shinyapps/TITE-BOIN12) | [AL and final selection](tite-boin12.md); [BDA](tite-boin12-bda.md) |
| ToxFinder | [desktop #14](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/14) | [Guide](toxfinder.md); [prior elicitation](toxfinder-prior-elicitation.md) |
| Toxicity Probability Intervals | [desktop #72](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/72) | [Guide](mtpi.md) |
| Two-arm BOP2: Bayesian Optimal Phase II two-arm Design | [online #150](https://biostatistics.mdanderson.org/shinyapps/rBOP2) | See catalog feature and validation notes |
| U2OET | [desktop #77](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/77) | [PDS/CMI and trials](u2oet.md), [GAO probabilities](u2oet-gao.md), [GAO trials](u2oet-gao-trials.md) |
| UAROET | [desktop #92](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/92) | [Guide](uaroet.md), [trial simulation](uaroet-trials.md) |
| WFMM | [desktop #70](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/70) | [Guide](wfmm.md) |

| RMC-COMPASS: Renal Medullary Carcinoma - Clinical and Outcomes Model for Prognostic Assessment and Survival Stratification | [online #174](https://biostatistics.mdanderson.org/shinyapps/RMC-COMPASS/) | [Rounded public model](rmc-compass.md) |

## Pending

| Program | Source entry | Python documentation |
| --- | --- | --- |
| A Bayesian Phase II Platform Trial Design for Drug Combinations | [online #140](https://biostatistics.mdanderson.org/shinyapps/ComPAS) | See catalog feature and validation notes |
| Bayesian Survival Function Posterior Estimates | [online #146](https://biostatistics.mdanderson.org/shinyapps/BayesianSurvival) | [Guide](bayesian-survival-source-status.md) |
| K-COMPASS: Estimate systemic-therapy free survival following MDT for oligometastatic clear cell RCC | [online #169](https://biostatistics.mdanderson.org/shinyapps/K-COMPASS/) | See catalog feature and validation notes |
| MDS-DPSS: An Interactive Dynamic Prognostic Scoring System Tool for MDS | [online #170](https://biostatistics.mdanderson.org/shinyapps/MDS-DPSS/) | See catalog feature and validation notes |
| Predicting survival for patients with malignant pleural effusions using the BLESS models | [online #149](https://biostatistics.mdanderson.org/shinyapps/BLESS) | See catalog feature and validation notes |
| Risk of CNS Metastasis in Clinically Localized Melanoma | [online #161](https://biostatistics.mdanderson.org/shinyapps/CNSRISK) | See catalog feature and validation notes |

October 9 continuation: Multc Lean's recovered native model/default and saved-study
workflow completes entry #12, with exact count estimates kept separate from
legacy Monte Carlo duration/balk means. The audit records original control
references and explicit numerical/CLR/RNG/layout differences. The snapshot is
now 91 implemented, 41 partial and 6 pending; see the
[current continuation record](../research/source-recovery-2026-10-09.md).
