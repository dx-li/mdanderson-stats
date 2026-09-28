# Statistical coverage priority audit — September 28, 2026

The user prioritizes usable Python statistical methods over reproducing every
desktop interface or adding CI for each small change. Catalog status covers
broader software workflows, so the 63 implemented / 66 partial / 9 pending entry
counts must not be interpreted as a percentage of missing statistical methods
or remaining engineering time.

A read-only review of six partial entries found the following distinctions:

| Entry | Statistical coverage already present | Remaining documented scope |
| --- | --- | --- |
| CID2BP #38 | All nine numerical menu options, including corrected approximations and exact-tail calculations | Native session/report interfaces; [guide](../docs/cid2bp.md) |
| CONFINT #64 | Normal, binomial, Poisson, binomial-difference and exponential-survival assurance/inversions, plus hazard peak/range search | Native session/report workflows; [guide](../docs/confint.md) |
| TTEConduct #63 | Exponential/inverse-gamma posterior rule, strict stopping cutoff, exposure monitoring and continuous stopping boundaries | Native report handling and day-rounding parity; the separate calendar-simulation program is catalog #98; [guide](../docs/tteconduct.md) |
| aPCoA #147 | Covariate-adjusted ordination equations and basic group plots | Source covariance ellipses and medoid/member connectors, interactive files/formulas and app audit; [guide](../docs/apcoa.md) |
| BCHM #158 | Native-weighted clustering and target-specific logistic-normal borrowing | Plot/report/file workflows and direct Shiny/JAGS parity; [guide](../docs/bchm.md) |
| DCT #164 | Continuous and binary weighted-z planning, partial/full decentralization, repeated exchangeable outcomes, unequal variances and achieved power | Native rounding and reports; [guide](../docs/dct-normal.md) |

This review does not promote these entries to implemented or erase source
convention gaps. It identifies no additional core statistical model to build
within this six-entry subset. Active implementation should therefore continue
with missing inference and trial workflows, such as EasyCellType multilevel
GSEA and PLBARPO platform progression, before cosmetic/native report work.

If extending aPCoA later, the local author source is
`research/raw/aPCoA/aPCoA/R/aPCoA.R`: its group ellipse uses the covariance scaled
by `sqrt(2 * F_(2,n-1)(0.95))`, and its medoid connectors use `pam(k=1)` on the
within-group original and adjusted distances. Those are concrete statistical
display features, but do not represent an absent ordination method.

## Next source-backed workflow candidates

The subsequent triage distinguishes native workflow gaps from new mathematical
extensions. Dose Schedule Finder calendar replay and PLBARPO control operating
characteristics are integrated, along with Multc Lean pending-outcome calendar
replay. U2OET GAO probabilities and likelihoods are also integrated; fitting
and native prior interpretation remain separate gaps.
These additions build on implemented posterior calculations and supply
missing trial workflows or named models.

| Entry | Concrete missing work | Available source / qualification |
| --- | --- | --- |
| Dose Schedule Finder #75 | Aggregate OCs/calibration and delayed low-grade-to-DLT classification after the integrated calendar replay | [Calendar guide](../docs/dose-schedule-trials.md), `research/dose-schedule-audit.md`, [2007 primary paper](https://odin.mdacc.tmc.edu/~pfthall/main/ClinTrials%20dose-sched%202007.pdf). Within-patient adaptation policies and native files/reports remain open. |
| PRT #69 | Native timing/operating-characteristic replication and projection safeguards after explicit-input calendar replay | Cached primary paper, conduct and simulation guides under `research/raw/PRT`; [replay guide](../docs/prt-calendar.md). The literal projection can produce invalid probabilities, and the Appendix-B timing generator remains unavailable. |
| Multc Lean #12 | Generated timing and aggregate duration simulation after integrated explicit-timing replay | The logistics guide establishes look-ahead suspension; the replay implements it. The user guide does not give a distinct toxicity ascertainment-time law; do not invent native timing assumptions. |
| STPLAN #41 | No missing native integer-allocation feature | Native 4.5 `SOURCE/abink.f` reads double-precision proportions/group sizes, and `SOURCE/qbink.f90` returns `grpsz(i)=n*prop(i)`. There is no native rounding/remainder rule. Existing fractional Python sizing matches that contract; integer allocation would be a separate extension. |
| BOIN12 #148 | Joint-cell RDS table for nonadditive utility | Existing joint-count decision/simulation kernels can support enumeration, but cached `escalation/R/boin12_rds.R` supports marginal/additive tables only. Treat joint enumeration as a Python extension unless a native source establishes that feature. |

The BOIN12 extension is feasible, but should not displace a missing advertised
statistical workflow merely because it is smaller to implement. These findings
do not change catalog completion labels.

## Named-method gaps after guide and module review

A further read-only audit checked the current module coverage against the
remaining-scope paragraphs in the guides. These are more specific than the
catalog's partial-entry count. Each still needs its exact primary-source
contract checked before implementation; this is not permission to substitute
a generic method with a similar name.

| Entry | Remaining named method or workflow | Qualification / evidence |
| --- | --- | --- |
| SYNERGY #18 | Four 2007 parametric response surfaces and 2008 wild-bootstrap intervals | Current semiparametric surface fitting is present; [guide](../docs/synergy-surface.md). |
| U2OET #77 | GAO fitting and native prior interpretation | Explicit-coefficient GAO probabilities and Gaussian likelihoods are now available in the [GAO guide](../docs/u2oet-gao.md). PDS/CMI/hybrid fitting, calendar replay and aggregate OCs already exist; [guide](../docs/u2oet.md). |
| BARD #165 | Titration/expansion, stage-two calendar timing, native per-arm quota behavior and calibration | BF-BLRM model fitting, stage-one calendar replay and stage-two continuation now connect eligible carryover, minimization and final OBD selection under a combined enrollment target. Hidden native settings need explicit caller configuration or further source evidence; [continuation guide](../docs/bard-two-stage.md), [calendar guide](../docs/bard-blrm-trials.md). |
| SurvivalContour #166 | Five neural prediction/learning workflows and interval-model bootstrap uncertainty | Ordinary/stratified right-censored Cox, AFT/splines, numeric forests, interval PH/competing-risk and shared-coefficient stratified interval-PH are available. The five named neural learners require distinct backend contracts; [contour guide](../docs/survival-contour.md), [stratified interval guide](../docs/interval-survival-stratified.md). |
| Proportional Density #78 | Full-data disease-curve bootstrap, unequal-censoring calibration and bootstrap parameter uncertainty | Existing failure-only bootstrap is present. The downloaded archive has no bootstrap code; distinguish paper workflows/extensions from missing native routines; [guide](../docs/proportional-density.md). |
| BOIN12 #148 | Two-stage mode and unresolved 3+3 run-in precedence | Tradeoff-to-utility mapping now supports the existing posterior/decision/OBD/simulation workflow. Multilevel endpoints are marked under development in the cached app. Nonadditive joint RDS native support remains unverified; [guide](../docs/boin12.md). |

This audit confirms that substantial statistical work remains alongside many
presentation-only or parity gaps. No percentage or fixed completion date can
be inferred from 63 implemented / 66 partial / 9 pending, and inaccessible
primary sources remain a separate constraint on full coverage.

## Run-in and parametric-source clarification

The SYNERGY parametric models still lack their exact 2007 equations and fitting
contract in the available cache; model names from a secondary review are not
enough to implement them faithfully. No new generic replacement was added.

BOIN12's `RunIn3+3.txt` says that its option imposes the 3+3 rule at sample
sizes three and six, explicitly triggering de-escalation at at least 2/3 or
2/6 DLTs. The ordinary phi-T=.25 BOIN boundary is approximately .2984, so it
also de-escalates at 1/3. The option is therefore potentially substantive,
but the help does not resolve whether 1/3 forces staying or only disables
the toxicity-triggered de-escalation before ordinary utility selection. The
third-party comparator has no BOIN12 run-in integration. The cached file
`research/raw/BOIN12/BOIN12-paper.pdf` is an HTML download page, not a paper.
No new option was added on the basis of an unverified interpretation.

BARD's cached primary paper supplies an exact BF-BLRM model and independent
normal priors on log-alpha/log-beta. Visual inspection confirms a raw dose
ratio in its predictor, not the customary log-dose ratio. Its simulation
example values are not native app defaults. This supports a new explicitly
configured paper-method component; the app guide itself describes BF-BOIN.

## Two-stage BOIN12 source boundary

A second audit and fresh primary-source search confirmed the stage switch and
stage-specific safety/efficacy criteria in the
[official two-stage help](https://biostatistics.mdanderson.org/shinyapps/BOIN12/BOIN12Stop.pdf).
However, that help does not specify the stage-one next-dose equation. The
[original BOIN12 article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7713525/)
describes the single-stage utility method, not the app's later two-stage option.
Substituting ordinary BOIN movement would be an explicit Python protocol
choice, not a verified port of the native option. No such substitution was
added. The existing BOIN and BOIN12 kernels remain available separately.

CiBolus complete-outcome cohort simulation has since been implemented and
checked against independent R outcome/decision references and sequential
actual posterior fits; [guide](../docs/cibolus-trials.md). Serial aggregate
operating characteristics now add selection/stop rates, allocation and pooled
outcome summaries, trial-clustered Monte Carlo errors and replayable seeds.
Calendar conduct and prior calibration remain open. Numeric survival-forest
out-of-bag curves and native-convention concordance error are now implemented
and checked against unchanged native C kernels. Explicit permutation importance
is also implemented, with independent native-kernel comparisons covering whole-
forest and smaller blocks, omitted tail trees and undefined block errors.
Categorical forest splits, missing-value handling, alternative splitting and
native anti-split importance remain open.

BARD stage-two continuation now includes eligible stage-one patients at the
chosen dose pair in both allocation history and the total enrollment target,
then applies final OBD selection. Independent base-R ledgers verify four
continuations. Accelerated-titration sequencing is defined in the cached guide,
but needs an explicit Python assessment-timing convention. Expansion has
unresolved cap-counting and response-eligibility differences between the app
help and paper; these are not silently substituted.
