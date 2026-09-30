# Statistical coverage priority audit — September 28, 2026

September 30 fitting/prediction update: the original 2010 GAO model now has
posterior fitting under explicit Gaussian-coordinate inputs, a joint
dose-grid validity restriction and uniform association. Independent integration
checks the constrained prior, complete/toxicity-only posteriors and correlation
posterior. Elicited prior-center calibration and native workflow mapping remain
open. WFMM now supplies posterior predictive coefficients for existing/new
random-effect levels and future replicates, with independent mixture moment
checks. PCA/PCw/wPC and automatic retention still lack exact native contracts;
the predictive interface does not claim native file-format parity. Historical
remaining-scope notes below describe their earlier checkpoints.

September 30 U2OET update: adaptive corner-utility precision is now connected
to PDS/CMI/hybrid calendar trials, including final follow-up and cached-data
decisions. The original 2010 GAO probability model is separately implemented
with centered raw doses and endpoint-specific signed interactions. It is
distinct from the existing 2017 raw-dose/shared-positive-interaction model;
the latter's fitted priors must not be substituted for the former. Original
GAO prior fitting, calibration and native parameter-file mapping remain open.
The TITE-Keyboard timing source check below identifies an unresolved posterior
likelihood contract; a generic timing helper would not complete that workflow.

September 29 transform/precision update: WFMM now accepts explicit square
analysis/synthesis inverse pairs, propagates covariance through the supplied
synthesis matrix and rejects severely ill-conditioned pairs. U2OET has a
standalone adaptive PDS/CMI/hybrid fitter that continues chains toward the
guide's per-chain corner-utility precision targets, with a finite draw cap
and separate split-Rhat diagnostics. Independent R comparisons and bounded
posterior checks are recorded in the component audits. WFMM PCA/PCw/wPC and
retained-component semantics, adaptive GAO fitting and adaptive precision in
trial drivers remain open. Current catalog counts are 63 implemented,
67 partial and eight pending; historical counts below describe earlier states.

September 29 conduct/simulation update: BaCIS simulations now optionally retain
native-definition ESS by replication/subgroup, with stable subgroup means and
Monte Carlo errors. Dose Schedule Finder now accepts explicit delayed toxicity
adjudications and creates likelihood records using only information available
at each analysis. Qualifying events are backdated to onset; actual treatment
after onset remains separately available. Pending earlier episodes block final
readiness. The paper does not specify a low-grade episode generator or fully
resolve an automatic day-14 adjudication boundary; neither was invented.

A review of RF-SRC 3.2.2's alternative `logrankscore` branch found an apparent
inconsistency between covariate-sort indices and score/member indices in
`logRankNCR`. The default implemented log-rank branch is separate. Extending
to this alternative needs an independent native/published-method comparison
before deciding whether to reproduce or correct that behavior; it was not
ported blindly.

September 29 update: BCHM's three advertised analysis plots and aPCoA's
source-convention data ellipses and row-profile medoid connectors are now
implemented and checked against independent R references. BCHM's cached
package/help exposes no additional scientific computation beyond its fit,
summaries and these plots. Its app advertises CSV, PDF and MCMC downloads,
but the available cache lacks their server handlers and output schemas.
These are remaining app I/O/report parity gaps; they are not evidence of a
missing calibration or simulation method. Prioritize other entries' missing
scientific workflows over reproducing those report formats. The historical
audit below records the earlier state.

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
replay. Dose Schedule Finder aggregate OCs and explicit-prior U2OET GAO
fitting are also integrated. Native GAO prior interpretation and prior calibration remain separate gaps;
the explicit-prior GAO calendar driver is now connected.
These additions build on implemented posterior calculations and supply
missing trial workflows or named models.

| Entry | Concrete missing work | Available source / qualification |
| --- | --- | --- |
| Dose Schedule Finder #75 | Automatic calibration and delayed low-grade-to-DLT classification after integrated calendar replay and aggregate OCs | [Calendar guide](../docs/dose-schedule-trials.md), `research/dose-schedule-audit.md`, [2007 primary paper](https://odin.mdacc.tmc.edu/~pfthall/main/ClinTrials%20dose-sched%202007.pdf). Within-patient adaptation policies and native files/reports remain open. |
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
| U2OET #77 | Native GAO prior interpretation and GAO prior calibration | Explicit-coefficient GAO probabilities, Gaussian likelihoods and [explicit-prior posterior fitting](../docs/u2oet-gao-fit.md) are available. PDS/CMI/hybrid fitting, calendar replay and aggregate OCs already exist; [guide](../docs/u2oet.md). |
| BARD #165 | Expansion, stage-two calendar timing, native per-arm quota behavior and calibration | BF-BLRM model fitting, calendar replay with optional accelerated titration and stage-two continuation now connect eligible carryover, minimization and final OBD selection under a combined enrollment target. Hidden native settings need explicit caller configuration or further source evidence; [continuation guide](../docs/bard-two-stage.md), [titration guide](../docs/bard-titration.md). |
| SurvivalContour #166 | Five neural prediction/learning workflows; stratified interval-PH bootstrap | Ordinary/stratified right-censored Cox, AFT/splines, numeric/categorical forests, interval PH/competing-risk and shared-coefficient stratified interval-PH are available. Ordinary and cluster PH coefficient bootstrap, competing-risk coefficient bootstrap and repeated-visit conversion are now included. The pinned icenReg source explicitly does not provide nonparametric baseline confidence bands. The five neural learners require distinct backend contracts; [contour guide](../docs/survival-contour.md), [cluster bootstrap guide](../docs/interval-survival-cluster-bootstrap.md). |
| Proportional Density #78 | Unequal-censoring treatment-effect null calibration and bootstrap parameter uncertainty | Failure-only and full-data disease-curve goodness-of-fit bootstraps are present. The downloaded archive has no bootstrap code; distinguish paper workflows/extensions from missing native routines; [guide](../docs/proportional-density.md). |
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
Balanced pseudo-data prior calibration and prior-predictive beta ESS are now
available with explicit joint elicitation tables and caller-selected variances;
[guide](../docs/cibolus-calibration.md). Calendar conduct and automatic variance
selection remain open. Numeric survival-forest
out-of-bag curves and native-convention concordance error are now implemented
and checked against unchanged native C kernels. Explicit permutation importance
is also implemented, with independent native-kernel comparisons covering whole-
forest and smaller blocks, omitted tail trees and undefined block errors.
Categorical forest splits and source-defined anti-split and random-routing
importance are now included. Missing-value handling and alternative splitting
remain open.

BARD stage-two continuation now includes eligible stage-one patients at the
chosen dose pair in both allocation history and the total enrollment target,
then applies final OBD selection. Independent base-R ledgers verify four
continuations. Accelerated-titration sequencing from the cached guide is now
implemented with explicit assessment timing and BF-BLRM safety conventions. Expansion has
unresolved cap-counting and response-eligibility differences between the app
help and paper; these are not silently substituted.

## BOP2 endpoint-scope correction

The cached #112 app snapshot, `research/raw/BOP2/app.html`, version
1.4.27.0 updated September 4, 2026, lists exactly six endpoint options: binary
efficacy, binary toxicity, efficacy and toxicity, multiple efficacy, ordinal
efficacy, and time to event. The existing Python monitoring, operating-
characteristic, calibration and sample-size guides cover those six families.
The TTE primary guide specifies the single-arm exponential/inverse-gamma model.
There is no advertised control-arm or joint-survival endpoint in this snapshot.
The separately recorded September 27 check observed version 1.4.29.0; the
endpoint mapping above does not assert optimizer parity with that newer build.

Earlier remaining-scope wording incorrectly presented two-arm/joint survival
as a missing #112 feature. It is now identified as a potential extension.
Native optimizer/report equivalence remains unverified; this correction does
not promote the catalog status or claim full native equivalence. The retired
#144 desktop, rBOP2 #150 and BOP2-DC #156 remain distinct catalog products.
See [BOP2 source provenance](../docs/bop2-sources.json),
[the survival guide](../docs/bop2-survival.md) and
[the desktop mapping](../docs/bop2-desktop.md).


## TITE-Keyboard adaptive timing source check

The cached primary paper, section 2.3, defines a shared scaled-Beta conditional
DLT-time distribution and weights as its posterior-averaged CDF. It does not
specify the timing-parameter posterior likelihood for censored or pending
observations, nor whether observed DLT times alone enter that fit. These choices
matter because event ascertainment depends on available follow-up. The method
text gives independent Gamma(.1,.1) priors as an example, while the simulation
sensitivity analysis uses Gamma(.5,.5); their parameter convention is not stated.
The cached app supports the already implemented uniform and informative
three-piece weights. An automatic adaptive timing fitter needs additional
source evidence; a supplied-draw CDF helper alone would not close this workflow
gap. No native adaptive-weight implementation is claimed.
