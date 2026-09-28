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
characteristics are integrated. Multc Lean pending-outcome accrual is under
review, and U2OET GAO probabilities/likelihood are the next Luna assignment.
These additions build on implemented posterior calculations and supply
missing trial workflows or named models.

| Entry | Concrete missing work | Available source / qualification |
| --- | --- | --- |
| Dose Schedule Finder #75 | Aggregate OCs/calibration and delayed low-grade-to-DLT classification after the integrated calendar replay | [Calendar guide](../docs/dose-schedule-trials.md), `research/dose-schedule-audit.md`, [2007 primary paper](https://odin.mdacc.tmc.edu/~pfthall/main/ClinTrials%20dose-sched%202007.pdf). Within-patient adaptation policies and native files/reports remain open. |
| PRT #69 | Calendar replay and operating characteristics | Cached primary paper, conduct and simulation guides under `research/raw/PRT`; reuse probit risk, state-space fit and isotonic conduct rules. |
| Multc Lean #12 | Duration/accrual simulation with pending outcomes | Official cached logistics guide; accrual can continue when pending outcomes cannot alter the next decision. Existing exact complete-outcome OCs do not implement this workflow. |
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
| U2OET #77 | GAO continuation-ratio probabilities and model fitting | PDS/CMI/hybrid fitting, calendar replay and aggregate OCs already exist; [guide](../docs/u2oet.md). Gaussian-copula scenario generation is also present and is a different feature. |
| BARD #165 | BF-BLRM route and integrated stages with titration/expansion | Existing covariate minimization and OBD selection remain useful; hidden native settings need explicit caller configuration or further source evidence; [guide](../docs/bard.md). |
| SurvivalContour #166 | Stratified interval-censored Cox, neural models and interval-model bootstrap uncertainty | Existing right-censored Cox/AFT and implemented interval families are separate completed components; [contour guide](../docs/survival-contour.md), [interval guide](../docs/interval-survival.md). Verify advertised native scope before extending model families. |
| Proportional Density #78 | Full-data disease-curve bootstrap, unequal-censoring calibration and bootstrap parameter uncertainty | Existing failure-only bootstrap is present. The downloaded archive has no bootstrap code; distinguish paper workflows/extensions from missing native routines; [guide](../docs/proportional-density.md). |
| BOIN12 #148 | Two-stage mode, 3+3 run-in, tradeoff utility and multilevel endpoints | Existing joint decisions/simulation are present. The proposed nonadditive joint RDS helper still lacks native-feature evidence and must remain an extension; [guide](../docs/boin12.md). |

This audit confirms that substantial statistical work remains alongside many
presentation-only or parity gaps. No percentage or fixed completion date can
be inferred from 63 implemented / 66 partial / 9 pending, and inaccessible
primary sources remain a separate constraint on full coverage.
