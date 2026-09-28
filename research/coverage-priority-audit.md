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
