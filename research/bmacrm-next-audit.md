# BMA-CRM coverage and next steps (2026-09-26)

After the STPLAN survival batch, prioritize this pending scientific method over
inactive STPLAN interface features. Catalog entries 81 (BMA CRM), 132 (CRM Suite),
and 133 (online BMACRM) need separate version/capability checks; do not mark all
three covered from one posterior calculation.

The first implementation batch now provides `fit_bmacrm`: power-model posterior
integration, marginal evidence, model averaging, and overdose probabilities.
Ten independent base-R cases agree with the Python implementation. The posterior
supports the newer CRM Suite's nondecreasing skeletons as well as the older
desktop's strictly increasing inputs. Fully observed decision rules are exposed
through `bmacrm_decision`. Entries 81 and 132 remain partial; online entry 133
has not yet been verified.

Official current desktop page:
https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/81

The page identifies version 2.2.4 (2018-02-26), combining BMA-CRM and delayed-outcome
DA-CRM. Since version 2.1.2 skeleton inputs are **prior medians**, directly used
as power-model probabilities. The method PDF's earlier prior-mean elicitation
therefore does not describe the current default. Version 2.2.2 also adds a raw
fully-observed toxicity-rate escalation restriction and an MTD rule favoring
levels with at least three treated patients, with an explicit fallback if none
qualify. Read the full guide before implementing these decision rules.

Original method document:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BMACRM/BMA-CRM_Description.pdf

Core model: for skeleton k, dose j, pi[k,j]=p[k,j]**exp(alpha[k]), with a normal
prior on alpha (documented mean zero, variance two). Binomial likelihoods give
one-dimensional posterior integrals and marginal likelihoods. Posterior model
weights combine prior model weights with these evidences; dose toxicity estimates
are their weighted posterior means. Safety stopping mixes the model-specific
posterior probabilities of lowest-dose toxicity exceeding the target. The PDF
also describes optional prior-mean-to-skeleton calibration and next-cohort
selection with no skipping of untried levels. These are scientific calculations,
not mere aliases to generic CRM routines.

Existing `bcrm_model.py` exposes log evidence but uses the Goodman CRM model and
a bounded uniform prior on beta. Inspect it for reusable numerical patterns, not
for assumed model equivalence. DA-CRM and simulated event timing remain separate
workflows requiring their own source audit. The newer online and CRM Suite
interfaces may differ from the older desktop package.

## Current conduct rules

The newer CRM Suite guide resolves details absent from the older guide:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/BMA-CRMSimulatorHelp.pdf

Appendix II, pages 30-31: the no-skip cap is the first untried dose, with lower
than starting doses treated as tried. Raw-rate escalation restrictions apply
to the current and intervening levels, not only the current level. A final MTD
with fewer than three treated patients falls back to the highest lower level
with three; if none exists, retain the candidate with an uncertainty flag.

## Further scientific coverage

Implement bounded complete-outcome trial simulation, then pending-outcome
look-ahead and DA-CRM after verifying their separate contracts. Look-ahead must
not assume that just the all-toxic and no-toxic completions bound every BMA
decision without proving it; model weights also change with outcomes. Exact
bounded completion enumeration is one possible route.

DA-CRM source:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/DA-CRM_Description.pdf

The paper specifies Bernoulli imputation, conditional power-model inference,
and gamma hazard updates (Section 2.3). Its examples use nine time intervals;
the current desktop guide uses six with trimester-based prior calibration and
an endpoint approximation. Preserve this version distinction. Gamma prior
dispersion, time units, posterior sampling precision, and the desktop's rule
for waiting before safety termination need an explicit contract. Do not count
discarding pending outcomes or fractional-binomial fitting as DA-CRM.
If reusing grouped CRM evidence to compare individual missing-outcome
completions, remove its binomial coefficients: those factors are constant
across skeletons for fixed data, but not across different completed outcomes.

## Follow-up after the CRM conduct batch

The DA posterior, prior calibration, decisions and bounded look-ahead now exist;
see `dacrm-audit.md` and `crm-conduct-audit.md`. Remaining implementation should
prioritize trial/cohort simulation and uncovered scientific choices rather than
more small validation cases for existing functions.

The online page was reachable on 2026-09-26 with its trailing slash:
https://biostatistics.mdanderson.org/shinyapps/BMACRM/

It identifies version 1.0.2.0, updated 2025-12-15, and exposes both Bayesian model
averaging (BMA) and Bayesian model selection (BMS). It cites Yin and Yuan (2009)
and Pan and Yuan (2016/2017), *A Default Method to Specify Skeletons for Bayesian
Model Averaging Continual Reassessment Method for Phase I Clinical Trials*.
Its visible surface has simulation and trial-conduct tabs with complete counts.
This confirms that the online entry has additional choices; it does not verify
hidden priors, safety rules, skeleton generation or executable parity. Entry
133 remains pending until these choices have a source-backed Python contract.

Version distinction to preserve: the older BMA-CRM Simulator guide explicitly
describes waiting when a DA safety calculation recommends stopping, then stopping
only on full-information CRM. The newer CRM Suite guide states a general safety
cutoff and does not repeat that exception. The current `crm_suite` policy follows
the newer written rule. A future older-desktop profile must state this difference
explicitly instead of claiming both native programs behave identically.

The authors' JASA paper is available at a working institutional URL:
https://saasresearch.hku.hk/~gyin/materials/2009YinYuanJASA.pdf

It explicitly defines BMS as choosing the skeleton with highest posterior
model probability at each allocation. It also defines an Occam-window variant
that retains models whose posterior weight divided by the largest weight
exceeds a threshold. These are source-backed scientific additions for a later
batch; the short MD Anderson method PDF describes averaging only.
For reproducing the JASA simulation tables, its stated alpha standard
deviation is 2, whereas the short method guide uses variance 2. Pass the
appropriate explicit `prior_sd`; do not compare the tables using the package's
sqrt(2) default and attribute differences solely to random sampling.

The Pan/Yuan paper's indexed full text supplies the Lee/Cheung indifference
interval recursion and defines nonequivalence as one minus average regression
R-squared for the log skeleton vectors:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5026535/

Direct page access currently shows a browser challenge. Its indexed equation
for Q contains an inconsistent second equality, so verify the PDF or author
code before implementing that expression. Also establish whether regressions
include an intercept. Candidate-set ranking alone is not the paper's full
calibration: it subsequently compares simulation-based correct-selection rates
across representative scenarios. Use bounded work and report Monte Carlo
uncertainty; do not label the highest-Q set as the statistically optimal set.
