# Data-augmentation CRM for delayed toxicity

`fit_dacrm` fits the DA-CRM posterior with observed and pending patient outcomes.
It uses the power-model CRM for binary toxicity and a piecewise exponential
working model for time to toxicity. Pending outcomes are inferred jointly with
both sets of parameters; they are not dropped or treated as completed non-events.

```python
import numpy as np
from mdanderson_stats import DACRMPrior, fit_dacrm

prior = DACRMPrior(
    breaks=[0, 1, 2, 3],
    shape=[0.5, 0.75, 1],
    rate=[1, 1, 1],
)
fit = fit_dacrm(
    [0.1, 0.25, 0.5],
    doses=[0, 1, 1, 2, 2],
    outcomes=[0, 1, -1, -1, -1],
    times=[3, 0.5, 0, 0.5, 1],
    prior=prior,
    target=0.3,
    rng=np.random.default_rng(7302),
)
print(fit.dose_mean)
print(fit.overdose_probability)
print(fit.pending_probability)
print(fit.parameter_summary.split_rhat)
print(fit.dose_summary.batch_mean_mcse)
```

This example uses explicit illustrative gamma priors, not claimed desktop
defaults. The assessment window ends at the last `breaks` value; all times use
the same unit. Doses are zero-based indices into one nondecreasing prior-median
skeleton. Outcomes are `1` for observed DLT, `0` for a completed observation
without DLT, and `-1` for a pending outcome. `times` contains the DLT time for
observed events, the full window for completed non-events, and current follow-up
for pending patients. A reported event at time zero is allowed; an event at an
interior interval boundary belongs to the interval on its right, and the final
interval includes the assessment endpoint.

## Statistical model

For dose `d`, toxicity probability is `p[d] ** exp(alpha)`. Alpha has a normal
prior with mean zero and default variance two. Each interval hazard has an
independent gamma prior with the specified **shape and rate**, so its mean is
`shape / rate`.

The sampler alternates binary imputation, a Gaussian-prior elliptical-slice
update of alpha, and conjugate gamma updates of the hazards. An imputed future
DLT contributes survival exposure through current follow-up, but no observed
event count. Completed non-events contribute no exposure to this conditional
time-to-DLT model. The conditional probability draws for pending outcomes
provide a less noisy posterior summary than averaging imputed binary values.

These updates implement the working likelihood in Section 2.3 of
[Liu, Yin and Yuan (2013)](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/DA-CRM_Description.pdf).
No additional time-truncation normalization is inserted. Gaussian-prior
elliptical-slice sampling is an explicit Python implementation choice; native
random sequences and hidden MCMC settings are not reproduced.

## Source-based prior calibration

`dacrm_uniform_prior(window, intervals=9, dispersion=2)` constructs the paper's
midpoint-hazard prior for approximately uniform conditional event timing.
`dacrm_trimester_prior(window, probabilities, dispersion=...)` constructs the
current desktop guide's six-interval calibration from probabilities for each
third of the window, conditional on DLT. Its final cumulative probability is
0.99, matching the guide's finite-hazard approximation. Calibrations whose
preceding cumulative probability reaches 0.99 are rejected.

For both helpers, gamma shape is mean hazard divided by dispersion, and rate
is the reciprocal of dispersion. The trimester helper requires dispersion
explicitly; it does not infer an undocumented desktop default. Dispersion has
units of inverse time. When multiplying every time by a factor, divide the
dispersion by that factor to preserve the same prior on physical event timing.

```python
from mdanderson_stats import dacrm_uniform_prior, dacrm_trimester_prior

paper_prior = dacrm_uniform_prior(3)
trimester_prior = dacrm_trimester_prior(3, [0.05, 0.15, 0.80], dispersion=2)
```

These are different elicitation conventions. The second example uses the
paper's dispersion as an explicit choice, not a verified native default.

## Dose decisions

`dacrm_decision` applies a named rule set to an existing posterior. Both policies
move at most one dose level in either direction between cohorts, and resolve
numerical ties toward the lower dose. The default `policy="paper"` follows
Section 2.4, with a safety cutoff of 0.96 and unrestricted nearest-target final
selection. `starting_dose` defaults to zero and can be configured, as in the
paper's clinical example.

```python
from mdanderson_stats import dacrm_decision

decision = dacrm_decision(fit, current_dose=2)
desktop_rules = dacrm_decision(fit, current_dose=2, policy="crm_suite", minimum_observed=2)
print(desktop_rules.action, desktop_rules.dose, desktop_rules.explanation)
```

The `crm_suite` policy adds the guide's observed-outcome escalation gate,
raw-toxicity restriction and final MTD eligibility rules, with a default safety
cutoff of 0.9. `minimum_observed` must be supplied explicitly: zero disables the
gate, which applies only while at least one outcome remains pending.
Pending patients count toward treated totals, but not observed totals or
raw toxicity rates. Final selection applies the first-untried restriction and
the three-treated-patient fallback. The result records a high-uncertainty flag
if no suitable lower dose has three treated patients.

For this Python policy, an insufficient-observations wait takes precedence over
the raw-rate cap. The guide does not specify the order when both apply; native
branch-order parity is not claimed. An irregular supplied history with an
untried lower dose also blocks upward movement. Safety stopping precedes these
post-initial rules; a cutoff of one disables it.

This policy follows the newer CRM Suite guide. The
[older BMA-CRM Simulator guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BMACRM/BMA-CRMSimulatorHelp.pdf)
instead describes waiting after a DA safety signal until full-information CRM
can decide whether to stop. That older desktop behavior is not implemented by
the `crm_suite` profile; native behavior across versions is not claimed identical.

This helper uses the supplied posterior, including any Monte Carlo uncertainty.
It does not refit, determine cohort completion, advance patient follow-up, or
decide whether final assessment is appropriate. A final decision may therefore
use pending outcomes when the caller explicitly requests it. Retain and assess
the posterior diagnostics before interpreting a decision near a boundary.
The helper continues to apply the chosen DA rules when all outcomes are known;
it does not automatically switch to ordinary CRM. For that separate complete-data
workflow, use `fit_bmacrm` with one skeleton and `bmacrm_decision`, or the
[calendar decision router](crm-conduct.md), which switches inference methods
according to the outcomes visible in its snapshot.

## Output and numerical precision

The immutable fit retains alpha, hazard, dose-probability, and pending-probability
draws with leading `(chain, draw)` axes. `pending_indices` maps pending summaries
back to the patient input rows. `overdose_probability` estimates the probability
that each dose's toxicity exceeds `target`, using alpha thresholds to avoid
artificial ties when probabilities round to zero or one.

Chain summaries provide classical split R-hat and batch-means Monte Carlo
standard errors. These diagnostics estimate sampling precision; they are not
proofs of convergence, and the R-hat is not rank-normalized. Retained draws
support further assessment. Unlike the complete-data `fit_bmacrm` quadrature,
ordinary DA posterior summaries have Monte Carlo error. With no patients or
only pending patients at zero follow-up, direct prior sampling is used.

Work is serial and bounded before large arrays are allocated. The implementation
supports up to 200 patients, 20 doses, 20 hazard intervals, and two to four
chains, with explicit limits on retained cells, transitions, and likelihood
evaluations. Exhausting a numerical budget raises an error rather than returning
an incomplete posterior. Gamma shape and rate inputs must lie in `(0, 1e8]`,
and alpha's prior standard deviation must lie in `[0.001, 10]`; unrepresentable
numerical states raise errors.

## Validation and remaining coverage

The reference program `tools/reference_dacrm.R` enumerates every completion in
six small synthetic datasets. It integrates gamma hazards analytically and
alpha with independent base-R quadrature. It uses patient-level Bernoulli
likelihoods, avoiding incorrect binomial reweighting across completions.
Cases cover complete observations, early/late pending observations, prior-only
data with and without zero-time patients, and observed events at interval edges.
The paper's Table 3 posterior means independently check its published adjacent
dose-allocation sequence; focused policy cases distinguish pending and observed
counts, waiting, raw-rate restrictions and final selection.

This adds DA posterior inference to the partial implementations of BMA-CRM
Simulator and CRM Suite (catalog 81 and 132), together with paper and desktop
dose-decision policies. [Calendar decisions](crm-conduct.md) provide record
replay and automatic complete-data routing. [Trial simulation](crm-simulation.md)
adds cohort scheduling and operating characteristics. Native files/reports and
the older desktop safety-wait convention remain separate work.
The current desktop uses six hazard intervals; the paper's simulation study
uses nine. Online entry 133 has partial [model-selection coverage](crm-model-selection.md),
with native conventions and automatic skeleton calibration still open. See
[source provenance](dacrm-sources.json).
