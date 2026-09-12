# TITE-BOIN12 approximate-likelihood methods

Catalog 152 is **partial**. This port provides binary-endpoint posterior
calculations and interim dose decisions under the approximate-likelihood (AL)
method. The Bayesian data-augmentation route, categorical endpoints, optional
3+3 run-in, integrated calendar simulation, native file/report adapters and
final OBD procedure remain outstanding.

## Patient-level data

Use `BOIN12Design` for toxicity/efficacy limits, utilities, admissibility cutoffs
and dose-exploration settings. The new functions take one-based patient dose
labels, separate toxicity and efficacy outcomes, separate follow-up arrays,
positive assessment windows, and the total number of doses.

Each endpoint is coded `1` for an observed event, `0` for a completed
assessment without an event, and `-1` for pending. A pending endpoint must
have follow-up shorter than its window; a completed non-event must have
follow-up at least as long as its window. Inputs describe information already
available at the analysis time, not eventual outcomes. Toxicity and efficacy
can have different assessment states for the same patient.

Use the **same time unit** for each follow-up array and its window. The web
app's settings use months while its template labels follow-up in days; this
API makes no unverified conversion between them. Permanently unevaluable
endpoints may be represented as pending with follow-up frozen at their last
assessment, but this is still an imputation assumption and affects suspension.
Input sizes are bounded at 1,000 patients and 100 doses.

## AL posterior

For each endpoint at each dose, let `y` be observed events, `m` be observed
non-events, and `w` each pending patient's follow-up divided by the window.
The effective binomial likelihood has sample size
`ESS = y + m + sum(w)`. Its maximum-likelihood rate is `p = y / ESS`.
A pending patient's conditional event probability is
`p*(1-w) / ((1-p) + p*(1-w))`. The denominator form avoids subtracting nearly
equal values. The non-event probability is evaluated directly as well.

Observed endpoints remain exactly zero or one. Joint outcome probabilities
are products only over the pending dimensions, so fully observed joint
outcomes retain their actual association. Expected cells are returned in
BOIN12 order: no toxicity/response, no toxicity/no response, toxicity/response,
and toxicity/no response. Their utility-weighted sum divided by 100 is the
quasi-event count `x`. With `N` enrolled patients, the utility posterior is
`Beta(1+x, 1+N-x)`, including the ordinary `Beta(1,1)` prior at an untried dose.

Safety and activity use the **effective-binomial** marginal posterior
`Beta(1+y, 1+ESS-y)`. Safety is the upper toxicity tail; activity is the lower
efficacy tail. Admissibility requires both tails to be strictly below the
design cutoffs. Utility desirability is the posterior probability of exceeding
the BOIN12 benchmark, not the posterior mean utility.

If a treated dose has zero effective information for an endpoint, its MLE is
unidentified. The posterior function raises an error rather than supplying an
arbitrary estimate; interim conduct suspends. Untried-dose marginal rate
estimates are NaN, while their prior posterior probabilities remain available.

## Interim decisions

`tite_boin12_decision` suspends when either current-dose pending fraction is
**greater than** its configured maximum (default one half), so exactly one
half is allowed. Otherwise it uses the AL toxicity rate and utility posterior
in the existing BOIN12 neighboring-dose rules. Prior eliminations are combined
with current inadmissibility, and eliminated doses cannot receive assignments.
The existing design's exploration, stay-sample-size and precision-stop settings
are carried into this Python conduct policy.

The result reports the action, next dose, pending counts, elimination state
and posterior information. Suspension can occur before a posterior is
computed. This is a stateless analysis: retain and pass the returned elimination
mask at subsequent looks. Trial accrual and pending-outcome updates remain the
caller's responsibility.

## Example

```python
from mdanderson_stats import BOIN12Design, tite_boin12_decision

result = tite_boin12_decision(
    BOIN12Design(0.35, 0.25),
    doses=[1, 1],
    toxicity=[0, -1],
    efficacy=[1, -1],
    toxicity_followup=[1.0, 0.5],
    efficacy_followup=[2.0, 1.0],
    toxicity_window=1.0,
    efficacy_window=2.0,
    n_doses=3,
    current_dose=1,
)
```

One of two patients is pending for each endpoint, so the default suspension
threshold permits a decision. Retain `result.eliminated` for the next look.

## Source verification and remaining uncertainty

The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9199061/),
Zhou et al., *Statistics in Medicine* 41, 1918–1931 (2022), DOI
10.1002/sim.9337, gives the conditional imputation and quasi-binomial utility
formulas. It refers the marginal estimator derivation to supplement S7 and
the final selection details to S2. The supplement download returned an HTML
challenge; those sections have **not** been inspected.

The effective likelihood is corroborated by the cited Lin–Yuan method's
[author code](https://github.com/ruitaolin/TITE-MAD), which uses completed
assessments plus standardized pending follow-up for its local posterior.
Its separate toxicity-elimination rule uses enrolled counts; that rule is
not evidence of the hidden TITE-BOIN12 safety implementation.
An [author-published later PK extension](https://github.com/EugeneHao/PKBOIN-12)
corroborates the ESS/MLE and conditional-imputation calculations. Its PK-specific
selection and simulation rules are not used as substitutes for TITE-BOIN12.
The effective-binomial safety posterior is the explicit likelihood-based
choice in this implementation; exact equality to the hidden native backend's
pending-data safety calculation remains unverified. No native RNG or full
application parity is claimed. Source hashes and retrieval limitations are
recorded in [tite-boin12-sources.json](tite-boin12-sources.json).

`tools/reference_tite_boin12.R` independently calculates effective sizes,
conditional endpoint probabilities, nonadditive joint utilities and posterior
tails using base R for all four ascertainment patterns. Focused checks also
verify complete-outcome reduction, unit rescaling, suspension boundaries and
persistent elimination. These checks validate the declared AL model, not the
unavailable native backend.
