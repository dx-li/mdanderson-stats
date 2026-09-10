# iBOIN: informative Bayesian optimal interval design

`IBOINDesign` implements prior elicitation, dose-specific decision boundaries and
complete-outcome dose assignment for MD Anderson catalog entry **145**, the
[iBOIN app](https://biostatistics.mdanderson.org/shinyapps/iBOIN/).
The live app identified PID 1040, version V1.6.3.0 when inspected September 10, 2026.
Its displayed update date was truncated to `11/20/202`; no year is inferred.

The mathematical source is Zhou, Lee, Wang, Bailey and Yuan,
*Incorporating historical information to improve phase I clinical trials*,
[published article](https://pubmed.ncbi.nlm.nih.gov/33793044/) and
[author manuscript](https://arxiv.org/abs/2004.12972), version 3 dated June 8, 2020.
The implementation uses manuscript equations (2.3) and (2.5), its Table 1,
and the [native user guide](https://biostatistics.mdanderson.org/shinyapps/iBOIN/iBOINGuide.pdf).
[Source provenance](iboin-sources.json) records downloaded artifacts;
original PDFs and app files are not redistributed.

## Historical prior

Supply the ordered toxicity skeleton `q` and a matching nonnegative **integer**
prior effective sample size `m` for each dose. The implementation supports 2–100
doses and ESS up to 10,000 per dose. These are computational limits, not a
recommendation to use a large historical sample. Fractional ESS is not silently
rounded: equation (2.5) averages a binomial experiment with integer sample size.

The three hypotheses are `[target, safe, toxic]`. For each dose, average the
normalized three-hypothesis likelihood over every possible historical DLT count
`X ~ Binomial(m, q)`. This produces the prior hypothesis probabilities. It is
not equivalent to simply adding historical successes and failures to a beta prior.
Likelihood normalization and averaging use log-sum-exp; `log_hypothesis_probability`
retains very small probabilities whose ordinary exponentials may underflow.

ESS zero gives equal hypothesis probabilities and ordinary BOIN decisions.
Default safe and toxic alternatives are .6 and 1.4 times the target. Target and
alternative validation follows the package's existing `BOINDesign`.

## Robust historical borrowing

Set `robust_prior=True` to apply the rule in the app's dedicated
[robust-prior help](https://biostatistics.mdanderson.org/shinyapps/iBOIN/iBOINRobust.pdf).
If the one-based prior MTD index is at least half the number of doses, historical
ESS is set to zero for doses strictly above that index. Otherwise all supplied
ESS values remain in use. The dedicated help explicitly includes equality at the
midpoint; the main guide omits this equality case. The default is `False`.

This option requires exactly one skeleton probability equal to the target,
matching the help's definition of the prior MTD. Ambiguous or missing matches
raise instead of assuming a nearest-dose or tie-breaking convention.
`prior_ess` retains the supplied values; read-only `effective_prior_ess` shows the
values actually used, whether or not robust borrowing is enabled.

```python
from mdanderson_stats import IBOINDesign

robust = IBOINDesign(
    [0.10, 0.19, 0.30, 0.42, 0.54], [2, 3, 4, 2, 2], target=0.30, robust_prior=True
)
assert robust.effective_prior_ess.tolist() == [2, 3, 4, 0, 0]
```

The two worked help examples are checked, including the even-dose midpoint case.
A live robust-prior table confirms that higher-dose boundaries revert to ordinary
BOIN for the default five-dose skeleton while the first three remain unchanged.

## Boundaries and conduct

`boundaries(patients)` returns dose-by-sample-size matrices. Patient counts must
be in 1–100,000, with at most 10,000 requested counts. Increasing current sample
size diminishes the historical prior's contribution to the boundaries.

**The live application's small-sample behavior differs from the manuscript's
printed truncation.** The manuscript prints a lower truncation of zero for the
escalation boundary. The app instead reports escalation as unavailable for its
default doses 4 and 5 at one patient, and dose 5 at two patients. Untruncated
likelihood crossings reproduce these cells; clipping to zero would incorrectly
allow escalation after zero DLTs. This implementation retains the untruncated
crossings, including values outside [0,1].

`escalate_max=-1` means escalation is impossible; `deescalate_min=n+1` means
de-escalation is impossible. Other count thresholds lie in 0..n. Comparisons are
inclusive, following the app and manuscript Table 1. The `eliminate_min` vector
is common to all doses, with `n+1` representing no possible elimination.
Overlapping continuous boundaries raise an error because an authoritative
conflict-resolution rule has not been established for those settings.

`next_dose` uses one-based indices and complete evaluated outcomes. Safety
elimination uses the uniform Beta(1,1) posterior independently of historical
information: with at least three evaluated patients, eliminate a dose and all
higher doses when its posterior overdose probability strictly exceeds the cutoff
(default .95). Elimination of dose 1 stops the trial. Carry `eliminated` into the
next decision to retain exclusions. An unavailable outward move becomes stay;
if the current dose is excluded, move to the highest remaining admissible dose.
The result uses the existing `BOINDecision` container. Returned arrays are read-only.

```python
from mdanderson_stats import IBOINDesign

design = IBOINDesign(
    skeleton=[0.10, 0.19, 0.30, 0.42, 0.54],
    prior_ess=[3, 3, 3, 3, 3],
    target=0.30,
)
table = design.boundaries([3, 6, 9, 12])
assert table.escalate_max[0, 0] == 1
step = design.next_dose([3, 0, 0, 0, 0], [1, 0, 0, 0, 0], current_dose=1)
assert step.action == "escalate" and step.next_dose == 2
```

## Validation and remaining scope

Focused tests check all 100 published Table 1 escalation/de-escalation
cells, all 120 live default escalation/de-escalation cells and safety thresholds,
an independent direct finite-sum prior calculation, extreme ESS/log probabilities,
reduction to ordinary BOIN at ESS zero, and prior-independent safety stopping.

**Catalog status remains partial.** Accelerated titration,
optional extra safety/precision stopping, final MTD estimation with optional prior
borrowing, operating-characteristic simulation and native report generation are
not yet implemented. The guide contains additional conventions for these options;
ordinary BOIN final selection is not presented as a reproduction of all iBOIN
selection options.
