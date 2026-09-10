# TITE-BOIN imputation and interim conduct

Catalog entry **129**, [TITE-BOIN](https://biostatistics.mdanderson.org/shinyapps/TITE-BOIN/),
is partially implemented. The app snapshot is version 2.5.4.0, updated November 20,
2025. [Provenance](tite-boin-sources.json) records its guides and the authors'
mathematical appendix to *Time-to-event Bayesian Optimal Interval Design to
Accelerate Phase I Trials*, DOI 10.1158/1078-0432.CCR-18-0246.

The implementation provides vectorized single-mean imputation, ordinary follow-up
thresholds, and interim dose decisions. Calendar replay/simulation, Rolling 6
comparison, optional 3+3 modifications, flowcharts and integrated protocol reports
remain pending. Final selection is available through `BOINDesign.select_mtd` once
all outcomes are ascertained.

## Imputation and follow-up thresholds

```python
from mdanderson_stats import BOINDesign, tite_boin_estimate, tite_boin_decision

design = BOINDesign(target=0.3)
summary = tite_boin_estimate(design, patients=3, toxicities=1, pending=1, stft=0.9)
assert summary.move == 0
print(summary.estimated_rate, summary.deescalate_stft)
```

For enrolled count `n`, observed DLT count `y`, pending count `c`, and standardized
pending follow-up `S`, the observed-outcome posterior has parameters
`a=y+target/2`, `b=n-c-y+1-target/2`. The imputed DLT count is
`y+(a/b)*(c-S)` and the decision statistic is that count divided by `n`.
It is a conservative approximation and can exceed one; it is not clipped or
presented as a fitted probability. With no pending data it reduces to `y/n`.

The result includes the posterior mean, imputed count, rate, ordinary move
(`+1`, `0`, `-1`), and thresholds `c-(n*boundary-y)/(a/b)`. Escalation additionally
requires `y/n < target`; de-escalation requires `y/n > target`. A prohibited
escalation threshold is `inf`, and a prohibited de-escalation threshold is `-inf`.
Thresholds outside `[0,c]` are retained: they describe decisions that always or
never apply over feasible follow-up. The thresholds omit accrual and safety gates.
Floating thresholds are numerical approximations; `move` uses the unrounded
imputed rate. Do not use displayed two-decimal thresholds for borderline decisions.

Counts and follow-up broadcast across scenarios. Each scenario accepts 1–200
patients, `y+c<=n`, and `0<=S<=c`. `S` sums pending follow-up/window ratios under
a uniform timing prior. For informative three-trimester masses, sum the existing
`toxicity_followup_weights` instead; these are conditional timing probabilities,
not overall toxicity probabilities.

## Interim conduct

```python
# One pending patient has completed 90% of the assessment window.
step = tite_boin_decision(design, [3, 0], [1, 0], [[81], []], 1, 90)
assert step.action == "stay"
assert step.next_dose == 1

# The current app requires at least 51% ascertained outcomes by default.
waiting = tite_boin_decision(design, [6, 0], [0, 0], [[45, 45, 45], []], 1, 90)
assert waiting.action == "suspend_pending"
```

Dose numbers are one-based. Patients include pending outcomes, while toxicities
include only observed DLTs. Supply one pending-time vector per dose, with times in
`[0,window)`. Returned diagnostics include per-dose pending counts, weighted STFT,
safety probabilities/exclusions, and the current-dose imputation.

The current app's defaults are `minimum_complete_fraction=0.51` and
`minimum_pending_followup=0.25`. Conduct first applies enrolled-count beta-binomial
overdose control and retained exclusions. Otherwise it suspends if the proportion
of ascertained outcomes is too low, except when the observed DLT rate already
meets the de-escalation cutoff. An actual escalation waits if the shortest pending
follow-up/window ratio is below the specified minimum. This second gate uses
elapsed time, even when the imputation has informative timing weights.

Observed DLTs count as ascertained outcomes. The allowed completion fraction is
`[0.25,1]`; the minimum follow-up fraction is `[0,1]`. The historical 50% rule can
be selected explicitly with `minimum_complete_fraction=0.5`, and the additional
follow-up gate disabled with `minimum_pending_followup=0`. These settings matter:
the appendix's older table permits some assignments the current defaults suspend.

Safety can override suspension. Precision stopping requires that the actual next
dose would be unchanged, consistent with ordinary BOIN. Physical dose limits and
excluded higher doses can make the assignment stay. Pass the returned `eliminated`
mask to subsequent decisions to retain earlier exclusions. Wait for all pending
outcomes before final MTD selection, including after precision stopping.

The optional BOIN 1/3 and 2/6 modifications currently raise `NotImplementedError`
in these TITE functions rather than silently applying a complete-data rule to
pending outcomes. Exact application-output equivalence has not been established.

## Validation

Twelve escalation/de-escalation thresholds agree with the appendix's rounded
Table S1. Independent rational arithmetic checks imputation for all valid
`n<=12` count combinations at a specified follow-up fraction. Complete-data
conduct agrees with ordinary BOIN, including safety. Focused checks cover both
suspension gates, the observed-toxicity exception, precision stopping, informative
weights, coherence despite inflated imputation, and time rescaling by `1e-200`.
