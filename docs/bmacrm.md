# Bayesian model-averaged CRM

`fit_bmacrm` implements the power-model posterior used by BMA-CRM and the
single-skeleton CRM case. It supports **fully observed binary toxicity counts**.
It is distinct from this package's `fit_bcrm`, which uses the Goodman logistic
model and a uniform slope prior.

```python
from mdanderson_stats import fit_bmacrm

fit = fit_bmacrm(
    [
        [0.10, 0.21, 0.24, 0.30, 0.45],
        [0.15, 0.26, 0.29, 0.35, 0.50],
        [0.20, 0.31, 0.34, 0.40, 0.55],
    ],
    events=[0, 1, 2, 0, 0],
    subjects=[3, 6, 6, 0, 0],
    target=0.30,
)
print(fit.posterior_model_weights)
print(fit.dose_mean)
print(fit.overdose_probability)
```

For model `k`, the dose probability is `p[k, j] ** exp(alpha)` and alpha has a
zero-mean normal prior. The default prior variance is two. Each model is fitted
separately, then its evidence updates the prior model weight. Dose means and
overdose probabilities average across the resulting model probabilities.
Optional [BMS and Occam-window aggregation](crm-model-selection.md) select the
best-supported model or average within a retained subset. The default remains BMA.

The inputs `p` are **prior medians**, as in BMA-CRM 2.2.4 and CRM Suite 1.0.0.
They are not converted from prior means. This follows the current
[version notes](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/81),
which supersede the elicitation convention in the
[2009 method description](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BMACRM/BMA-CRM_Description.pdf).

`model_log_evidence` includes binomial coefficients for grouped counts. These
common factors cancel when calculating posterior model weights. The model
weights default to equal values; `model_prior` accepts nonnegative relative
weights, including zero. `input_model_prior` preserves these original relative
weights for later refits, while `prior_model_weights` contains normalized
probabilities. Use the original inputs for refitting: very unequal finite
weights can underflow to zero during normalization. `alpha_mean` and `alpha_sd`
describe each conditional model posterior, not a single shared alpha
distribution. `target` defines the event `toxicity probability > target` for
every returned overdose probability.

The implementation uses stable log likelihoods and adaptive integration over
the entire normal prior support, centered and scaled at each posterior mode.
It does not replace the normal prior by a fixed finite interval. Arrays in the
result are immutable. Integration failures and exhaustion of the shared
`max_evaluations` budget raise errors rather than returning partial results.
`integration_error` reports a quadrature diagnostic relative to each model's
normalizing integral; it is not a posterior uncertainty interval. Probabilities
smaller than floating-point precision can round to zero or one.

Inputs allow one to five models, one to twenty doses, nondecreasing skeletons
strictly inside `(0, 1)`, and at most 10,000 fully observed patients. Counts must
be nonnegative integers with events no greater than subjects. A one-dimensional
skeleton requests ordinary CRM. The wider probability/count ranges, support
for one dose, and configurable `prior_sd` in `[0.001, 10]` are Python extensions
to the desktop interface. Work is serial, with a small quadrature cache and a
maximum of 200,000 numerical evaluations per fit.

## Complete-outcome dose decisions

```python
from mdanderson_stats import bmacrm_decision

next_cohort = bmacrm_decision(fit, current_dose=2)
final_selection = bmacrm_decision(fit, final=True)
print(next_cohort.action, next_cohort.dose)
print(final_selection.dose, final_selection.high_uncertainty)
```

Dose indices are zero-based. Call this helper only when all enrolled patients'
outcomes are known. `current_dose` identifies the most recent cohort's dose;
it must have an observed subject for a noninitial, nonfinal decision. With no
patients, the first cohort starts at `starting_dose` (default zero). `final=True`
requests MTD selection explicitly; the helper does not infer trial completion
from a maximum sample size or enforce a cohort schedule.

The implemented policies follow
[CRM Suite Appendix II](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/BMA-CRMSimulatorHelp.pdf):

| Step | Behavior |
| --- | --- |
| Safety | Stop if the lowest-dose mixture overdose probability exceeds `safety_cutoff` (default 0.9). Setting it to one disables this rule. |
| No skipping | Consider doses only through the first untried level; levels below the configured starting dose count as tried. |
| Next cohort | Choose the posterior mean nearest target, then cap escalation at the first current or intervening level whose observed toxicity rate exceeds target. |
| Final MTD | Start from the nearest eligible level. If it has fewer than three patients, use the highest lower level with at least three; if none exists, retain it and set `high_uncertainty=True`. |

Distance ties within floating-point roundoff choose the lower dose, an explicit
Python convention (eight machine epsilons of absolute and relative tolerance).
The raw-rate escalation restriction is not used for final MTD selection.
The result includes the unconstrained and no-skip recommendations, the safety
probability, the final action/dose, and an explanation. A safety stop has
`dose=None`; a final selection has `action="select_mtd"`. The uncertainty flag
is essential when the fallback returns a level with little or no observation.

## Evidence and coverage

The independent base-R program `tools/reference_bmacrm.R` integrates the same
statistical model directly in alpha using R's binomial density and scalar
quadrature. Its fixtures cover ten scenarios: the prior, mixed outcomes,
unequal/zero model weights, all-toxic and no-toxic samples, ordinary CRM,
10,000-patient concentrated and boundary cases, extreme skeleton values, and
a tiny prior model weight rescued by strongly informative data.
The skeleton in the first eight scenarios comes from the official guide;
the patient counts are synthetic audit cases, not native simulation output.

This is partial coverage of catalog entries 81 and 132. [DA-CRM](dacrm.md)
handles delayed outcomes, and [calendar decisions](crm-conduct.md) add bounded
pending-outcome look-ahead and time-specific record replay.
[Trial simulation](crm-simulation.md) provides cohort scheduling and operating
characteristics. Native file/report workflows and older-version conduct
differences remain separate work. Online entry 133 now has partial coverage
through [model selection](crm-model-selection.md); automatic skeleton calibration
and hidden native conventions remain open. No Windows executable parity or
random-seed parity is claimed.
Original programs and manuals are not bundled. See [source provenance](bmacrm-sources.json).
