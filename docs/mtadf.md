# MTADF: optimal biological dose by isotonic regression

`mtadf_decision` implements the isotonic method in Zang, Lee and Yuan's
[2014 paper](https://odin.mdacc.tmc.edu/~yyuan/Software/TargetAgent/OBD_rev.pdf).
It seeks the lowest dose with the highest estimated efficacy among admissible
observed doses. The current [MTADF application](https://biostatistics.mdanderson.org/shinyapps/MTADF/)
identifies this method as its computational basis. This independent Python
implementation exposes its assumptions and does not claim native app parity.

```python
from mdanderson_stats import mtadf_decision, mtadf_toxicity_prior

prior = mtadf_toxicity_prior(toxicity_limit=.30, safety_cutoff=.80)
counts = dict(subjects=[3, 6, 3, 0], toxicities=[0, 1, 1, 0],
              responses=[0, 4, 1, 0])
interim = mtadf_decision(**counts, current_dose=2, prior=prior)
final = mtadf_decision(**counts, final=True, prior=prior)
print(interim.dose, final.dose)  # zero-based dose indices
```

All outcomes are binary and fully observed. The toxicity and response counts
may overlap: one patient can experience both. The decision calculation uses
their marginal counts and does not require a joint contingency table. Counts
at each dose must not exceed enrollment; unobserved doses have zero counts.
The decision result retains raw and pooled overdose probabilities, admissibility,
fitted efficacy and an action/reason. Actions are `start`, `treat`, `select_obd`
or `stop`; a stop has `dose=None`. Result arrays are read-only. Decisions accept
1–20 dose levels and at most 10,000 total subjects. Final selection requires
observed data; an interim decision after enrollment requires an observed
`current_dose`. The `split` and `peak` fields describe the efficacy fit before
safety filtering; the recommended `dose` also accounts for admissibility.

## Statistical choices

Each dose has an independent beta-binomial toxicity model. With prior
`MTADFPrior(alpha, beta)`, its posterior is
`Beta(alpha + toxicities, beta + subjects - toxicities)`. The complementary
beta CDF gives the overdose probability above `toxicity_limit`. Increasing
PAVA with equal dose weights pools these probabilities. A dose is admissible
only when its adjusted probability is strictly below `safety_cutoff`.

The default common prior is elicited once from

```
alpha + beta = concentration
BetaCDF(toxicity_limit; alpha, beta) = 1 - safety_cutoff + margin
```

Defaults are concentration `.5` and margin `.05`, so the prior overdose
probability is `.75` with the default `.8` safety cutoff. An explicitly supplied
prior is used as given; it can rule out all doses before enrollment.
Calibration requires `0 < margin < safety_cutoff` and concentration in `(0, 100]`.
For a cutoff at or below `.05`, supply a smaller margin when eliciting the prior
and pass that prior to the decision function. Numerically unrepresentable
calibrations raise an error.

`double_sided_isotonic` enumerates a split after every input dose. It fits an
increasing left segment and a decreasing right segment by weighted PAVA, then
chooses the smallest **unweighted** sum of squared deviations from the input
rates. Patient counts supply the PAVA weights in `mtadf_decision`; the standalone
fit defaults to equal weights. These two weighting choices follow the paper's
example and displayed criterion. The split index is distinct from the actual
peak, which may occur on either side of the split. Exact score ties choose the
first split; efficacy plateaus favor the lowest dose.

Only observed dose rates enter the efficacy fit. Their original dose indices
are preserved, including gaps in the observed sequence. Untried efficacy is
undefined and cannot win final selection. Interim assignment normally moves
one level toward the observed admissible efficacy peak. When the current dose
is both that peak and the highest tried dose, it explores the next higher dose
if admissible. If the current dose is inadmissible, safety takes priority and
assignment drops directly to the highest admissible dose; this can exceed a
one-level move. With no admissible dose, the recommendation is to stop.

## Serial simulation

`simulate_mtadf` generates toxicity and efficacy independently for each complete
cohort, updates the dose decision, and returns compact trial and aggregate
summaries. Independence is an explicit data-generation assumption, not a
requirement for the marginal decision calculation.

```python
from mdanderson_stats import simulate_mtadf

simulation = simulate_mtadf(
    true_toxicity=[.05, .10, .25, .45],
    true_efficacy=[.10, .35, .55, .50],
    cohorts=4,
    cohort_size=3,
    trials=20,
    rng=914,
)
print(simulation.selection_probability, simulation.no_selection_probability)
```

Small examples illustrate usage; they do not provide precise operating
characteristics. Trials run serially and retain dose-level counts rather than
patient-level simulation tensors. The common prior is calibrated once per
simulation. Work limits are checked before sampling, and exceeding a limit
raises instead of returning a partially simulated result. No-selection and
early-stopping summaries are separate: a safety decision at the planned sample
size is not early stopping. A prior that prevents enrollment produces zero
patients, no selection and an early stop. With an explicit prior, `margin` and
`concentration` do not recalibrate it.

Per-trial `patients`, `toxicities` and `responses` have shape `(trials, doses)`;
`selected_dose` uses `-1` for no selection. Selection probabilities use all
trials as their denominator and sum to one together with the no-selection
probability. The result includes their binomial Monte Carlo standard errors,
early-stop summaries, per-dose means and per-trial stopping reasons. At most
10,000 trials and 1,000 planned patients per trial are allowed. The default work
budget is 200,000 decisions, including each initial review; it can be increased
explicitly up to 1,000,000 with `max_total_decisions`.

## Validation and remaining coverage

Independent base-R references use the min-max characterization of weighted
isotonic regression and direct-alpha prior calibration. They provide a separate
calculation for curve fits, split scores, beta priors and posterior tails.
Focused decision and small simulation checks cover the dose policy and count
conservation. See the [audit](../research/mtadf-audit.md) and
[source record](mtadf-sources.json).

Catalog entry 114 remains partial. The app's visible cutoff description and the
rendered author R program use an inclusive boundary, while the paper prints a
strict one. The rendered program also uses fixed prior calibration and retains
at least the lowest dose. This implementation follows the documented paper
policy and allows a complete safety stop. Hidden app settings, native output,
full raw-source comparison and random-sequence equivalence remain unverified.
The paper's global and local logistic designs are separate methods and are
outside this isotonic implementation.
