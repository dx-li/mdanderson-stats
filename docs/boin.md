# Bayesian optimal interval dose finding

Catalog entry **120**, the [BOIN application](https://biostatistics.mdanderson.org/shinyapps/BOIN/),
is **partially implemented**: single-agent local BOIN boundaries, cohort decisions,
overdose elimination, final MTD selection and fixed-cohort simulation are available.
Accelerated titration, direct boundary-to-probability inversion, the app's 3+3
comparators, protocol generation and animation remain pending. Desktop entry 99
and BOIN combination/time-to-event variants are separate, unaudited entries.

The application was inspected at version **3.0.20.0**, updated September 4, 2026.
Its technical PDFs and the [BOIN R-package paper](https://doi.org/10.18637/jss.v094.i13)
provide the statistical specification. File hashes and native-reference details
are recorded in [provenance](boin-sources.json). This is an independent Python
implementation; R is used only to generate validation fixtures.

```python
from mdanderson_stats import BOINDesign, simulate_boin

design = BOINDesign(target=0.3)
table = design.boundary_table(max_patients=30)
assert table.escalate_max[2] == 0  # at three patients, escalate with zero DLTs
assert table.deescalate_min[2] == 2
assert table.eliminate_min[2] == 3

next_step = design.next_dose([3, 6, 0], [0, 2, 0], current_dose=2)
assert next_step.next_dose == 2
mtd = design.select_mtd([3, 3, 15, 9, 0], [0, 0, 4, 4, 0])
assert mtd.dose == 3

simulation = simulate_boin(design, [0.05, 0.15, 0.3, 0.45, 0.6], trials=1000, rng=6)
print(simulation.selection_probability)  # no selection, then doses 1 through 5
print(simulation.mean_patients)
```

## Boundaries and conduct

Inputs are complete, evaluable patient and dose-limiting toxicity (DLT) counts.
Dose indices are **one-based**. Untreated doses have zero counts, not synthetic
outcomes. Pending outcomes are outside this method's scope.

The default indifference probabilities are `safe_probability=0.6*target` and
`toxic_probability=1.4*target`. Custom values must satisfy
`0 < safe < target < toxic < 1`, with target in `[.05,.6]`. Log-likelihood crossings
use `log1p` differences to retain precision for nearby probabilities. Escalate at
observed DLT rate **at most** the lower boundary; deescalate at rate **at least**
the upper boundary; otherwise stay. At the end of the dose range, an unavailable
move becomes stay. Decisions never escalate into an eliminated dose.

For safety, use the uniform Beta(1,1) prior and eliminate a dose and all higher
doses when its posterior probability of toxicity exceeding target is **strictly
greater** than `elimination_probability` (default .95), with at least three
patients treated there. `extra_safe=True` applies the lower cutoff
`elimination_probability-safety_offset` at the lowest dose (offset .05 by default).
Eliminating the lowest dose stops the trial with no MTD. **Pass the returned
`eliminated` mask into later decisions and selection** to retain earlier exclusions.
The simulator does this automatically.

Table rows represent patient counts 1 through `max_patients`; escalation is an
inclusive maximum DLT count, while deescalation and elimination are inclusive
minimum counts. An elimination cutoff of `n+1` means elimination is impossible at
that sample size. Safety rules take precedence over the ordinary rate boundaries.

Optional source modifications are `stay_at_one_of_three=True` for targets in
[.25,.279], and `deescalate_at_two_of_six=True` for targets in [.28,.33]. With
`early_stop_patients=m`, stop for precision once the current dose has at least m
patients **and the underlying rate rule says stay**. This option is off by default.
A stopped decision has `next_dose=None`; final selection is a separate calculation.

## Final estimation and MTD selection

Following the R software's estimation convention, each treated dose has a
Beta(.05,.05) working prior. Its posterior mean is `(y+.05)/(n+.1)` and its variance
is `(y+.05)*(n-y+.05)/((n+.1)**2*(n+1.1))`. The mean is fit by increasing isotonic
regression with inverse-variance weights, using SciPy's compiled pool-adjacent-
violators algorithm. Untreated doses have NaN reporting estimates.

`isotonic_mean` fits all treated doses for reporting. `selection_mean` refits only
treated, noneliminated doses. Select the admissible fitted mean closest to target;
for tied distances choose the highest dose if all tied means are below target,
otherwise the lowest dose. `bound_mtd=True` additionally requires a fitted mean
at or below the deescalation boundary. No admissible dose yields `dose=None`.

`isotonic_interval` applies the same weighted isotonic transformation to the
individual .025 and .975 beta quantiles. It reproduces a source reporting
convention; it is **not a joint 95% posterior credible band**. The unweighted
isotonic `report_overdose_probability` uses the weak working prior, whereas
`safety_overdose_probability` uses Beta(1,1) without isotonic smoothing. These
probabilities have different purposes and should not be interchanged.

## Simulation and source differences

`simulate_boin` generates independent binomial DLT totals at each cohort, with
all outcomes available before the next decision. Defaults are 10 cohorts of 3,
1,000 trials, starting at dose 1. Returned trial-level patient/toxicity counts
allow further operating-characteristic summaries. Selection index zero means no
MTD. Selection probabilities include a no-selection bin followed by all doses,
and `selection_mcse` reports marginal Monte Carlo standard errors. Stop reasons
distinguish safety, precision and the enrollment cap. Probabilities need not be
monotone, allowing evaluation under misspecified dose ordering. NumPy random
streams do not reproduce R draws for equal seeds.

The older Probability PDF prints a strict upper rate inequality; the published
paper and R implementation use the inclusive rule implemented here. The older
Guide says more than three patients for safety, whereas the paper/native rules
use at least three. The latest standalone modification PDF broadens the 1/3
option's target range beyond the older Guide's .25 example. The latest sample-size
PDF requires stay for precision stopping; R 2.7.2 instead stops unconditionally at
its sample threshold. Native simulation comparisons therefore disable that stop.
The R selection implementation adds a tiny artificial trend to break ties;
Python resolves ties explicitly without perturbing fitted probabilities, so
pathological near-ties or values exactly on a selection bound may differ.

Validation checks six published boundary pairs, the full .30-target cohort table,
exact rational beta/binomial safety identities, 64 original R MTD selections and
reported estimates, and a 10,000-trial original R operating-characteristic run.
Deterministic safe/unsafe trial paths and the exact one-cohort binomial law check
simulation independently. The reference runner requires BOIN 2.7.2 and its Iso
dependency; neither is a Python runtime dependency.
