# PoPdesign

The posterior predictive phase-I design is listed by
[MD Anderson](https://biostatistics.mdanderson.org/shinyapps/PoPdesign/) and
described by Fu, Zhou and Lee (2025),
[JASA, DOI 10.1080/01621459.2025.2484044](https://pmc.ncbi.nlm.nih.gov/articles/PMC12700609/).
The Python calculations implement the mathematical design independently;
the CRAN PoPdesign 1.1.0 routines supply numerical references. No R dependency
or vendor source is bundled. [Source provenance](pop-design-sources.json)
records the inspected versions and hashes.

```python
from mdanderson_stats import PoPDesign, simulate_pop

design = PoPDesign(target=0.25)
table = design.boundaries(30, cohort_size=3)
decision = design.decision(1, [3, 0, 0], [0, 0, 0])
assert decision.next_dose == 2

result = simulate_pop(
    design, [0.1, 0.25, 0.45], total_patients=30,
    cohort_size=3, trials=1000, titration=True, seed=175,
)
print(result.selection_probability)  # [no selection, dose 1, dose 2, dose 3]
print(result.selection_mcse)
```

Doses are one-based; `None` denotes no admissible selection in the selector,
and simulation records use zero for that outcome. Boundary rows use `-1`
for an impossible low-count action and `n+1` for an impossible high-count
action. Cohort-multiple boundary tables require a divisible maximum count;
the full table includes every integer count.

## Predictive Bayes factor

At a dose with `y` DLTs among `n` evaluable patients, let `q=(y+1)/(n+2)`.
The log predictive Bayes factor is

```text
log_PrBF = 1 + y*log(target/q) + (n-y)*log((1-target)/(1-q))
```

The binomial coefficients cancel. This compares the target likelihood with
the likelihood at the posterior mean; it is not a beta-binomial predictive
mass. Log arithmetic avoids underflow in decisions for extreme counts.
Defaults are transition cutoff `C=2.5` and exclusion cutoff `E=5/24`.
The supported target range is [0.05, 0.6], with `1 < C < e` and `0 < E < C`.
The paper derives `C=(b2-b3)/b1` and `E=b3/(1-b1)` from decision losses;
its example uses `(b1,b2,b3)=(0.2,2/3,1/6)`.

When PrBF is strictly below C, move toward the target according to the
observed rate; otherwise stay. When it is strictly below E, exclude the
current and lower doses if the rate is below target, or current and higher
doses if above target. Exclusions persist for allocation. There is no dose
skipping, and allocation ends when every dose is excluded. Zero observations
provide no reason to transition. Threshold equality does not trigger a move
or exclusion, following the package's executable rules.

## MTD estimation

The working Beta(0.05,0.05) posterior means for treated doses are fitted by
increasing isotonic regression, weighted by inverse posterior variance.
Native reporting adds `1e-10 * treated_rank` to break flat fitted estimates.
The closest admissible fitted value determines the MTD, with the highest
dose selected on an exact remaining distance tie. Untreated doses are never
selected, and safety masks are mapped using original dose labels.

Final safety uses a separate Beta(1,1) posterior. A tail probability above
0.95 eliminates the dose and all higher doses from selection. The paper
requires at least three observations at the triggering dose, although the
CRAN code omits that guard. The Python default follows the paper; setting
`safety_min_patients=0` reproduces the package's guard. Allocation exclusions
and final selection safety are distinct: after enrollment ends, including
early termination, final estimation uses the observed data and this safety
rule. Early stopping therefore does not necessarily mean no MTD is selected.
The `early_stop` flag records all-dose exclusion, including a trigger on the
final planned cohort; it does not by itself imply reduced enrollment.

## Simulation semantics

Complete binary outcomes are available before the next cohort is assigned.
Optional accelerated titration treats individuals, escalating after each
non-DLT until the highest dose or first DLT. The first DLT invokes the
transition rule and ends titration. Exclusion decisions begin with ordinary
cohorts. The final cohort can be smaller than the requested size.

True toxicity probabilities must be nondecreasing. Sequential independent
draws at the assigned dose have the same trial law as the native monotone
potential-outcome construction, since each patient's outcome is used at only
one dose. Random streams differ from R. This avoids allocating every
patient's hypothetical outcomes at every dose.

Python resource limits allow up to 1,000 planned patients, 2–100 doses and
cohort sizes 1–4, with `trials * planned_patients <= 100000` and
`trials * doses <= 100000` checked before allocation. Boundaries are computed
once per simulation call. Read-only per-trial counts and selections are
retained, together with selection, exclusion-stop and risk probabilities and
their Monte Carlo standard errors. These are sampling errors, not uncertainty
intervals for the underlying dose-toxicity curve.

Underdose/overdose risk uses a strict comparison against
`risk_cutoff * planned_sample_size`, and defines the true MTD as the first
dose closest to target. The supplied risk cutoff is honored; the native
wrapper accidentally omits it when calling its inner simulation routine.

## Validation and native differences

[reference_pop_design.R](../tools/reference_pop_design.R) runs inspected
native routines directly without installing the R package. It saves 360
boundary rows across six targets, predictive Bayes factors, and 30 MTD
selection cases. Small two-dose trials are also enumerated over all cohort
DLT counts, giving exact selection probabilities, expected counts, early
stopping and allocation risks for comparison with simulation.

The 1,000-trial example above took approximately 0.10 seconds and 109 MiB peak
process RSS on the development machine, with zero process swaps. Its selection
proportions were `[0, 0.166, 0.711, 0.123]`. This is a seeded Python example,
not an R random-stream match or a precise operating-characteristic estimate.

The CRAN selector incorrectly indexes a compressed vector of treated doses
with indices from the full dose set. For example, three patients with zero
DLTs at dose 1 and no patients at two higher doses produce `NA` in R. The
Python selector correctly retains dose 1. The exact reference enumeration
uses compact native inputs and maps labels back to avoid that defect.
Standalone and simulation selection share one tie rule in Python; native
functions disagree on exact ties. Native HTML/Word protocol generation,
plots, scenario-file handling and report downloads remain open. Catalog
entry 175 remains partial.
