# U-BOIN joint posterior and allocation

Catalog entry 142 is **partial**. These functions implement the complete-outcome
Dirichlet utility model, complete-outcome two-stage conduct and the three
allocation probability rules from
[Zhou, Lee and Yuan (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7927960/)
and the [official U-BOIN application](https://biostatistics.mdanderson.org/shinyapps/UBOIN/)
(V2.4.4.0, PID 1014). Source hashes are in [uboin-sources.json](uboin-sources.json).

```python
from mdanderson_stats import uboin_posterior, uboin_allocation

posterior = uboin_posterior(
    [[[2, 1], [5, 2]], [[3, 0], [4, 1]]],
    prior=[[.25, .25], [.25, .25]],
    utilities=[[30, 0], [100, 50]],
)
probabilities = uboin_allocation(
    posterior, eligible=[True, True], method="proportional"
)
```

Counts have shape `(dose, efficacy, toxicity)`, with both category axes in
ascending order: response and toxicity increase along their respective axes.
Each endpoint supports two or three categories; `response_level` and `dlt_level`
specify the first category counted as response or DLT. Categories at or above
the threshold are pooled for admissibility only. The utility calculation keeps
all joint cells. In the binary example, the matrix therefore lists no-response
first and no-DLT first, unlike the paper's flattened outcome order.

The prior is required, either one common matrix or a matrix per dose. The paper
specifies positive cell parameters with total prior mass one, without identifying
the individual parameters. The uniform example above is an explicit choice.
Other positive prior masses are supported. Utilities are required on the 0–100
scale. A three-category example, with efficacy rows PD, SD, PR/CR and toxicity
columns minor, moderate, severe, is `[[30,15,0],[50,30,0],[100,45,15]]`.

For posterior cell shapes `a = prior + counts`, total `A`, and utility `u`,
the mean is `sum(a*u)/A`. The variance is
`sum((a/A)*(u-mean)**2)/(A+1)`, in squared utility units. Toxicity and efficacy
tails use the exact Beta marginals of this joint Dirichlet distribution.
A dose is admissible when both tails are **at most** their cutoffs. These are
posterior means and probabilities, without Monte Carlo error.

`uboin_allocation` intersects admissibility with the caller's required `eligible`
mask. `winner` assigns probability one to the maximum-mean dose, with the lowest
array index resolving exact ties; `proportional` normalizes mean utilities;
`equal` assigns uniform probabilities. No eligible dose returns all zeros.
Proportional allocation with zero utility at every eligible dose raises an error.
Tie and zero-utility policies are explicit Python choices, not verified native
defaults. Posterior and allocation arrays are read-only.

## Two-stage trial conduct

`UBOINDesign` combines the posterior and allocation rules with stage-I BOIN
exploration, the stage transition, stage-II exploration, stopping and final OBD
selection. Supply complete joint counts from both stages at every decision.
Dose indices are one-based.

```python
from mdanderson_stats import UBOINDesign

trial = UBOINDesign(
    prior=[[.25, .25], [.25, .25]], utilities=[[30, 0], [100, 50]],
    candidate_scope="tried", s1=12, s2=24, max_patients=54,
)
decision = trial.decision(
    [[[0, 0], [12, 0]], [[0, 0], [0, 0]]], current_dose=1, stage=1
)
assert decision.stage == 2
assert decision.next_dose == 2  # Explore above the highest tried safe dose.
```

Pass the returned `stage` and `eliminated` mask into the next call. They preserve
stage and safety history; cumulative counts alone do not reconstruct past
eliminations. With empty data the controller assigns the starting dose. It
returns allocation probabilities for every continuing decision; `next_dose` is
available when the allocation is deterministic. For randomized assignment, draw
one dose for the next cohort using those probabilities.

Stage I uses BOIN boundaries at `toxicity_limit - delta`, with default `delta=.05`.
Its separate overdose test is `Pr(toxicity > toxicity_limit) > .95` under a
Beta(1,1) prior, after at least three patients at a dose. Crossing that threshold
eliminates that dose and all higher doses, persistently. Elimination of the lowest
dose stops the trial. Neither the stage-II prior nor its safety cutoff changes
this stage-I rule. `run_in_3plus3=True` enables the official optional 3+3 rules at
three or six patients, restricted to a BOIN target of .25.

After any dose reaches `s1`, the controller switches to stage II. Before utility
allocation, it explores one level above the highest tried dose when that dose's
empirical DLT rate is at most the escalation boundary and the next level has not
been eliminated. This B1 step precedes the B2 admissibility calculation in the
paper. Otherwise, it allocates by the configured `method`. Stage-II toxicity and
futility exclusion use the joint posterior's per-dose admissibility; there is no
new automatic suffix elimination in stage II.

`candidate_scope` is required because the sources do not explicitly settle
whether B2 and final selection include untried doses. `"tried"` restricts both to
doses with observations; `"all"` permits all configured doses, including their
prior-only summaries. Both choices respect prior eliminations and posterior
admissibility. The all-dose option can assign an untried dose through B2 even
when the B1 escalation condition fails. These are explicit alternatives, not a
claim about the native default.

At `max_patients` or when any dose reaches `s2`, allocation stops and
`selected_dose` reports the admissible maximum-mean utility dose, or `None`.
`select_obd(counts, eliminated=...)` also performs that selection on demand;
it does not reconstruct stage-I safety history. Final selection uses the maximum
mean regardless of the interim allocation method. Exact ties favor the lowest
dose index. Stage-I movement is clamped at the lowest/highest level; a blocked
escalation stays at the current dose, and an eliminated current dose moves to an
available lower dose. These edge policies are documented Python choices.

The controller accepts at most 100 doses and 1,000 total observations. Accelerated
titration, delayed-efficacy/immune-response imputation and native reports
remain unimplemented. The application's categorical controls also carry
under-development labels; native categorical output parity is not claimed.

The independent base-R script `tools/reference_uboin.R` checks binary, 3×3 and
prior-only 2×3 cases, including utility variance from Dirichlet cross moments.
Its fixtures validate the declared mathematical model, not hidden native output.
The paper supplement was inaccessible behind a challenge page during the audit.

`tools/reference_uboin_conduct.R` independently computes 28 stage-I boundary
rows across four targets; decisions on both sides of these cutoffs check
escalation, de-escalation and toxicity elimination against base R.

## Complete-outcome operating characteristics

```python
from mdanderson_stats import simulate_uboin, uboin_gumbel_probabilities

joint = uboin_gumbel_probabilities([.15, .35], [.35, .65], association=.2)
operating = simulate_uboin(trial, joint, trials=100, cohort_size=3, seed=142)
print(operating.selection_probability)  # no OBD, dose 1, dose 2
print(operating.selection_mcse)
```

The scenario may instead be any joint `(dose, efficacy, toxicity)` probability
table matching the design's categories. Each dose's cells must sum to one within
an absolute tolerance of `1e-12`; accepted roundoff is normalized. For binary
outcomes, the Gumbel helper implements the official association model. It uses
`tanh(association/2)` and factored nonnegative cells, avoiding exponential overflow
even for association values of ±1,000. Association zero gives independence.

Simulation calls the controller after each complete multinomial cohort and
carries its stage and elimination mask forward. The last cohort is shortened to
respect `max_patients`. The `s1` and `s2` thresholds are evaluated after a cohort,
so counts may exceed either threshold by up to `cohort_size - 1`. The result
contains final joint counts and selections for each trial, stop reasons, mean
patient/DLT/response counts per dose, selection probabilities and their binomial
Monte Carlo standard errors. Selection zero means no OBD. Early stopping means
fewer than `max_patients` accrued, which is distinct from failing to select an OBD.

Runs use one local NumPy random generator; specify a seed for reproducibility.
They retain no patient histories or potential outcomes. Limits are 100,000 for
`trials * max_patients`, 100,000 retained joint cells, and 1,000,000 for
`trials * max_patients * number_of_doses`. Cohort size is between 1 and 100.
These bounds keep memory and computation predictable; the simulator has no
parallel workers. Native random-number-stream parity is not claimed.

Independent references comprise eight Gumbel cases and an exact enumeration of
a two-dose, two-cohort trial (`N=6, s1=3, s2=9`, uniform cell prior, tried-only,
winner allocation). At toxicity rates `[.15,.35]`, efficacy rates `[.35,.65]` and
association `.2`, base R gives selection probabilities
`[.035787468407932, .627499667282430, .336712864309637]` for no OBD, dose 1 and dose 2.
Python controller enumeration checks every path against these aggregate results;
no large random simulation is needed for that comparison. These references
validate the declared conduct choices, not the application's undisclosed settings.
