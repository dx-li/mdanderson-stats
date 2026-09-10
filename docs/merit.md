# MERIT dose optimization

The Python implementation covers fixed-size MERIT dose selection, simulation,
sample-size/boundary search, and a standalone Bayesian interim decision.
The [MD Anderson application](https://biostatistics.mdanderson.org/shinyapps/MERIT/)
is catalog entry 160, version 1.1.3.0 (displayed update 01/06/2026).
The algorithm source is Yang et al.'s [author manuscript, version 2](https://arxiv.org/abs/2302.09612v2),
sections 2.2–2.5 and Table 2. The published article is *Statistics in Medicine*
43:2972–2986 (2024), [doi:10.1002/sim.10093](https://doi.org/10.1002/sim.10093).
[Source provenance](merit-sources.json) records the retrieved files; those files
are not redistributed with the package.

## Decisions and monitoring

`MERITDesign(patients_per_arm, toxicity_max, efficacy_min, doses=2)` accepts
2–4 ordered doses. `select(toxicity, efficacy)` selects every dose with toxicity
count at most `toxicity_max` and efficacy count at least `efficacy_min`.
Inputs can be individual dose vectors or batches with doses on the last axis.
Patients per arm must be equal, and both endpoints must be completely assessed.

By default, both sets of counts undergo equal-weight increasing isotonic pooling.
The fitted counts can be fractional and are compared directly to the boundaries.
Either endpoint's pooling can be disabled explicitly. The output is an admissible
dose set; final optimal biological dose selection requires the additional
benefit-risk information described in the paper.

`merit_monitor(n, toxicity, efficacy, toxicity_target=..., efficacy_target=...)`
uses the acceptable **alternative** rates as its targets. It returns upper-tail
toxicity and lower-tail efficacy posterior probabilities and separate strict
stopping flags. Defaults are Beta(.1,.1) priors and .95 probability cutoffs.
Pooling is on counts before computing these probabilities; this convention is
explicit and does not establish parity with the current app's interim internals.
The function operates on one equal-size interim and does not infer previous stops.

## Simulation and optimization

`simulate_merit` generates correlated binary endpoints through the paper's latent
Gaussian model. `correlation` is the latent normal correlation, not the binary
Pearson correlation. Arms and patients are independent before optional isotonic
pooling. Rates zero/one and latent correlation -1/1 are supported.
An optional boolean `truly_admissible` dose vector enables estimation of:

- Power I: the selected set is nonempty and contains no unacceptable dose.
- Power II: at least one acceptable dose is selected.

`merit_sample_size` searches sample sizes starting at one and all integer boundary
pairs. It evaluates all `(J+1)(J+2)/2` null corner configurations and all
`J(J+1)/2` alternative configurations, retaining per-configuration results.
It uses the event definitions of both powers directly, including the dependence
introduced by pooling, rather than multiplying marginal rejection probabilities.

The search stops at the first sample size meeting the **estimated** constraints.
Among feasible boundary pairs it prefers greater chosen global power, then smaller
global type I error, smaller toxicity maximum, and greater efficacy minimum.
These tie conventions and common random numbers are Python implementation choices;
exact native RNG/search parity is not claimed.

Simulation error and selection over candidate designs affect the result. Validate
selected designs with fresh simulations and sufficient precision for the intended
use. The maximum error is over the paper's enumerated corners; it is not a proof
of uniform control across the entire continuous composite null. Returned scenario
rates have axes `(scenario, dose, endpoint)`, with toxicity before efficacy.

The search supports up to 200 patients per arm and 100,000 replicates per scenario.
It stores running counts, not every patient's outcomes. Inclusion-exclusion and
integer cumulative histograms evaluate all boundary pairs without allocating a
trial-by-boundary tensor. For 2–4 doses, a batched min-max isotonic calculation
avoids invoking a separate solver for each simulated trial.

```python
from mdanderson_stats import merit_sample_size, simulate_merit

result = merit_sample_size(
    toxicity_null=0.4,
    toxicity_alternative=0.2,
    efficacy_null=0.2,
    efficacy_alternative=0.4,
    doses=2,
    alpha=0.1,
    power=0.6,
    power_definition=2,
    trials=5000,
    rng=160,
)
# This seeded run reproduces Table 2's (n, mT, mE) = (26, 7, 9).
assert result.design.patients_per_arm == 26
assert (result.design.toxicity_max, result.design.efficacy_min) == (7, 9)
validation = simulate_merit(
    result.design,
    [0.2, 0.4],
    [0.4, 0.4],
    truly_admissible=[True, False],
    trials=50000,
    rng=2026,
)
```

That search took approximately 0.11 seconds in the development environment,
with estimated maximum corner error .0988 and minimum power II .6156.
Timing is illustrative, not a general performance guarantee. Changing the seed
or simulation size can change the selected boundary or sample size.

Four focused tests compare isotonic pooling with SciPy's solver, every boundary
against exhaustive three-dose count enumeration, monitoring probabilities with
Beta distribution calculations, independent endpoint simulation with exact
binomial probabilities, correlated endpoints with an analytic Gaussian orthant
probability, and searched designs with independent scenario simulations.

## Remaining coverage

**Catalog status is partial.** A complete trial runner with persistent interim
stopping, unequal arm sizes after stopping, separate endpoint assessment schedules,
interim boundary tables, app scenario files and reports remain pending. The
simulation and optimization currently use fixed-size trials without interim
stopping. Current app source/default and published-version parity also remain
subject to audit; the numerical implementation is based on the public v2 manuscript.
