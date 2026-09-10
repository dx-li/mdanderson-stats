# Response and Survival: adaptive randomization with an early response

`response_survival_posterior` and `simulate_response_survival` implement the two
procedures in MD Anderson catalog entry 82, **Response and Survival**, by Jing
Ning. The method is described by Huang, Ning, Li, Estey, Issa and Berry,
“Using short-term response information to facilitate adaptive randomization for
survival clinical trials,” Statistics in Medicine 28:1680–1689 (2009),
[DOI 10.1002/sim.3578](https://doi.org/10.1002/sim.3578).

## Model and posterior comparison

Response is an immediately observed categorical outcome. Within each arm and
response category, survival is exponential with an unknown mean. Independent
Dirichlet priors govern response probabilities and inverse-gamma(shape, scale)
priors govern category-specific mean survival. The inverse-gamma expectation
is scale/(shape-1) when shape exceeds one. It is not scale/shape.

The posterior Dirichlet concentration adds category counts. The inverse-gamma
shape adds observed event counts, while its scale adds total event/censored
exposure. Each posterior draw independently samples response probabilities and
category means in each arm, then compares the two probability-weighted mixture
means. The result estimates **P(mean survival B > mean survival A)**, matching
the archived `comp` function (the paper writes the complementary A probability).
A category with no patients retains its prior. A censored patient contributes
response information and observed exposure, but no event.

```python
from mdanderson_stats import response_survival_posterior

posterior = response_survival_posterior(
    category_counts=[[8, 12], [5, 15]],
    events=[[4, 3], [2, 2]],
    exposure=[[20, 90], [18, 140]],
    response_prior=[0.5, 0.5],
    survival_shape=[5, 5],
    survival_scale=[12, 36],
    draws=10000,
    seed=82,
)
print(posterior.probability_b_superior, posterior.monte_carlo_standard_error)
```

Sufficient statistics use shape (2,K), with A in row zero and B in row one.
Priors may have shape (K,) for the archive's shared-arm prior or (2,K) for
arm-specific priors. All concentrations, shapes and scales must be positive;
K may range from 1 to 100. Arrays returned in the result are immutable. The
reported Monte Carlo standard error is sqrt(p*(1-p)/draws), measuring simulation
error rather than model uncertainty. Values zero and one can occur with finite
sampling and do not establish posterior certainty.

Independent gamma draws generate inverse-gamma means. Mixture sums are compared
in log space, avoiding overflow from explicit reciprocal gamma values or very
large survival units. Sampling is batched to bound memory. Unrepresentable
gamma draws raise an error instead of silently assigning an infinite mean.
A local seeded generator avoids changing global NumPy random state. R and
NumPy random streams are different, so exact seeded trajectory parity is not
claimed.

## Trial simulation

`simulate_response_survival(response_probabilities, survival_means, ...)`
accepts two (2,K) matrices for the true response probabilities and conditional
survival means. Row probabilities sum to one. Every survival, exposure and
calendar quantity must use the same unit: the paper uses weeks, while source
prompts say months. There is no automatic unit conversion.

```python
from mdanderson_stats import simulate_response_survival

simulation = simulate_response_survival(
    [[0.6, 0.4], [0.3, 0.7]],
    [[3, 9], [3, 12]],
    response_prior=[0.5, 0.5],
    survival_shape=[5, 5],
    survival_scale=[12, 36],
    max_patients=16,
    initial_patients=4,
    accrual_per_period=2,
    additional_followup=8,
    cutoff=0.95,
    replicates=20,
    posterior_draws=10000,
    seed=82,
)
print(simulation.selection_probability, simulation.mean_enrollment)
```

Patients arrive at times 1,2,... in batches of `accrual_per_period`, a positive
integer as in the archive. Initial patients receive independent 50:50 assignments,
not forced balance. Before each later enrollment, all previous responses and
survival observed up to that arrival time update the posterior. A patient's
exposure is min(lifetime, current time minus arrival), with an event exactly
when lifetime is no greater than elapsed follow-up. Latent future failure times
never enter an interim posterior. Responses are assumed immediately available,
including earlier patients in the same arrival batch, as in the paper and code.
This implementation does not substitute a delayed-response design.

The next patient receives B with probability P(B superior), unless that
probability is at least cutoff or at most 1-cutoff. In those cases the trial
stops and selects B or A respectively. These inclusive comparisons follow the
R code; the paper uses strict inequalities. A trial that reaches maximum
accrual receives its final analysis after `additional_followup` beyond the last
scheduled arrival. If neither arm crosses a boundary it is inconclusive.

The result supplies all four archive outputs: selection probabilities for A/B
and mean enrollments in A/B. It also includes their Monte Carlo standard errors
and per-trial enrollment, selection (-1 inconclusive, 0 A, 1 B), early-stop flag,
analysis time and final comparison probability. Enrollment standard errors are
NaN for one replicate, where a between-trial variance is unavailable. Selection
standard errors reflect outer trial replication; they do not remove the inner
posterior probability estimation error. Increase posterior draws and calibrate
design cutoffs by simulation when evaluating operating characteristics.

Maximum limits are 10,000 patients, 10,000 replicates, 1 million posterior draws,
and 1 billion patient/draw/category work units. The simulator supplies finite
trial replication, not live patient enrollment or a clinical decision service.

## Source audit, repairs and validation

The download contains one R file and the accompanying article. The R file
exports `comp` and `AR_survival`; both are covered above. All interactive inputs
and four reported summaries have explicit Python equivalents. Original files
are not bundled; hashes appear in `response-survival-sources.json`.

The native simulator marks a stopping decision but then enrolls another patient
before exiting its loop. Python stops before that extra enrollment. The source
also builds `rep(1:n, each=rate)` and uses the last element for final follow-up,
even though only its first n entries are used for accrual. At rate>1 that waits
until n+followup, instead of ceil(n/rate)+followup. Python uses the latter.
No extra native-compatible mode preserves these bookkeeping errors.

For the reference run, the archived `comp` body was unchanged. The harness bypassed
R-package imports and supplied the two required distribution primitives:
Dirichlet draws from normalized independent gamma draws, and inverse-gamma draws
as reciprocals of gamma(shape, rate=scale). Scripted inputs replaced prompts.
For the trial comparison, precisely the two bookkeeping repairs above were
applied to the R body; extra output captured its existing per-trial quantities.
The fixture records these modifications, Monte Carlo settings and results.

Three posterior comparisons, including a symmetric null and one-category case,
agree with native Monte Carlo output within five combined standard errors.
A 200-trial, two-category design was compared against that repaired native
simulator, using independent random streams and sampling-error comparisons.
Python took approximately 12 seconds locally for 200 trials, maximum 16 patients
and 10,000 posterior draws per decision. These comparisons assess the same
statistical algorithm, not a reproduction of the paper's full 5,000-trial tables.

Two focused tests check exact conjugate updates, an analytic one-category
inverse-gamma probability, time-unit scaling by 1e200, immutable results, early
stopping without extra enrollment, and final follow-up under batched accrual.
No CI jobs or dependencies were added. This catalog entry is implemented for
its supplied two-arm simulator; it does not claim general delayed responses,
non-exponential within-category survival or more than two treatments.
