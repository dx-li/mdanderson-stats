# Phase2Delay: delayed binary outcomes

This port implements the interim multiple-imputation method linked by the
[MD Anderson Phase2Delay application](https://biostatistics.mdanderson.org/shinyapps/Phase2Delay/).
The method is described by Cai, Liu and Yuan (2014),
[Statistics in Medicine 33, 4017–4028](https://pmc.ncbi.nlm.nih.gov/articles/PMC4435968/).
The app supports response, toxicity and progression. Source snapshots and hashes
are recorded in [phase2delay-sources.json](phase2delay-sources.json).

```python
from mdanderson_stats import phase2_delay_monitor

result = phase2_delay_monitor(
    event=[1, 0, 0],
    event_time=[0.25, 0, 0],
    followup=[0.25, 1, 0.5],
    window=1,
    endpoint="response",
    threshold=0.4,
    cutoff=0.95,
    prior_alpha=0.3,
    prior_beta=0.7,
    intervals=2,
    hazard_c=[2, 3],
    lambda0=0.4,
    burn_in=500,
    hazard_draws=4000,
    seed=141,
)
print(result.posterior_probability, result.posterior_probability_mc_se)
print(result.decision)
```

These explicit priors illustrate the independent two-interval reference below;
they are not recovered app defaults. `lambda0` has inverse-time units. Multiply
times and the window by the same conversion factor and divide `lambda0` by
that factor when changing units.

## Statistical contract

Each record contains an event indicator, event time, and current follow-up, all
in the same units as the evaluation window. An observed event determines the
binary outcome immediately. A non-event is complete only at the end of the
window; earlier non-events remain pending. Exposure ends at an observed event
or current censoring time. Events exactly on an internal partition boundary
belong to the interval beginning there; the final interval includes the window
endpoint.

The imputation model has constant hazards within equally spaced intervals. The
hazard prior is a Gamma martingale: conditional on the preceding hazard,
`lambda[j] ~ Gamma(shape=c[j], rate=c[j]/lambda[j-1])`. The first preceding
hazard is the explicitly supplied `lambda0`. The paper calls the second
parameter a *scale*, which contradicts its stated conditional mean. This port
uses a **rate** so that the conditional mean is the preceding hazard. This is
a documented mathematical interpretation, not verified parity with hidden
Shiny code. Concentrations must be positive; zero is not a proper Gamma prior.

For review of the sampler, let `D[j]` and `E[j]` be the event count and summed
exposure in interval `j`. The interior full conditional is proportional to
`lambda[j]**(D[j]+c[j]-c[j+1]-1)` times
`exp(-(E[j]+c[j]/lambda[j-1])*lambda[j]-c[j+1]*lambda[j+1]/lambda[j])`.
The final interval drops the next-interval terms and has a Gamma conditional.
Changing to log hazards adds the Jacobian, removing the exponent's `-1`.

For a pending patient, conditional event probability is
`-expm1(-sum(hazard * remaining_interval_exposure))`. Each posterior hazard
draw supplies an imputed binary outcome for each pending patient. Completed
datasets update the supplied Beta prior, and averaging their Beta tail
probabilities gives the multiple-imputation estimate. Response uses the lower
tail below the acceptable rate; toxicity and progression use the upper tail
above the unacceptable rate. Stopping requires the estimate to be **strictly
greater** than the probability cutoff. With no pending outcomes, the result
reduces to the exact complete-data Beta tail.

## Numerical interpretation

Sampling uses log hazards and normalized times, preserving very small hazard
states without clipping the posterior distribution. The finite sampler uses
slice updates within Gibbs sweeps; bracketing or shrinkage failure raises an
error. Retained traces permit inspection and independent runs with different
seeds. Burn-in and retained draw counts alone do not establish convergence.
`hazard_trace` and `log_hazard_trace` use the original inverse-time units;
reporting a tiny hazard as zero does not truncate the log-hazard sampler state.
`probability_trace` contains one Beta tail per imputed dataset. Complete-data
results return empty traces.

The implementation allows at most 1,000 subjects and 100 intervals. Before
sampling it limits `(burn_in + hazard_draws) * intervals` to 200,000,
`hazard_draws * imputations_per_draw * subjects` to 2,000,000, and
`hazard_draws * subjects * intervals` to 5,000,000. These are Python resource
limits. Traces retain hazard states and tail probabilities, not a matrix of
every imputed patient outcome.

`posterior_probability_sd` describes variation between imputed-data tail
probabilities, not uncertainty in the response rate. The Monte Carlo standard
error uses batch means of the per-hazard averages, accounting for the shared
hazard draw when multiple imputations are requested. It is an estimated
sampling error, not a convergence guarantee. A stopping estimate near the
cutoff needs assessment of that error and chain behavior. With fewer than two
retained hazard draws the error is unestimable (`NaN`); complete-data results
are exact and have zero Monte Carlo error.

## Independent validation

[reference_phase2delay.R](../tools/reference_phase2delay.R) uses base R to
integrate out the hazards without a Markov chain or random imputations. Its
one-interval case uses the Gamma Laplace transform and enumerates both pending
outcomes. Its two-interval case analytically integrates the second hazard,
then uses one-dimensional quadrature for the first hazard. The latter checks
the prior dependence between neighboring hazards. The saved fixture includes
all three endpoint directions and posterior hazard means.

The official 20-patient toxicity data were also exercised with window 6,
threshold 0.4, Beta(0.1, 0.2), `c=0.01`, `lambda0=0.1`, six intervals, 200
burn-in sweeps and 500 draws (seed 141). This gives 15 observed and five pending
outcomes, with estimated upper-tail probability 0.811745 and Monte Carlo
standard error 0.005349. This Python smoke check took 0.22 seconds and about
109 MiB peak process RSS on the development machine, with zero process swaps.
It is not an independently recovered native app output or a convergence claim.

## Remaining coverage

Native server priors, burn-in, draw counts and RNG behavior are unpublished.
Python numerical settings must not be interpreted as recovered app defaults.
The app's upload schema is only publicly illustrated for toxicity; the Python
API accepts numerical arrays without claiming native file-format parity.
Calendar trial simulation, operating-characteristic calibration and the
native report workflow remain unimplemented. Catalog entry 141 remains partial.
