# EffTox prior-calibration source contract

The probability-moment and calibration APIs are implemented; native Windows
optimizer and integration parity remain unverified.
The primary method is [Thall et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4229398/),
[doi:10.1177/1740774514547397](https://doi.org/10.1177/1740774514547397).
On 2026-09-28 the direct PMC page returned a CAPTCHA, while indexed primary
full text exposed the computational-algorithm section.

For each outcome, inputs are doses, elicited marginal probability means at
each dose, and a target effective sample size. For model-implied probability
mean `m_k` and variance `v_k`, the beta-moment approximation is
`ESS_k = m_k*(1-m_k)/v_k - 1`; average it over doses. This is distinct from
the curvature-trace ESS in `regression_ess.py` and from MCMC effective draws.

Optimize four Gaussian hyperparameters (intercept/slope means and SDs) per
outcome, using the published objective

```text
sum((elicited_mean_k - m_k)**2)
  + 0.1*(mean(ESS_k) - target_ESS)**2
  + 0.02*(sd_intercept - sd_slope)**2.
```

The paper uses Nelder–Mead. Endpoint elicited logits initialize the two means.
With `v_k = p_k*(1-p_k)/(target_ESS+1)`, equal starting SDs use
`s**2 = sqrt(v_1*v_K) / mean((p_k*(1-p_k))**2*(1+x_k**2))`, where the mean
uses only the two endpoints. The published example has doses
`[1, 2, 4, 6.6, 10]`, efficacy means `[.2, .4, .6, .8, .9]`, toxicity means
`[.02, .04, .06, .08, .1]`, and target ESS `.9` for both outcomes.

## Model and native-comparison limits

The implementation must distinguish fixed **hyperparameters** for a random
coefficient from fixing that coefficient itself. Full-prior efficacy moments
must include uncertainty in the supplied quadratic coefficient. A source
review initially inferred that this uncertainty was omitted from calibration;
that inference was not established and must not become an implicit default.

The accessible [trialr source](https://rdrr.io/cran/trialr/src/R/get_efftox_priors.R)
explicitly samples the quadratic coefficient with SD `.2`. Its
[prior-container example](https://rdrr.io/cran/trialr/src/R/efftox_priors.R)
also labels the published hyperparameters as SDs. This is corroborating
implementation evidence, not the original Windows calibration kernel.
It differs from the intended paper/model contract: its toxicity slope is
untruncated, its SD-difference penalty is `.2` instead of the paper's `.02`,
and it optimizes both outcomes together using Monte Carlo moments. Do not
copy those choices as native EffTox facts.

Python should expose its monotone-toxicity truncation, curvature prior,
integration rule, optimizer status and achieved moments/ESS explicitly.
The Windows kernel's exact integration settings, parameter bounds and
stopping tolerances remain unverified. Rounded published hyperparameters
are contextual reference values, not an exact numerical acceptance oracle.

## Independent probability-moment references

`tools/reference_efftox_calibration.R` uses base-R adaptive normal integration
to generate `tests/fixtures/efftox-prior-moments.csv`. Its 34 outcome/dose
rows cover the published four-decimal hyperparameters with and without
toxicity-slope truncation, a negative untruncated slope mean conditioned
positive, and a nearly fixed prior. Efficacy integrates the complete
Gaussian linear predictor, including independent curvature uncertainty.
Toxicity uses nested integration when the slope is truncated. Variance is
integrated after centering, avoiding subtraction of nearly equal moments.

At the published hyperparameters, average efficacy ESS is 0.90784698. Average
toxicity ESS is 0.90535706 for the untruncated normal and 1.29129905 when
conditioned positive. The former agrees closely with the published target
0.9. This is numerical evidence for an untruncated calibration convention;
it does not establish the Windows posterior or calibration implementation.
The primary paper's retrieved prior-parameterization passage states normal
priors without specifying truncation there. The earlier source review's
unqualified truncation claim was therefore too strong. Python must retain
the choice explicitly and evaluate the actual supplied prior consistently.

The reference uses relative tolerance 1e-11 and absolute tolerance 1e-30;
the latter matters for variances near 1e-12. At zero log dose, the nearly
fixed symmetric efficacy variance agrees with its small-SD expansion within
3.1e-12 relative. Generation completed in 0.64 seconds with 96.1 MiB peak
child resident memory and zero swaps. These are independent mathematical
references, not executed native EffTox calibration outputs.

## Python integration validation

The public API matches all 34 R rows: maximum absolute mean error
`5.00e-16`, variance error `2.50e-16`, and relative ESS error `1.57e-12`.
Four focused tests, targeted lint/format and type checks passed. Checks include
a fixed intercept with random positive slope and truly fixed probabilities.

The public five-dose calibration example with untruncated toxicity and target
ESS 0.9 converged in 364 efficacy and 399 toxicity evaluations. Achieved mean
ESS values were `0.90946338` and `0.90538019`, with objectives `0.00580248`
and `0.00291032`. Both objective values were recomputed independently from
the returned moments and SDs. Neither optimum hit the reported bounds.
Adaptive integration's largest reported absolute error estimate was
`1.87e-12`. This estimate concerns probability integrals, not ESS error or
the discrepancy from elicited targets.

The guide example took 16.97 seconds; the combined reference/example process
peaked at 115.3 MiB RSS with zero swaps. It uses sequential deterministic
integration with bounded optimizer evaluations. No large simulation or
whole-repository test run was required for this integration checkpoint.

Wheel and source-distribution builds from committed revision `b2092b7`
passed. All 486 Python modules matched the committed source byte for byte in
both archives, including calibration, trinary fitting, simulation and legacy
contours. The wheel includes third-party notices; neither archive included
raw native research files or compiled local binaries. This validates package
contents, not all methods in the catalog. GitHub publication remains pending.
