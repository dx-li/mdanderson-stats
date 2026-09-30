# Six-dose posterior integration by importance sampling

The six-dose model also has an opt-in bounded importance sampler. It uses the
same response regression, independent normal coefficient prior, binomial
likelihood and analytic beta toxicity probabilities as
`fit_phase12_model`. It does not change that function's elliptical-slice
default.

```python
import numpy as np

from mdanderson_stats import fit_phase12_importance

tally = np.zeros((6, 4))
tally[:, 0] = [8, 7, 6, 5, 4, 3]  # no response, response, no DLT, DLT
tally[:, 1] = [2, 3, 4, 5, 6, 7]
fit = fit_phase12_importance(
    tally,
    max_integrations=2_000,
    rng=np.random.default_rng(20260929),
)
print(fit.efficacy_probability)
print(fit.efficacy_probability_mc_se)
print(fit.integrations, fit.converged)
```

The proposal is a mixture of a four-dimensional normal centered at the fitted
posterior mode with covariance equal to the inverse posterior Hessian, and the
product-normal prior. The multivariate component defaults to `.99`, leaving a
`.01` prior tail component. Each draw is weighted by
`likelihood * prior_density / full_mixture_density`; the density of only the
selected mixture component is never used. The mode uses bounded BFGS with an
analytic gradient, and the proposal covariance uses the exact local logistic
Hessian. Those Python choices replace the archived Nelder–Mead and numerical
curvature routines, so exact native proposal draws or bitwise parity are not
claimed.

The first 60 posterior summaries follow the C++ indicator order: six strict
comparisons with dose zero (with the source's dose-zero value fixed at `.5`),
six `response >= efficacy_target` probabilities, six `response >
future_target` probabilities, 36 strict pairwise comparisons in row-major
order, and six posterior second moments of response probability. The result
also includes six posterior response means as a Python extension. Toxicity
exceedance probabilities are evaluated analytically from the beta priors and
observed toxicity counts. Importance fits can be passed to the existing source
decision and final-selection functions.

The default integration target is relative error `.001` with at most 10,000
draws; the maximum may be raised explicitly to 1,000,000 and must be a multiple
of 100. The integrator checks every 100 draws. It applies the native criterion
to the evidence integral and the 60 source summary integrals:

```text
SE(I_k) <= target * max(I_k, 0.01 * I_evidence)
```

Here `SE` is the ordinary sample standard error of the importance integrand.
This is a raw-integral stopping rule, not a direct bound on a normalized
posterior ratio's error. `log_integral_mc_se` reports log-scale raw-integral
standard errors; `ratio_mc_se` and the named `*_mc_se` fields report
paired-draw delta-method standard errors for normalized summaries.
`converged=False` means the cap was reached without meeting the stopping
criterion; the last 100-draw checkpoint can pass. The reported `log_evidence`
omits binomial combinatorial constants, matching the archived likelihood
kernel. As in the source stopping rule, an
indicator never observed in the proposal sample has estimated zero standard
error; rare zero-hit probabilities therefore need care even when `converged`
is true.

`converged=True` means the raw-integral rule passed; it does not certify that
every normalized ratio's MCSE is below the requested relative error. In
particular, summaries below one percent of evidence use the source's
one-percent floor. Inspect named `*_mc_se` fields when a decision is close to
its cutoff.

The implementation stores no draws and caps the request at 67 million
summary-component evaluations. Given a fixed seed, callers can replay a run
with a newly created NumPy `Generator` using the same bit generator and seed.
