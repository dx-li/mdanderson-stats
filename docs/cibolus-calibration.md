# CiBolus prior calibration

CiBolus prior calibration generates balanced pseudo data, fits each pseudo
posterior and averages its log-parameter means. A separate prior-predictive
function helps assess caller-selected prior standard deviations. It builds on
the [CiBolus model and observation conventions](cibolus.md).

The [paper's Section 4.2](https://odin.mdacc.tmc.edu/~pfthall/main/Biometrics_IAtPA_2011.pdf)
uses repeated balanced pseudo samples to choose log-parameter means, then
assesses prior information and simulated design performance to choose variances.
Its example uses 400 observations per pseudo sample: 50 at each of eight
regimens. The Python fitter supports that sample size. The paper's diffuse
pseudo prior and final variance are study-specific choices, not universal
defaults.

## Inputs and results

Supply the full joint response/toxicity table for every concentration/bolus
pair and the observation endpoints. Response categories are immediate bolus
response, successive detection intervals and failure to respond by time one;
toxicity columns are absent and present. Each regimen's table sums to one.
The routine does not guess an interpolation from a few elicited probabilities.

Each pseudo sample has the same configured number of patients per regimen.
Interval observations preserve the upper-endpoint treatment and toxicity
convention. The explicit pseudo prior uses eleven independent normal
log-parameter coordinates. Zero standard deviation fixes a coordinate.

The result retains replicate counts, seeds, fitted means and diagnostics,
along with the average log-parameter mean and its between-replicate uncertainty.
Use these means with explicitly chosen prior standard deviations for a
subsequent analysis. Calibration does not automatically choose those deviations
or certify a design's operating characteristics.

```python
import numpy as np
from mdanderson_stats import (
    CiBolusPrior,
    calibrate_cibolus_prior,
    cibolus_predict,
    cibolus_prior_predictive_moments,
)

mean = np.log([.5, .7, .8, .08, 1.4, 1.6, .03, .9, .12, .25, .2])
mean[6] = -2.0
sd = np.zeros(11)
sd[6] = 0.6  # Only log(beta0) varies in this small demonstration.
concentrations, boluses, endpoints = [.2, .4], [.1, .2], [.5, 1.0]
joint = cibolus_predict(
    mean, concentrations, boluses, endpoints, utility=np.zeros((4, 2))
).joint
calibration = calibrate_cibolus_prior(
    concentrations, boluses, endpoints, joint,
    repetitions=2, patients_per_regimen=2,
    pseudo_prior=CiBolusPrior(mean, sd), resulting_sd=np.full(11, .25),
    pseudo_draws=32, pseudo_warmup=16, pseudo_chains=2,
    rng=np.random.default_rng(8611),
)
assert calibration.joint_counts.sum() == 16
print(calibration.prior.mean, calibration.replicate_mean_mcse)
```

This example checks the interface using a reduced model and short chains. It
does not reproduce the paper's diffuse eleven-coordinate calibration or
establish precision. The returned standard deviations are exactly the supplied
`resulting_sd` values.

```python
moments = cibolus_prior_predictive_moments(
    calibration.prior, concentrations, boluses, endpoints,
    draws=32, chains=2, rng=np.random.default_rng(8612),
)
assert moments.source_probability_ess.shape == (2, 2, 4)
print(moments.source_probability_names, moments.source_probability_ess)
```

## Prior information and numerical limits

For probability mean `m` and variance `v`, beta moment matching gives
`ESS = m * (1 - m) / v - 1`. Variance uses the empirical population moment
(divisor equal to the number of draws). Constant interior probabilities have
infinite limiting ESS; constant probabilities zero or one have undefined ESS.
The source's information check concerns response
and conditional toxicity probabilities at times zero and one. Additional
endpoint and failure summaries help inspect the model but are distinct from
that source subset. Conditional toxicity after failure is also distinct from
conditional toxicity for a response at time one.

Pseudo fits run serially under cumulative likelihood and work budgets, with a
combined retained-storage limit. Numerical failures propagate; extreme prior
draws are not silently discarded. Inspect sampling errors and chain diagnostics
before interpreting the fitted prior means. Native elicitation-file handling,
automatic variance selection and reproduction of the published calibration
remain unverified.

Each call supports up to 1,000 pseudo replicates, 50 patients per regimen and
400 patients per pseudo sample, subject to the combined budgets. Adequate MCMC
settings can require splitting a large calibration into serial batches. Combine
the retained replicate means across those batches; do not give unequal-size
batches equal weight. This API does not launch parallel fits.

Independent R quadrature checks two reduced pseudo posteriors and prior
probability moments, while a separate bounded run exercises 400 pseudo patients.
See the [numerical audit](../research/cibolus-calibration-audit.md) for the
settings, errors and limitations. Response moments use the response CDF directly
so varying toxicity parameters cannot create artificial response variance.
