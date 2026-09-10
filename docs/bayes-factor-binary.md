# Bayes Factor Binary

Independent Python implementation of MD Anderson's
[Bayes Factor Binary](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/94),
based on the [version 1.0 guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BayesFactorBinary/UsersGuide_BayesFactorBinary.pdf)
(November 26, 2012). The guide cites Johnson and Cook, *Clinical Trials* 2009,
6:217–226, and Johnson and Rossell, *JRSS B* 2010, 72:143–170. Original binaries
are not redistributed; [source provenance](bayes-factor-binary-source.json) records
the reference used.

## Model and stopping rules

The null fixes the response probability at `null_rate`. The alternative is a
**distribution**, not a point hypothesis: a one-sided iMOM prior with `k=1`,
`nu=2`, restricted and normalized on `(null_rate, 1)`. Its scale is
`tau = 1.5*(alternative_mode-null_rate)**2`. Writing `d=theta-null_rate`,
its density on that interval is
`2*tau*exp(tau/(1-null_rate)**2) * d**(-3) * exp(-tau/d**2)`.
The normalization is essential after truncating the prior. The guide's example
has null 0.2, mode 0.3 and tau 0.015; the sentence calling 0.2 the mode is a typo.

Equal prior model odds imply posterior alternative probability
`expit(log_bayes_factor)`. Inferiority requires probability strictly below its
cutoff; superiority requires probability strictly above its cutoff. Boundary
comparisons use log odds, preserving decisions when displayed probabilities
round to zero or one. At maximum enrollment, a trial between the boundaries is
**inconclusive**. Cutoffs zero and one disable their respective stopping rules.

```python
import numpy as np
from mdanderson_stats import bayes_factor_binary_design

trial = bayes_factor_binary_design(50)
np.testing.assert_array_equal(trial.inferiority_max, [-1, 0, 1, 3, 4, 5, 6, 8, 9])
np.testing.assert_array_equal(trial.superiority_min, [6, 8, 9, 11, 12, 13, 15, 16, 17])
np.testing.assert_array_equal(
    trial.monitor([9, 10, 16, 17], 50).decision,
    ["inferiority", "inconclusive", "inconclusive", "superiority"],
)
oc = trial.operating_characteristics([0.2, 0.25, 0.3, 0.35, 0.4])
np.testing.assert_allclose(oc.superiority + oc.inferiority + oc.inconclusive, 1)
median_enrollment = oc.sample_size_quantile(0.5)
simulation = trial.simulate([0.2, 0.3, 0.4], n_trials=10000, rng=94)
```

`monitor` broadcasts response counts and sample sizes. `monitor_outcomes` accepts
one or more patient paths and preserves the first stopping decision; probability
fields still describe cumulative observations supplied by the caller. Exact
operating characteristics propagate Bernoulli path mass through each analysis,
including final superiority, inferiority and inconclusive outcomes. They expose
stopping probabilities by look, enrollment PMF, mean, SD and discrete quantiles.
Simulation returns patient counts, response counts and decisions for each path,
plus outcome probabilities and plug-in Monte Carlo standard errors. The last
axis of `decision_probability` and `monte_carlo_se` is ordered **inferiority,
superiority, inconclusive**. A zero estimated MCSE is not proof of zero event risk.

Maximum enrollment is 1–400. Cohort size must divide it; minimum enrollment above
the cohort size must also be divisible by cohort size. A smaller minimum starts
monitoring after the first complete cohort. The null and alternative mode must
satisfy `0 < null < mode < 1`. Simulation supports 1–100,000 trials, with at most
10 million scenario-by-trial paths. NumPy seeds are reproducible within this
implementation; they do not reproduce the native program's random stream.

## Input and HTML report

`parse_bayes_factor_binary_input` accepts the guide's format: ten single-value
settings (seed, N, null, mode, inferiority cutoff, superiority cutoff, simulation
count, Yes/No boundary-table switch, minimum enrollment, cohort size), followed
by one or more scenario response rates. Blank lines and `#` comments are ignored.
It validates settings and returns a job whose `run()` returns both exact and
simulated results. `to_html()` creates a standalone report with the input settings,
scenario results, enrollment quantiles and optional inclusive stopping boundaries.
The report is an independent Python layout.

```python
from pathlib import Path
from mdanderson_stats import parse_bayes_factor_binary_input

text = """94 # seed
50
0.2
0.3
0.1
0.9
10000
Yes
10
5
0.2
0.3
0.4
"""
job = parse_bayes_factor_binary_input(text)
report = job.run()
Path("bayes-factor-binary.html").write_text(report.to_html(), encoding="utf-8")
```

## Numerical method and checks

The integration coordinate is `t=log((theta-null)/sqrt(tau))`. Composite
Gauss–Legendre quadrature integrates the normalized prior times the likelihood
in log space. Each panel spans at most one t unit. Orders double from 8 to at
most 256 until all maximum-sample-size log marginal likelihoods change by less
than `2e-10`. An exponentially small prior tail is truncated using a
sample-size/null-dependent cutoff. Earlier likelihoods follow by stable
`logaddexp` recursion, avoiding repeated integration for each sample size.
Prior normalization and Bayes-factor monotonicity are checked. Failure to
converge raises `ArithmeticError`; the convergence criterion is a numerical
estimate, not a rigorous error certificate for every finite input.

Five focused tests reproduce every published boundary; compare likelihoods with
independent normalized-prior integration; enumerate every path of a small trial;
check simulation against exact probabilities; check disabled stopping, persistent
path decisions and final inconclusiveness; and validate a 400-patient rare-null
case with log Bayes factor above 5,000 against direct integration. Reported native
Monte Carlo results agree within their simulation uncertainty. No new CI
workflow is introduced.
