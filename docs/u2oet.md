# U2OET: two-agent ordinal efficacy/toxicity models

Entry 77 is **partial**. The Python implementation supplies PDS, CMI and hybrid
model probabilities, grouped likelihoods and conditional expected utilities.
Posterior sampling, prior calibration, dose allocation, trial simulation and
native input/report workflows remain pending. GAO with its Gaussian copula is
also pending; it is not replaced with the FGM model.

The sources are the [official U2OET 1.8 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/U2OET/U2OET_V1.8.zip),
its user guide and the [author-hosted paper](https://odin.mdacc.tmc.edu/~pfthall/main/JRCCS_2017_ph12_2agent_utility.pdf)
by Thall, Nguyen and Zinner, JRSS Series C 66 (2017), 201–224,
doi:10.1111/rssc.12162. [Source hashes](u2oet-sources.json) record the archive,
its files and the paper. The archive contains a Windows executable and examples,
but no primary C++ source. Its comparison R programs implement other designs.
No original executable, paper, examples or comparator code is redistributed.

## Python API

```python
import numpy as np
from mdanderson_stats import U2OETMarginal, u2oet_probabilities

efficacy = U2OETMarginal(
    intercepts=[-0.8, 0.1, -0.3],
    slopes=[[0.7, 1.2], [0.4, 0.8], [0.3, 1.1]],
    powers=[0.4, 2],
    link=0.7,
)
toxicity = U2OETMarginal(
    intercepts=[0.8, -0.1, 0.3],
    slopes=[[1.2, 0.7], [0.8, 0.4], [1.1, 0.3]],
    powers=[2, 0.4],
    link=1.8,
)
probability = u2oet_probabilities(
    [4, 5, 6],
    [40, 60, 80],
    efficacy=efficacy,
    toxicity=toxicity,
    association=0.6,
)
# Axes: agent 1, agent 2, efficacy, toxicity. Indices start at zero.
assert probability.joint.shape == (3, 3, 4, 4)
counts = np.zeros((3, 3, 4, 4))
counts[0, 1, 2, 0] = 3
print(probability.loglikelihood(counts))
# Example utilities; rows are efficacy categories, columns toxicity categories.
utility = np.array([[20, 15, 5, 0], [50, 35, 20, 10], [75, 60, 40, 20], [100, 90, 65, 35]])
print(probability.expected_utility(utility))
```

These example parameters are illustrative, not calibrated prior centers or
fitted estimates. The result is conditional on these parameters, not a posterior
recommendation. Native prior-vector ordering has not been established and is
not inferred from the example files.

Each outcome has 2–4 ordered categories and each agent 2–5 strictly increasing,
positive doses. An intercept and two positive slopes are required for each
continuation threshold. Intercepts do not require ordering. The two positive
powers modify the interior standardized doses while anchoring the endpoints.
`link` is the positive Aranda–Ordaz shape; one gives logistic continuation
probabilities. A small positive value approximates the complementary log-log
limit; zero is rejected.

`model="pds"` requires zero interaction. `model="cmi"` requires both powers to
be one and log centering. `model="pds+cmi"` allows PDS powers and an interaction
coefficient shared across thresholds for each outcome. Interaction can destroy
monotonicity. `centering="linear"` uses standardized dose minus one; the default
uses its logarithm. Association lies in [-1,1] and is the FGM copula parameter,
not Pearson correlation.

## Numerical implementation

Dose standardization first divides by the maximum dose to avoid overflow of
raw means and differences. Changing the dose measurement units preserves the
model. Dose ratios that underflow are rejected.

Continuation probabilities and category products are calculated in log space.
The Aranda–Ordaz calculation uses log-softplus and `expm1`, with asymptotic
branches below machine precision. FGM rectangle probabilities are evaluated
as nonnegative sums rather than subtracting four nearly equal CDF values.
This also preserves tiny joint probabilities at association endpoints ±1.
`log_joint`, `log_efficacy` and `log_toxicity` are immutable arrays. Ordinary
`joint` probabilities can underflow; use the log arrays for likelihood work.

The grouped likelihood excludes parameter-independent multinomial constants.
Counts describe fully observed efficacy/toxicity pairs; missing or delayed
outcomes are not handled. Empty counts give zero log likelihood. Impossible or
unrepresentably small observed probabilities can give negative infinity;
invalid finite inputs and overflowing linear predictors raise errors.

## Validation and remaining work

Two focused tests cover independent reference agreement and numerical extremes.
The independently written [R reference](../tests/fixtures/u2oet-reference.R)
evaluates the published equations directly, including four-CDF copula
rectangles, across five cases covering 2, 3 and 4 categories, the three model
variants, both centerings and association -1, 0 and 1. All 549 joint cells agree
with Python within 8e-16 absolute plus 2e-12 relative tolerance. This is
independent equation validation, **not native executable parity**.

Further checks cover marginal preservation, total probability, grouped
likelihoods, expected utilities, PDS monotonicity, dose rescaling by 1e±300,
link shapes 1e±300, and a rare/rare log probability near -3000 at association
-1. The latter has an analytic logistic/FGM reference. The R script generates
a fixed reference table; R is not a runtime or CI dependency.

Remaining coverage includes:

- GAO continuation probabilities and its Gaussian copula.
- Priors, pseudosampling prior centers, effective sample size calibration and
  posterior fitting with convergence assessment.
- Posterior acceptability, hybrid greedy/randomized allocation, escalation
  restrictions, cohort conduct and final selection.
- Calendar-based simulation, operating characteristics, scenario/native-file
  import and reports.

The original program is freely downloadable, but a general redistribution
license was not found in the inspected guide or README. This implementation
uses the published mathematics and does not assert a license for the vendor
software. The repository's release and licensing status is unchanged.
