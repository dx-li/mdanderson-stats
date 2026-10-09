# SYNERGY parametric response surfaces

The recovered [MD Anderson SYNERGY archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/SYNERGY/SYNERGY_V3.zip)
contains the Greco, Machado–Robinson, Plummer–Short and Carter S-PLUS/R models.
Python now provides their response functions, unweighted logit least-squares
fits, parameter uncertainty, single-drug/fixed-ratio curves and contour figures.
The response is **fraction surviving**, using the original fitting wrappers'
normalized control/base convention of one/zero. This differs from fraction affected.

SYNERGY remains partial: the separate
[semiparametric bootstrap](synergy-surface-bootstrap.md) interval convention
and its native workflow still need source recovery. The
[median-effect and interaction-index methods](interaction-index.md) are also available.

## Fit, predict and save a figure

```python
import numpy as np
from scipy.special import expit, logit
from mdanderson_stats import (
    synergy_parametric_response,
    fit_synergy_parametric,
    predict_synergy_parametric,
    plot_synergy_parametric,
)

single = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
dose1 = np.r_[single, np.zeros(5), np.tile([0.2, 1.0, 4.0], 3)]
dose2 = np.r_[np.zeros(5), single, np.repeat([0.2, 1.0, 4.0], 3)]
expected = synergy_parametric_response("machado", [-0.7, -1.3, 1.8, 0.9, 0.6], dose1, dose2)
response = expit(logit(expected) + 0.04 * np.cos(np.arange(dose1.size) * 1.7))
fit = fit_synergy_parametric(dose1, dose2, response, model="machado")
predicted = predict_synergy_parametric(fit, [0.5, 1.0], [0.5, 2.0])
figure = plot_synergy_parametric(fit, dose_ratio=2.0)
figure.savefig("synergy-machado.png", dpi=150)
```

The caller owns the returned matplotlib figure and should close it when done.
Figures require the `plot` extra. `dose_ratio` means dose2/dose1; the mixture
curve's horizontal axis is total dose. Observations within `1e-4` of the ratio
are displayed, following the original scripts. Contour levels and grid resolution
are explicit; no optimizer runs when plotting or predicting.

Inputs may be unsorted. Automatic initialization requires two distinct positive
single-drug doses for each drug. The supplied `initial_parameters` option uses
the parameter order below. Initial values affect nonlinear convergence.

## Models and parameter order

| Model | Parameters | Additive interaction | Direction for fraction surviving |
| --- | --- | --- | --- |
| `greco` | `m1, m2, dm1, dm2, alpha` | `alpha=0` | Positive alpha lowers survival; valid negative alpha raises it. |
| `machado` | `m1, m2, dm1, dm2, eta` | `eta=1` | `eta<1` lowers survival; `eta>1` raises it. |
| `plummer` | `beta0, beta1, beta2, beta3, beta4` | `beta4=0` | Positive beta4 lowers survival with decreasing marginal curves. |
| `carter` | `beta0, beta1, beta2, beta12` | `beta12=0` | Negative beta12 lowers survival at positive combination doses. |

Let `z=log(E/(1-E))`, `A=dose1/dm1`, and `B=dose2/dm2`. Greco solves

```text
A*exp(-z/m1) + B*exp(-z/m2)
  + alpha*A*B*exp(-z*(1/m1+1/m2)/2) = 1
```

The interaction amplitude is `A*B`, including for equal slopes. Machado solves

```text
A**eta * exp(-eta*z/m1) + B**eta * exp(-eta*z/m2) = 1
```

Plummer first solves `x = dose2 + dose1*exp(-beta2)*x**(-beta3)`, sets
`p=exp(beta2)*x**beta3`, and uses
`z=beta0+beta1*log(dose1+p*dose2+beta4*sqrt(dose1*p*dose2))`.
Carter uses `z=beta0+beta1*dose1+beta2*dose2+beta12*dose1*dose2`.

Python solves the implicit models on logarithmic scales and checks their defining
equations. Both-zero doses have theoretical survival one in the three log-dose
models. A fitted model with observed controls preserves the original plots'
observed control mean at that coordinate. It is an explicit display convention;
controls do not enter those three fits. Carter predicts its fitted intercept.

## Estimation and uncertainty

Greco, Machado and Plummer omit both-zero dose controls and require all remaining
responses strictly between zero and one. Carter omits responses exactly zero
or one and retains interior control observations. `included_rows` records this
selection. Fits use unweighted squared errors on the **logit** scale.

The result retains initial/fitted parameters, covariance, standard errors,
t statistics, two-sided t p-values, parameter correlations, residual sum of
squares, residual standard error, residual degrees of freedom, original data
and fitted responses. Inference uses the local least-squares Jacobian and
residual variance with `n-p` degrees of freedom. It assumes the logit-scale
error model and provides neither binomial-likelihood inference nor a guarantee
of global optimum. A sign alone does not establish statistical significance.

Carter is linear on this scale and is solved directly. The other models use
bounded least squares; Greco reparameterizes its interaction to preserve a
unique monotone root on the training dose grid, converting the Jacobian back
to the reported physical parameters before calculating covariance. Antagonistic
Greco predictions at sufficiently large new doses may violate this condition
and are rejected. The native unconstrained routine can return endpoint artifacts
or ambiguous roots in that region.

Supported numerical bounds are explicit:

| Parameters | Bounds |
| --- | --- |
| Greco/Machado slopes | `[-1000, -1e-6]` |
| Greco/Machado median doses | `[1e-12, 1e12]` |
| Greco alpha | `[-1000,1000]`, plus dose-dependent root validity |
| Machado eta | `[1e-6,1000]` |
| Plummer beta0 / beta1 / beta2 | `[-10000,10000]` / `[-1000,-1e-6]` / `[-100,100]` |
| Plummer beta3 / beta4 | `[-0.999,1000]` / `[-1.999999,1000]` |
| Carter coefficients | `[-10000,10000]` |

Training is limited to 500 records, prediction to 100,000 points, and fitting to
2,000 actual residual evaluations including finite-difference Jacobian calls.
Plot resolution is 10–150 points per axis. Singular/ill-conditioned information,
zero residual variance, active optimizer constraints and unrepresentable roots
are rejected. These numerical policies are Python choices; exact native nls
optimizer and endpoint parity are not claimed.

## Validation and sources

The unchanged author response kernels supply 138 predictions, with maximum
absolute response discrepancy `1.25e-15`. Nine original-wrapper R fits cover
the published data, synthetic data for every model, and an antagonistic Greco
example. Parameter estimates and their uncertainty agree within declared
tolerances for R's default nls stopping rule. The original scripts are external
reference inputs; mathematical equations are independently implemented in Python.
See the [audit](../research/synergy-parametric-audit.md) and
[archive/source checksums](synergy-parametric-source.json).
