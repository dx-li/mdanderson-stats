# SYNERGY remaining response surfaces

The [official catalog entry 18](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/18)
lists four one-parameter response surfaces from Lee et al. (2007),
DOI `10.1080/10543400701199593`, and the 2008 semiparametric method. Existing
`interaction_index.py` covers the separate median-effect/Loewe interval methods.
These additional surfaces remain unimplemented.

## Semiparametric source contract

The [Kong–Lee primary manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC5096313/)
and [publisher record](https://academic.oup.com/biometrics/article/64/2/396/7331586)
describe `Y = F_p(d1,d2) + f(d1,d2) + error`. Fit additive baseline `F_p` from
marginal observations, then a natural bivariate thin-plate spline to
`(Y-F_p)*I(d1!=0 and d2!=0)`. Knots are distinct observed dose pairs.

```text
f(d) = gamma0 + gamma1*d1 + gamma2*d2 + sum_k nu_k*eta(||d-knot_k||)
eta(r) = r²*log(r²)/(16*pi), eta(0)=0
Omega[k,l] = eta(||knot_k-knot_l||)
T*nu = 0, where T has rows 1, knot1, knot2
penalty = lambda*nu'*Omega*nu
```

The smoothing parameter uses mixed-model REML and spline predictions use BLUP.
Two baseline classes occur: transformed response linear in raw doses, or in
log doses with varying relative potency. Neither alone establishes full coverage.
The two-stage wild bootstrap uses Mammen weights, residuals from half the fitted
smoothing parameter and fitted values from twice it. Intervals are pointwise.
For a decreasing response, negative departure denotes synergy.

An initial source summary incorrectly described `Omega` as raw distances;
the primary equation uses the kernel above.

## Baselines and bootstrap details

The response scale is `Y=g(E)`, with a caller-selected monotone transformation.
For raw doses, fit a common-intercept linear model to marginal records only
(`d1==0 or d2==0`); its additive prediction is `beta0+beta1*d1+beta2*d2`.

For log-dose marginal curves `beta0+beta1*log(d1)` and
`alpha0+alpha1*log(d2)`, define

```text
gamma1 = (alpha0-beta0)/beta1
gamma2 = alpha1/beta1-1
u - gamma1 - gamma2*log(d1*exp(-u)+d2) = 0
rho = exp(u)
F_p = beta0 + beta1*log(d1+rho*d2)
```

With positive combination doses and same-direction slopes, this root is
unique: its derivative is `1+gamma2*d1/(d1+exp(u)*d2)>0`. Use stable log sums.
The source gives no offset for a both-zero dose in this model.

Bootstrap all observations on the transformed scale:

```text
residual_i = Y_i - F_p_hat(d_i) - f_half_lambda(d_i)
Y_i_star = F_p_hat(d_i) + f_twice_lambda(d_i) + residual_i*weight_i
weight values = (1-sqrt(5))/2, (1+sqrt(5))/2
probabilities = (sqrt(5)+1)/(2*sqrt(5)), (sqrt(5)-1)/(2*sqrt(5))
```

Each replicate refits the marginal baseline and REML spline. The pointwise
normal interval centers on the original fitted departure. The exact bootstrap
SD denominator/centering remains unresolved in primary rendered equations;
do not silently adopt a secondary transcription's fitted-center RMS formula.

## Source availability

Catalog metadata records `SYNERGY_V3.zip`, download 293/version 154, 25 KB.
No original archive is present locally. The official download endpoint returned
a web cache miss, and the shell download attempt failed DNS resolution.
Direct PMC retrieval returned a browser challenge; indexed primary full text
provided the mathematical evidence above. These retrieval limits do not
establish native code behavior, reproduction of case studies or archive terms.
The four 2007 parametric models still need their own exact source contracts.

The [2007 primary abstract](https://pubmed.ncbi.nlm.nih.gov/17479394/) confirms
four one-parameter Loewe-based surfaces and accompanying S-PLUS/R listings,
but does not specify their equations or fitting/inference procedures. Publisher
full text returned HTTP 403 during the bounded source review. A later review
names Greco, Machado–Robinson, Plummer–Short and Carter models; those names and
parameter signs alone cannot verify the exact SYNERGY parameterizations.
Third-party implementations remain provenance leads, not native-contract proof.

## Independent spline reference

`tools/reference_synergy_surface.R` uses a raw-dose common-intercept marginal
baseline and nine distinct dose pairs, with four replicated observations. For
explicit smoothing parameters 0.02 and 0.2, it directly solves the constrained
augmented thin-plate system rather than using a mixed-model eigendecomposition:

```text
[K + lambda*diag(1/count), T] [nu   ] = [mean residual]
[T',                         0] [gamma]   [0            ]
```

The script verifies affine-nullspace constraints and the penalized normal
equations, and retains predictions at observed and additional dose pairs.
`synergy-surface-*.csv` fixtures retain inputs, baseline/affine coefficients,
residual sums of squares, roughness and predictions. Base-R generation ran in
0.085 seconds. This supplies an independent reference for the Python spline
fit, without claiming native smoothing selection, bootstrap or case-study parity.

The same R script also profiles REML through a direct full covariance matrix,
with explicit determinants and GLS projection rather than the Python
error-contrast eigendecomposition. Its interior optimum is
`lambda=6.60856248525525e-5`, with unscaled objective `-24.5152377989773` and
residual variance `1.38755525407976e-5`. The development core agrees within
`4.68e-9` relative error for lambda, `1.32e-13` absolute objective error and
`2.80e-9` relative variance error after accounting for response scaling.
All 72 fixed-lambda baseline/surface/total predictions match within `3.78e-15`.
These calculations check spline and REML mathematics through different algebra;
they do not establish native software random streams or bootstrap inference.

The additional bounded source check through Europe PMC full-text XML, NCBI
BioC XML and PMC XML view was inaccessible. Indexed searches exposed neither
the bootstrap SD equation nor an original author-code supplement. The exact
SD centering and denominator remain unresolved; that inference gap is separate
from the now independently checked surface-fitting calculations.
