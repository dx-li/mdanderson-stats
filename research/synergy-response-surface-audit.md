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
the primary equation uses the kernel above. Before implementation, recover
the exact baseline transformations, potency equation and bootstrap refitting
sequence; do not replace REML with GCV or ignore marginal-fit uncertainty.

## Source availability

Catalog metadata records `SYNERGY_V3.zip`, download 293/version 154, 25 KB.
No original archive is present locally. The official download endpoint returned
a web cache miss, and the shell download attempt failed DNS resolution.
Direct PMC retrieval returned a browser challenge; indexed primary full text
provided the mathematical evidence above. These retrieval limits do not
establish native code behavior, reproduction of case studies or archive terms.
The four 2007 parametric models still need their own exact source contracts.
