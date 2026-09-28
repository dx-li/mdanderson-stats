# BaCIS classification-model DIC

## Which model is scored

The archived bacistool 1.0.0 `bacisCheckDIC` calls `ModelOne` in
`R/internal.R`. That function creates `model1Str`, the first-stage independent
two-component logistic-normal classification model. It then calls
`dic.samples` with five chains and returns the sum of the per-node deviance
and penalty. The within-cluster second-stage borrowing model is not the model
scored by this diagnostic. Reusing `bacis_fit.probability_samples` would
therefore calculate a different diagnostic.

The rjags [source](https://github.com/cran/rjags/blob/master/R/dic.R) defaults
to the `pD` penalty. Its [documentation](https://cran.r-universe.dev/rjags/doc/manual.html)
identifies the Plummer definition, which uses independent chains, rather than
the posterior-mean plug-in definition or half the deviance variance.

## Pinned JAGS calculation

JAGS 4.3.2 source was inspected at commit
`75272ee4eaa3e24fe49b1c49382dfa665048b24e` (`release-4_3-2`). File URLs and
local source hashes are recorded in [bacis-dic-sources.json](bacis-dic-sources.json).
Reference files are kept under the ignored `research/raw/BaCIS/jags-4.3.2/`
directory and are not part of the Python distribution.

[`PDMonitor`](https://github.com/jags/JAGS/blob/75272ee4eaa3e24fe49b1c49382dfa665048b24e/src/modules/dic/PDMonitor.cc)
averages the two directed KL divergences over unordered pairs of chains, with
unit weights and a factor of one half. Its header sets the default scale to
one, and `PDMonitorFactory` uses that default. The population target is thus
the expected KL divergence between two independent posterior parameter draws.

[`DBin::KL`](https://github.com/jags/JAGS/blob/75272ee4eaa3e24fe49b1c49382dfa665048b24e/src/modules/bugs/distributions/DBin.cc)
uses the analytic binomial KL with fixed patient count. `ScalarStochasticNode`
uses this analytic value for untruncated binomial nodes; it does not need the
fallback simulation count. [`DevianceMean`](https://github.com/jags/JAGS/blob/75272ee4eaa3e24fe49b1c49382dfa665048b24e/src/modules/dic/DevianceMean.cc)
averages minus twice the full observed-node log likelihood over chains and
iterations. The binomial coefficient is included.

## Deterministic reduction

For one subgroup, let `eta=logit(p)` under the first-stage posterior and let
`(p', eta')` be an independent draw. The source's penalty has the exact
population identity

```text
pD = n E[KL(Bernoulli(p) || Bernoulli(p'))]
   = n {E[p*eta] - E[p] E[eta]}.
```

The expected deviance is

```text
Dbar = -2 {log choose(n,y) + y E[log(p)] + (n-y) E[log(1-p)]}.
DIC = sum_i (Dbar_i + pD_i).
```

These are independent one-dimensional integrals for each of the two posterior
mixture components. The existing BaCIS classification evidence supplies their
posterior weights. Centering `eta` within each component and combining
within-component covariance plus between-component mean differences avoids
subtracting large raw moments. Stable log probabilities must be retained even
if `p` rounds to zero or one.

The latent normal sign variable controls only the equally weighted prior
mixture membership, so its positive precision does not affect these integrals.
The adaptive classification cutoff and the second-stage priors likewise do not
enter this first-stage model score.

`tools/reference_bacis_dic.R` independently evaluates these integrals with
base-R quadrature for 20 subgroup cases. These include the native example,
unequal sample sizes, diffuse component priors, zero/all-response extremes,
and a deliberately separated posterior mixture. The resulting per-subgroup
mean deviance, penalty and sum are saved in
`tests/fixtures/bacis-dic-reference.csv`. An explicit directed-binomial-KL
sum on a three-point probability distribution also agrees with the covariance
identity to within `1e-13`.

Reference generation completed successfully without JAGS. Agreement with a
Python diagnostic remains pending implementation. These references target
the population expectations of the inspected JAGS monitor, not its particular
finite chain sequence; a separated mixture can also violate the asymptotic
assumptions used to interpret DIC.
