# BCSTTE log-odds-rate family: sourced numerical core

The cached BCSTTE User's Guide §4.6 gives the family label, its scale/shape
coordinates, log-logistic special case, Weibull limit, and moments. Its displayed
survival formula omits a factor of `c` inside the bracket. The independent
primary article by Shen and Thall defines the generalized odds-rate model as

```text
S(t; lambda, phi, c) = [1 + c * (t/lambda)^phi]^(-1/c),
lambda > 0, phi > 0, c > 0.
```

It explicitly identifies `c=1` as log-logistic and the `c -> 0` limit as
Weibull, `exp(-(t/lambda)^phi)`. The article's covariate form is
`[1 + c*(t/lambda)^phi*exp(b'Z)]^(-1/c)`. See Shen and Thall, “Parametric
likelihoods for multiple non-fatal competing risks and death,” *Statistics in
Medicine* 17 (1998), 999–1015, §2, especially equation (7),
[author-hosted primary article](https://odin.mdacc.tmc.edu/~pfthall/main/SIM_05_Shen-Thall%201998.pdf).
The relevant evidence is §2, equation (7), and the preceding paragraph (journal
pages 1001–1002). The equation, positive parameter domain, and the
two limiting cases are reproduced above. Source checked 2026-10-03. No PDF was
downloaded for this task, so no local checksum applies; the cached BCSTTE guide
and primary article remain distinct sources.

Writing `d = log(t/lambda)`, `h = phi*d`, and `q = log(1+c*exp(h))/c`, the
right-censored per-row contributions are `log(S)=-q` and
`log(f)=log(phi)-log(lambda)+(phi-1)*d-q-log(1+c*exp(h))`. The CDF is
`1-exp(-q)`. The private helper uses `log(shape)`, `log(scale/time_reference)`,
and `log(c)` with relative log times, returning the density on centered time
units. A caller restores `-log(time_reference)` once per observed event; a
censored survivor term receives no time-density Jacobian.

For small `c*exp(h)`, the helper evaluates the continuous Weibull limit without
subtracting nearly equal values or dividing a tiny `log1p` result by `c`.
Positive `c` is required for the finite family; `c=0` is a limiting case, not
a finite parameter value. No native BCSTTE prior, fitting method, or numerical
software parity is claimed by this module.

The guide's moment statement is also imprecise: for integer/general positive
order `r`, the sourced survival implies

```text
E[T^r] = r * lambda^r * c^(-r/phi) / phi
         * B(r/phi, 1/c - r/phi),  phi > r*c.
```

The condition `phi>c` suffices for the mean (`r=1`) but not for the listed
second moment (`r=2`), which requires `phi>2c`. This module intentionally does
not expose a moment API.

## Independent reference

`tools/reference_log_odds_rate.R` uses direct base-R density, survival and CDF
equations and deterministic tensor Gauss-Legendre integration under an explicit
proper correlated Normal prior on the three log coordinates. The prior is a
Python/API convention, not a source claim. It covers complete and right-censored
likelihoods and reports paired posterior summaries. Native prior/fitter parity
and Johnson diagnostics for censored data remain outside scope.
