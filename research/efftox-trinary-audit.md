# EffTox continuation-ratio model

[Thall and Cook (2004), p. 686](https://www.johndcook.com/efftox.pdf)
specify four coefficients `(mu_T,beta_T,mu_E,beta_E)`, with both slopes positive:

```text
t = logistic(mu_T + beta_T*x)
q = logistic(mu_E + beta_E*x)
P(toxicity) = t
P(efficacy) = (1-t)*q
P(neither) = (1-t)*(1-q)
```

Here `q` is conditional efficacy given no toxicity. Marginal efficacy is
`(1-t)*q`; it must be used for desirability and efficacy thresholds. Outcomes
are mutually exclusive, so the current binary association model cannot
represent this likelihood. Centered log doses retain the existing zero-dose
convention. The original assignment rule permits toxicity-only exploration
at the lowest untried dose above the starting level and prohibits skipping
untried doses on escalation.

`tools/reference_efftox_trinary.R` creates independent base-R likelihood and
posterior fixtures. The reduced posterior has fixed positive slopes and
independent Gaussian intercept priors. Its likelihood factorizes into toxicity
versus no toxicity, and efficacy versus neither among non-toxic patients.
Each intercept posterior is integrated independently. The reference includes
the induced negative posterior efficacy/toxicity covariance. These references
prepare an implementation still in progress; native Windows parity is unverified.

The [2006 report, pp. 6–8](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/EffTox/NewTradeOffFunctions.pdf)
elicits a trinary contour through `(e0,0)`, `(em,tm)` and `(eh,th)` with
`eh+th=1`. In its Lp equation, the toxicity-axis intercept is an analytical
parameter, not necessarily a probability. The current binary contour's
restriction to an intercept below one must not be carried over silently.

## Independent trinary contour reference

The report's eliminated-intercept equation has a zero root that is not a
valid Lp exponent. For a stable positive-root construction, define
`a=1-e0`, `b=1-em`, `c=th=1-eh`, `d=tm`, then
`A=-log(b/a)`, `C=-log(c/a)`, `D=-log(d/c)`. The strict point ordering gives
`0<A<C` and `D>0`. Algebraic elimination yields

```text
g(p) = log(1-exp(-A*p)) - log(1-exp(-C*p)) + D*p = 0.
```

This is our derivation from the report, rather than its printed numerical
algorithm. Its limit at zero is `log(A/C)<0`, and its limit at infinity is
positive. The derivative is
`A/expm1(A*p)-C/expm1(C*p)+D>0`, since `u/expm1(u*p)` decreases in `u`.
Thus exactly one positive shape exists. Compute
`log(t*)=log(c)-log(1-exp(-C*p))/p`; `t*` may exceed one.

`tools/reference_efftox_trinary_contour.R` solves for `log(p)` with base R
and computes the intercept independently from the middle point. Its 48
fixture rows cover the stroke elicitation, known linear and elliptical
contours, and a concave contour with `t*=2`. Assertions check all elicited
utilities are zero, the ideal has utility one, and halving distance to the
ideal along a target ray gives utility one half. The stroke targets yield
`p=2.10360951613587` and `t*=0.165995389836415`. These are independently
computed modern Lp values for those targets, not original Windows outputs.
Fixture generation passed in 0.084 seconds. Python contour implementation
and comparison remain pending.
