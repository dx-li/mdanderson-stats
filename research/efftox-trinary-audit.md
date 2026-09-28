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
