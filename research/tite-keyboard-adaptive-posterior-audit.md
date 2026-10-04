# TITE-Keyboard adaptive timing posterior

This is a source-derived Python extension, not an implementation of the app's
native adaptive-sampler conventions. The pinned methodological-paper record is
in [`docs/tite-keyboard-sources.json`](../docs/tite-keyboard-sources.json); the
cached primary text is `research/raw/TITE-KEYBOARD/paper.txt` (arXiv:1807.08393).

Section 2.1 (paper-text lines 220–231) specifies independent dose probabilities
`p_j ~ Beta(1,1)`. Section 2.2 (lines 280–344, equations 2.3–2.4) specifies the
observed-data likelihood: an observed DLT contributes `p_j`; a completed
non-DLT contributes `1-p_j`; and a pending DLT-free assessment at follow-up
`u_i` contributes `1-p_j F(u_i/τ | λ,γ)`. Section 2.3 (lines 469–487) specifies
the shared conditional time model `T_i/τ | DLT, λ,γ ~ Beta(λ,γ)`, a prior
`π(λ,γ)`, and the adaptive weight as the posterior expectation of the Beta CDF.
The text gives independent `Gamma(0.1,0.1)` as an example prior and uses
independent `Gamma(0.5,0.5)` in its sensitivity analysis (lines 808–818); it
does not mandate a Gamma parameterization or an app default.

For explicit independent Gamma(shape, rate) priors, the posterior for the
shared timing shapes is obtained by multiplying their prior density, the Beta
densities of observed normalized DLT times, and the dose-specific likelihoods
with `p_j` integrated out. For dose `j`, with `y_j` observed DLTs, `m_j`
completed non-DLTs, and pending follow-up fractions `v_i`, the marginal factor
is

```text
I_j(λ,γ) = ∫₀¹ p^y_j (1-p)^m_j ∏ᵢ[(1-p) + p S(v_i | λ,γ)] dp,
S(v | λ,γ) = 1 - I_v(λ,γ),
```

where `I_v` is the regularized incomplete Beta CDF. This positive mixture
integral includes pending survival and is evaluated as a nonnegative Beta-mixture
recurrence; it avoids alternating polynomial coefficients. Each observed event
also contributes `BetaPDF(t_i/τ | λ,γ)`; the omitted `1/τ` density factor is
constant in the timing parameters and cancels from their posterior. Completed
non-DLTs contain no conditional event-time observation.

The sampler uses a serial componentwise random-walk Metropolis chain on the two
log-shapes and summarizes posterior-mean CDF weights. MCMC iteration count,
proposal adaptation, independent Gamma priors, R-hat and weight-MCSE thresholds
are explicit Python policies. Reported diagnostics are estimates, not proof of
convergence. The decision wrapper refuses to decide when its configured
diagnostic checks fail. It then injects posterior-mean weights into the existing
approximate effective-binomial Keyboard likelihood (equations 2.5–2.7); it does
not use the exact joint posterior over toxicity probabilities as a replacement
decision rule.

The paper recommends its uniform and piecewise-uniform schemes for general use
because the adaptive version showed minimal operating-characteristic improvement
while timing data are sparse. This extension does not switch any default and
makes no native application parity claim.
