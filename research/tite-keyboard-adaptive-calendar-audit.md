# Adaptive TITE-Keyboard calendar integration audit

This extension connects the source-defined adaptive timing model to calendar
trial replay and bounded serial operating-characteristic simulation. It is a
Python implementation of the paper's model and existing Python decision
semantics; it does not claim native TITE-Keyboard sampler, prior, RNG, or report
parity.

The cached primary manuscript is `research/raw/TITE-KEYBOARD/paper.txt`. Section
2.3, equations (2.5)–(2.7), gives the effective binomial count formed from
observed patients and posterior mean pending weights. Section 2.3 also defines
the adaptive weight as the posterior expectation of
`F(u_i / W | lambda, gamma)` under the shared timing-shape posterior. That
posterior uses observed DLT-time densities and pending DLT-free survival
factors. The model and its numerical implementation are documented in
`research/tite-keyboard-adaptive-posterior-audit.md` and
`docs/tite-keyboard-adaptive.md`.

At each calendar decision this implementation conditions only on events already
observed at that time, expressed as event age since enrollment, and the pending
follow-up ages then available. It never passes the latent future delay matrix
to the adaptive fitter. No fit is needed when there are no pending patients.
The ordinary controller also runs without a fit when an immediate safety stop,
current-dose elimination, or pending-fraction suspension fixes the action
independently of adaptive weights. Endpoint event ages are rejected only when
they would enter the adaptive timing density, which is defined on the open
assessment interval.

The simulation keeps the generating event-time scenario separate from the
adaptive analysis model. Timing-shape priors are explicit independent
Gamma(shape, rate) pairs; the paper supplies examples but does not establish a
native default parameterization. The simulation uses separate deterministic
outcome and timing-fit random streams, requires an integer seed for this new
mode, records the derived seeds, and processes trials serially. A per-fit
conservative work check and one aggregate work cap cover repeated fits at
ordinary and suspension-triggered decisions. Failed sampler diagnostics
propagate as errors, so incomplete trials do not enter simulation summaries.

The paper recommends uniform or piecewise-uniform weights for general use when
timing data are sparse; adaptive weighting remains an explicit option and does
not change existing defaults. Posterior diagnostics are empirical checks, not
proof of MCMC convergence.
