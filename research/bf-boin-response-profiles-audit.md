# BF-BOIN response-profile simulation audit

The BARD guide's response model uses categorical factor profiles, factor-level
odds ratios, and dose-specific marginal response probabilities. The Python
`BARDResponseModel` records an explicit joint profile distribution and
calibrates each dose intercept so those margins are respected. Its conditional
probability evaluator is shared with this simulator.

`simulate_bf_boin(..., response_model=...)` now draws one profile at each
enrollment and samples the response conditional on that patient's dose and
profile. The stage-one helper validates that the model's population margins
match `true_response`, recomputes conditional probabilities from the stored
intercepts/profiles/odds ratios, and returns that canonical table. A supplied
joint event table has entries `P(DLT and response | dose, profile)` and is
checked against the two marginal probabilities with floating-point
roundoff-scale tolerance. Its conditional response draws use
`q / P(DLT)` after a DLT and `(p_response - q) / (1 - P(DLT))` otherwise.
When omitted, conditional independence given dose/profile is the explicit
Python policy; no dependence structure is inferred from narrative source
examples.

Profile row indices (zero-based), factor-category matrices (one-based),
marginal and sampled response probabilities, and final cumulative eliminated
masks are retained by trial in enrollment order. The no-profile path leaves
these optional fields absent and keeps the pre-existing per-dose response
random draw in place. This is a Python simulation extension; native BARD
software/RNG parity is not claimed.

Focused validation checks reproducible aligned ledgers under a nonuniform
profile mixture and supplied joint endpoints, stable default-path metadata,
and rejection of a mismatched response margin and rare-event Frechet violation
before advancing a caller-provided generator. The full BARD two-stage
operating-characteristic workflow remains separately documented in
[`bard-remaining-simulation-audit.md`](bard-remaining-simulation-audit.md).
