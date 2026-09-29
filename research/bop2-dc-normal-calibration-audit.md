# BOP2-DC Normal endpoint finite-grid calibration

The primary paper's §2.1.2 specifies `Y_i ~ Normal(theta, sigma^2)` with a
Normal-Inverse-Gamma prior: `theta | sigma^2 ~ Normal(theta0, sigma^2/n0)` and
`sigma^2 ~ IG(a2, b2)`. The Python API requires these four values explicitly and
uses the exact conjugate posterior; it does not infer BCSTTE/app priors. The
marginal posterior for `theta` is Student-t. Section 2.2 supplies the strict
dual-criterion rule: an interim no-go requires both upper-tail probabilities to
fall below their information-scaled cutoffs; final go/no-go require both strict
comparisons in the same direction, with consider otherwise.

The complete-data replay and serial Normal-truth simulation implement those
rules at explicit sample-size looks. No accrual-time or missing-data model is
implied. The finite-grid optimizer reuses standardized Normal paths across both
truth means and all candidate settings. It enforces empirical false-go and
false-no-go limits and optional false-consider limits; Monte Carlo estimates
and MCSEs are not guarantees of the true operating characteristics. A separate
SeedSequence child generates independent validation paths. Validation failure
is reported without reselection.

Calibration is bounded by 10,000 candidates, 100,000 trials per stage, a
one-million-cell path budget, 50 million candidate/path work units, and a
two-million-cell combined evidence/workspace budget. These are Python resource
limits, not source defaults. Candidate order follows the product order of the
four caller-supplied grids, and exact ties retain the first candidate.

## Independent reference and resource verification

`tools/reference_bop2_dc_normal_calibration.R` independently updates the NIG
posterior using centered sufficient statistics, evaluates Student-t tails,
and replays absorbing decisions for each candidate. It receives a deterministic
truth-centered path tape, with exactly the same latent observations supplied
to both truth scenarios. No package function is used by the oracle.

`tools/check_bop2_dc_normal_calibration.py` verifies all 18 candidates, their
feasibility, selected parameters and independent holdouts under both objectives:
836 probability, enrollment and Monte Carlo-error summaries agree. CGR selects
candidate 0 and futile expected sample size selects candidate 4. The CGR choice
passes the 4% calibration false-go limit at `2/64` and fails holdout at `3/48`;
the result retains that choice and reports failure. ESS selection passes its
holdout. Adding `1e15` to every location input preserves candidate and holdout
results exactly.

The check takes .6839 seconds after Python imports, with Python peak resident
memory of 115.06 MiB and an R child peak of 83.72 MiB. Their sum, 198.78 MiB,
is a conservative combined upper bound; no Python swaps were reported.
Three focused worker tests pass. A dense 81-candidate grid with the default
5,000 calibration and 5,000 validation trials also passes for eight patients
and three looks. The dense-grid call takes .2041 seconds after imports;
the combined guide/smoke process peaks at 110.48 MiB with no swaps.

Review corrected three resource/numerical issues before public integration:
absolute simulated means no longer erase latent residual variation; calibration
paths are deleted before holdout allocation; and candidate batch width is
computed from remaining space after fixed results and path buffers. The last
correction prevents otherwise feasible default-count grids from being rejected
merely because a larger temporary candidate chunk was chosen. No additional CI
workflow or native optimizer/RNG parity is claimed.
