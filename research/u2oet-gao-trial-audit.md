# GAO calendar trial boundary

This Python driver applies the existing U2OET calendar, as-of snapshot, cohort,
no-skip, surplus, AR(2), and final-scope rules while fitting the generalized
Aranda–Ordaz response surface. The likelihood and marginal link follow the
GAO model; every Gaussian prior mean and SD is caller-supplied in the exact
coordinate order returned by `u2oet_gao_parameter_names`, including log link
parameters and the shared association Fisher-z. This is an explicit Python
prior convention, not a recovered native prior-file mapping.

The `scenario` argument supplies the complete dose-pair joint ordinal truth.
Arrival schedules may be generated from an exponential gap mean or passed as
an explicit schedule. Four per-patient uniforms encode allocation, joint
outcome category, efficacy delay, and toxicity delay. Exact tape inputs are
marked `replay_only`; they are retained to reconstruct data paths but are
rejected by the independent-replicate OC summarizer. Generated results retain
the original data and posterior substream seeds. Repeating the original input
RNG seed/state reproduces the full trial; the returned substream seeds alone
are not accepted as replacements for the original seed.

At each arrival, only already observed outcomes enter the fit: complete
joint-category counts, toxicity-only counts, and ignored pending outcomes are
reported separately. A new fit is run only when those sufficient counts
change. Likelihood evaluations and grid work are accumulated across all fits,
and a combined retained-array preflight bounds fit, posterior, tape, and
calendar history storage. Positive delays that cannot advance an absolute
calendar time are rejected rather than treated as observed immediately.

No native RNG, executable, or prior-mapping parity is claimed. The fitting
sampler's convergence diagnostics are estimates and do not establish chain
convergence. Native calibration's pseudo-prior construction and association
prior do not specify equivalent GAO-coordinate priors, so GAO prior
calibration remains open.

## Independent calendar check

`tools/reference_u2oet_gao_calendar.R` supplies Appendix-A probability and
utility calculations plus four explicit calendar ledgers. Inputs and outputs
are preserved under `tests/fixtures/u2oet-gao-calendar/`;
`tools/check_u2oet_gao_calendar.py` compares actual trial-driver results.
Run the R script with that fixture directory as its argument, then run the
Python checker with this checkout's `src` on `PYTHONPATH`.

All four cases pass: acceptable versus tried final selection, an absorbing
no-acceptable-pair stop, and cohort/surplus adaptive randomization. Nine patient
rows, six decision snapshots, four final snapshots and sixteen posterior
utility values agree, including toxicity-only and ignored pending outcomes.
Maximum absolute calendar/utility error is 6.66e-16. The root comparison took
1.596 seconds, peaked at 119.42 MiB RSS and reported zero swaps.

These fixtures fix model coordinates to isolate calendar and allocation
behavior. They complement the existing independent free-coordinate posterior
quadrature; they do not establish native prior mapping or trial operating
characteristics. The R ledger includes the initial arrival snapshot, while
the Python decision history starts before the second assignment; the checker
aligns those distinct records explicitly.

Four focused driver tests pass. Root additionally ran the existing U2OET
summary and affected CiBolus checks together: fourteen checks passed in
3.58 seconds, with 142.48 MiB peak RSS and zero swaps. No new CI workflow was
added and the full package numerical suite was not rerun.
