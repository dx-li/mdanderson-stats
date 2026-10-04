# Bayes Factor TTE report audit

The cached BayesFactorTTE version 1.0.0 User's Guide (May 21, 2012), §§2.2–2.3
and Appendix, specifies the report's input and output fields. Inputs include a
seed, maximum patient count, null and alternative medians, inferiority and
superiority cutoffs, accrual rate, repetition count, an optional stopping
boundary table, and true-median scenarios. The Appendix displays true medians
with corresponding exponential means, probabilities stopping for the
alternative and null, and average patients enrolled with 10th/90th percentiles.
The boundary table indexes by event count and reports maximum total time on
test for inferiority and minimum total time on test for superiority.

The Python report composes the existing `bayes_factor_survival_boundaries` and
`simulate_bayes_factor_survival` kernels. Its terminal stopping summaries
classify the first early boundary crossing as terminal; only trials without an
early crossing use the final-monitor decision. This differs from directly
reporting final-monitor probabilities, which can change after additional
follow-up. Early and final-monitor probabilities are therefore shown
separately alongside terminal probabilities. Scenario simulations run
serially with deterministic child seeds, and only compact summaries are kept.

The guide does not define the native arrival law, interim-check schedule, or
final-follow-up timing. The report consequently requires explicit Python
`check_times` and `final_followup`, and labels its timing policy. The guide's
boundary table uses integer days while the existing solver returns continuous
roots; the report preserves caller-unit roots and makes no day-rounding claim.
The guide's suggested limits include at most 500 patients, medians up to 24
months, and accrual up to 100 patients per month. Python retains its own
validated kernel limits and arbitrary consistent time units rather than
reproducing the desktop UI limits.

The report checks worst-case simulation work and Bayes-factor evaluations
aggregated across all scenarios before starting any simulation. Optional
boundary rows are bounded by an explicit requested-row cap. Boundary quadrature
and root-solving costs are adaptive and the current kernel does not expose an
exact evaluation counter, so the row cap is not represented as an exact CPU
budget.
