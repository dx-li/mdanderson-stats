# Multc Lean timing simulation audit

October 8 recovery update: native duration instructions establish clipped
exponential response times, shared paired follow-up and balked arrivals.
[The native duration audit](multc-native-duration-audit.md) records 34 executable
references. The historical uncertainty below is resolved for that separate
compatibility kernel; the existing observation-aware calendar policy remains
explicit. The October 9 [managed-model/workflow audit](multc-lean-model-audit.md)
subsequently completes Multc Lean #12, alongside implemented Multc99 #3.
The older source-uncertainty statements below describe the earlier calendar review.

This addition composes the existing paired-binary calendar replay with serial
random generation and bounded duration summaries. The tutorial represents
response/toxicity truth by the four joint categories (both, response only,
toxicity only, neither); users must supply their joint probabilities, so
association is retained and independence is never inferred. Accepted
probabilities are normalized once and the returned values match the sampling
distribution. The user guide
specifies exponential accrual and response observation with 95% probability by
a configured window, truncated at that window. Because “truncated” does not
resolve conditional truncation versus clipping at the window, the API requires
that timing convention explicitly. Nonresponses are observed at the window.
The guide does not specify a toxicity availability-time law, so this API
requires a fixed toxicity delay from the caller.

The first patient starts at time zero, matching the calendar replay's established
convention; all reported durations are measured from first enrollment, not study
opening. Subsequent exponential accrual gaps use the replay's accrual-open clock,
which freezes during pauses. This behavior and all observation timing are
explicit Python conventions; the implementation does not claim native RNG,
calendar policy, or report parity. Scope is Multc Lean and its existing Phase
IIa mapping, not the broader Multc99 multiple-event program.

Aggregate runs retain only child seeds, counts, and compact per-trial duration
vectors needed for means, MCSEs, and type-7 empirical intervals. Replay histories
are released after each trial. Work and peak storage are preflighted against
configurable limits, including a conservative allowance for wait/resume history
records at endpoint-availability events. Focused validation covers paired truth
and explicit endpoint timing, child-seed replay, candidate probability/resource
validation, and the analytic fixed-enrollment duration mean and MCSE. No full
Monte Carlo calibration or native executable comparison is claimed.

Root integration exposes both simulation functions and result/configuration
types. All four public guide blocks run successfully, including a 32-trial
example with mean duration 6.432968328 and MCSE 0.1991564651. The integrated
guide check took 1.922 seconds, peaked at 123.22 MiB RSS and reported zero
swaps. At that checkpoint native timing and random-stream parity were open. The
October 8 update above supersedes the timing uncertainty and catalog counts.
