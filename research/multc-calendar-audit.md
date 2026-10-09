# Multc Lean calendar replay audit — September 28, 2026

October 8 recovery update: native duration instructions establish clipped
exponential response times, shared paired follow-up and balked arrivals.
[The native duration audit](multc-native-duration-audit.md) records 34 executable
references. The historical uncertainty below is resolved for that separate
compatibility kernel; the existing observation-aware calendar policy remains
explicit. Multc99 #3 is now implemented; Multc Lean #12 remains partial.

Luna implemented the explicit-timing replay in `d691b56` and numerical time
guards in `898e4be`, integrated as `52bd380` and `3c84b2a`. It reuses established
marginal stopping bounds. The scope is paired binary Multc Lean and its existing
Phase IIa mapping, not general Multc99 multiple-event designs.

## Source contract

The official [logistics guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcLogistics.pdf)
and statistical tutorial section 2.3.2 permit enrollment while outcomes are
pending if no possible pending completion could alter continuation. Monitoring
occurs only at scheduled cohort looks after minimum enrollment. Otherwise,
accrual suspends until the stop/continue decision resolves.

The [user guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcUsersGuide.pdf)
section 3.3 specifies exponential inter-arrivals and response observation from
an exponential distribution with 95% probability by the window, truncated to
that window. Nonresponse is known at the window end. It does not specify a
separate toxicity availability-time law. The replay requires separate explicit
endpoint delays and does not infer such a law. Freezing the accrual-open clock
during pauses and inclusive exact-time observations are documented Python
conventions; native queue behavior and random-stream parity are unverified.

## Review and validation

Read-only review verified that look-ahead uses hypothetical pending count
ranges rather than actual future bits, exact-time observations are available,
pretrial screening reuses the existing contract, and cap completion is distinct
from adverse-outcome stopping. A detected metadata defect was corrected: the
cap look now retains its actual design bounds while its action remains
`cap_complete`. Stop labels describe guaranteed causes; unresolved other
endpoints can still cross their boundaries later.

`tools/reference_multc_calendar.R` independently evaluates beta tails and
enumerates possible pending count completions. Five scenarios cover suspension
and resumption, continuation with pending data, toxic stopping before complete
follow-up, a certain response stop with unresolved toxicity, and tied endpoints
that establish both stopping causes. All 18 patient rows, 12 look rows and five
summary rows agree exactly with Python. The comparison took 0.0051 seconds
after import, peaked at 117.64 MiB and had no process swaps.

After the cap-bound fix, all three focused calendar checks passed in 1.49
seconds. The worker also ran six existing Multc reference checks; its nine
checks passed in 1.87 seconds before the metadata-only correction. Targeted
Ruff, formatting and mypy checks passed. Numerical jobs used one BLAS/OpenMP
thread; no broad suite, large Monte Carlo run, dependency installation or CI
change was performed. The source catalog remains partial: aggregate duration
simulation and native configuration/report/protocol workflows remain open.
