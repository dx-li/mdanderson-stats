# BayesFactorTTE calendar extension audit

The locally recovered 2012 user guide specifies the continuous-monitoring
Bayes factor, inferiority/superiority probability cutoffs, maximum patients,
accrual rate, repetitions and true-median scenarios. Its example report gives
stopping probabilities for the alternative and null, mean patients treated,
and 10th/90th patient-count percentiles. The guide does not specify arrival
generation, interim-check schedule, post-maximum monitoring or final follow-up
timing. This implementation therefore exposes deterministic event/arrival
tapes and check/final times, and labels all simulation timing as an explicit
Python convention rather than native calendar parity.

Replay includes arrivals tied with checks before checking evidence, counts
events tied with checks or censor times, and treats events as observed on an
event/censor tie. The first interim boundary crossing stops new enrollment;
the supplied final time remains fixed so enrolled patients can accrue more
follow-up. The optional independent censor duration defaults to positive
infinity. Simulation uses a time-zero first arrival, exponential later gaps,
caller-supplied absolute check times and exponential event durations with the
caller-supplied median. Each trial has separate accrual/outcome seed streams,
and returned seed pairs allow exact trial replay.

The serial simulation defaults to at most 20,000 patient/check work units and
1,000 Bayes-factor evaluations. Callers may explicitly raise those limits up
to hard ceilings of 1,000,000 work units and 100,000 evaluations. Retained
outputs plus the current trial and conservative adaptive-quadrature scratch
must fit the requested storage budget, capped at 128 MiB; no trial histories
are retained across replications. Work and memory bounds are checked before
random tapes are generated.

After integration, focused calendar tests and the existing BayesFactorTTE
numerical tests passed: 10 tests in 2.37 seconds with warnings treated as
errors. Peak RSS was 144.73 MiB, with zero process swaps. This includes the
regression for positive follow-up that is lost at a very large calendar
horizon. An independent three-patient ledger also verifies exact event counts
and time-on-test at three interims and the final look, including an event/censor
tie. Ruff check/format, targeted mypy and diff checks passed.
