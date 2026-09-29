# ARAND serial simulation and operating characteristics audit

The ARAND 5.2 user guide describes binary response probabilities, exponential
survival means or medians, a monthly mean accrual rate, and arm-level selection,
removal, and enrollment summaries. `arand_simulation.py` generates arrivals as
a homogeneous Poisson process beginning at time zero, so the first arrival is
also an exponential gap. Exponential event-time scale is the supplied mean, or
median divided by `log(2)`.

Simulation is serial. Independent child seeds make each replicate replayable;
only trial seeds, aggregate counts, and duration vectors survive each replay.
Candidate arrivals and aggregate work/storage have explicit caps. A trial whose
candidate tape cannot certify enrollment or calendar-horizon completion raises
instead of being reported as complete. Arm suspension, permanent futility, and
displacement by an early winner are separate Python events because the guide
does not define how its native “dropped early for any reason” probability is
formed. Patient-count intervals use NumPy's linear empirical quantile method
(type 7); the guide does not specify the quantile convention. These outputs
therefore provide source-informed operating characteristics, not exact native
RNG or report parity.

Focused validation comprises the simulator test for replaying the sole
single-arm binary replicate from its returned seed and for undefined one-trial
MCSEs, plus candidate-budget exhaustion under both duration-precedence policies.
Together with the existing ARAND calendar replay tests, the bounded check ran 13
tests. It does not establish broad Monte Carlo calibration or native numeric
parity.

## Integrated numerical checks

Root integration passed all 15 affected posterior, calendar and simulator tests.
Three two-arm exponential trials gave the same assignments, event counts and
selection after converting true means and inverse-gamma prior scales to median
units; posterior probabilities agree within 2e-10 absolute and 2e-9 relative
tolerance. An independent Poisson identity check simulated 512 single-arm
duration-limited trials with rate 2 and duration 0.25. Mean enrollment was
0.5234375 versus the exact 0.5, and the zero-enrollment proportion was
0.58203125 versus exp(-0.5)=0.6065306597; both deviations are within six
analytical Monte Carlo standard errors. Every selection/no-winner event was
accounted for, and the decision/completion durations matched the configured
0.25/1.25 exactly. This is a bounded generator check, not native calibration.

These checks together took 1.639 seconds at 147.12 MiB peak RSS, zero swaps.
The public two-arm binary simulation/seed-replay guide also passed (1.558 seconds,
122.50 MiB, zero swaps). Targeted type and lint checks pass; root applied the
repository formatter to the new module and tests. No dependency or CI additions
were needed. Numerical runs remained serial with one BLAS/OpenMP thread.
