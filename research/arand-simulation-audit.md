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
