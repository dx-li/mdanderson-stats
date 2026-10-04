# Parallel Phase I/II source and output crosswalk

The two archived programs are separate implementations with different
workflows. This batch adds a named, reproducible Python report for the four-arm
C program's source-defined probability input, and supplements the six-dose
calendar OC with source-indexed duration summaries. It does not claim a shared
model or byte-for-byte native output.

## Four-arm C program

In `research/raw/P12Xuelin/extracted/SwatiBiswasCode/main_4arms_nobugs.c`,
the run reads a seed from `in_seed`, derives a second seed from system time,
then appends the seed values to the output (lines 24–45). The probability
input is four response probabilities followed by four toxicity probabilities
(lines 64–73). The output row includes selected arm, reported probability,
total enrollment, and per-arm enrollment/toxicity/response/admissibility
(lines 283–306). The report factory consumes already validated Python
four-arm scenario requests and reuses `simulate_parallel_phase12_oc`; it does
not reconstruct native seed behavior or the C executable's full row format.

The wrapper parser accepts exactly eight probabilities and maps the first
four to response, last four to toxicity. The source file has no scenario
label, replication count, or explicit reproducibility seed; these are required
separately. C's design constants and stopping/allocation rules remain in the
existing source-backed implementation and are not configurable through the
native probability file.

## Six-dose C++ calendar program

The source configures simulation replicates and seed behavior in
`research/raw/P12Xuelin/extracted/Phase12Xuelin/Simulations/SimulationsMP.cpp`
(lines 103–132); `SetupTrial` defines the six-dose calendar settings (lines
238–267), and `SimulationCases` contains its seven hard-coded truth scenarios
(lines 310–405). Python exposes a caller-configurable six-dose simulation and
bounded OC interface rather than copying that fixed list.

`DFKernel/TrialDesign.cpp::InitSims` defines seven duration order-statistic
indices using integer division (lines 193–204). `TallySim` records one
enrollment stop time for every simulated replicate (lines 216–236), including
early stopping paths. `TallyScenario` converts days to months with 12/365,
computes the mean and `E[D²]−E[D]²`, sorts durations and reads those exact
indices (lines 239–269). The member called `m_dStdDurations` and printed as
"Std" therefore contains a population variance in month-squared units, not a
standard deviation. Python exposes that quantity under an explicit variance
name, retains the existing day-scale mean/MCSE, and returns NaN for order
indices that are unavailable at small replicate counts instead of reading an
uninitialized native slot or inventing an interpolated quantile.

The native parameter output at `TrialDesign.cpp::PrintResults` (lines 951–987)
also includes posterior parameter and probability summaries. `Kernel::Integrate`
sets parameter summaries to the posterior mode and diagonal of its fitted
Laplace covariance (`DFKernel/DFKernel.cpp`, lines 240–243), not posterior
means and variances from MCMC draws. The trial aggregator computes the across-
dataset mean of modes and the mixture variance
`E[conditional_variance + mode²]−E[mode]²`
(`TrialDesign.cpp`, lines 216–224 and 275–281). No-fit trials still tally the
parameter summary arrays; `TrialDesign.h::Clear` resets current probabilities
but not `m_vPostMeans`/`m_vPostVars`, so these values can be stale after an
early stop. Python does not imitate this behavior.

For kernel probability vectors, `TrialDesign::SetupTrial` allocates
`3*nDoses + nDoses*nDoses + nDoses` summaries: for six doses this is 60
components (`TrialDesign.cpp`, lines 85–90). `Kernel::operator()` defines their
order and indicators (`DFKernel.h`, lines 328–356): six reference-superiority
values (the first is fixed at 0.5, the rest compare each dose strictly against
dose one), six efficacy indicators using `>= U`, six future-study indicators
using `> FU`, 36 strict ordered pairwise response comparisons in dose-major,
then comparator-major order, and six squared response probabilities. The
analytic toxicity exceedance vector is separate. `FullIntegrand.cpp`, lines
38–66, places evidence in component zero and indicator integrals in components
1–60; `Kernel::Integrate`, lines 225–261, divides each numerator by evidence
and pushes the resulting 60-vector after a successful integration.

The accumulator implementation is present in the cached BiostatGeneral
dependency at
`research/raw/P12Xuelin/extracted/Map_L_drive_here/BiostatGeneral/v8.1/StatUtilities/`.
`VectorValuedRunningVariance.cpp`, lines 13–31, applies a scalar running
variance to each component. `RunningVariance.cpp`, lines 12–50, uses Welford's
update for centered sums of squares, returns arithmetic mean `sum/n`, and
sample variance `M2/(n-1)` (zero for fewer than two values). `DFKernel.h`,
lines 146–153, clears this accumulator per scenario. `TrialDesign.cpp`, lines
591–597 and 642–648, shows that analysis calls originate in both winner
selection and stopping-rule evaluation; the summary population is successful
analysis calls, not one final fit per trial. `Kernel::PrintResults`, lines
80–92, displays the 60 means and variances as ten unlabelled rows of six.

Python `fit_phase12_importance` already exposes the per-fit values for these
components, their paired-ratio MC standard errors, and the six posterior
response-probability second moments. The new
`summarize_phase12_importance_fits` callable streams a caller-selected iterable
of those fits and returns componentwise means and sample variances in native
order, along with the analysis-call count and number of nonconverged fits.
Nonconverged estimates are included rather than silently filtered, matching
the native cap-return behavior. It does not claim an independence or MC-error
interpretation, alter the calendar to retain all analysis calls, or reproduce
the native integration/random-number implementation. The mean uses streaming
Welford updates rather than native `sum/n`, so last-bit differences are
possible. See [`parallel-phase12-probability-summary.md`](../docs/parallel-phase12-probability-summary.md)
for the explicit scope and example.

The six-dose DF3+3 progression comparator is implemented in
`parallel_phase12_progression.py` and checked against the extracted decision,
opening, and randomization methods in `phase12-progression-reference.json`.
Its transition report averages all source fair-coin branches; it does not claim
native random-stream parity. The calendar OC also now reports the source
DF3Plus3 phase-I tally: per-dose patient/toxicity/admissibility means, the
source early-toxic-stop count, and Monte Carlo errors. These tallies include
only trials that reach the native `TallySim` call, use all trials as the
denominator, and count generated phase-I toxicity events whether observed or
pending. They crosswalk to the native `#Patients`, `#Tox`, `#Pat`, and
`%Admissible` output fields; the native `#Tox` label refers to a count. The
reported Monte Carlo errors are Python OC additions. The existing
final-trial admissibility summary remains separate and includes paths that
were interrupted before the native tally point.

Unresolved work includes exact native RNG and adaptive integration parity, plus
the C++ posterior-parameter accumulator's stale-value behavior on no-fit trial
paths. The Python probability summary operates on explicitly supplied
importance-backend fits and is not relabeled as the native posterior-mode or
MCMC output.
