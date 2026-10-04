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

For kernel probability vectors, `Kernel::Integrate` pushes the per-fit vector
to `m_vvrtTot` (`DFKernel.cpp`, line 261), and `Kernel::PrintResults` prints
each component's running mean and variance (`DFKernel.cpp`, lines 78–89).
The implementation of `VectorValuedRunningVariance` is absent from the cached
archive, and the current Python fit result does not expose the same integrated
vector. Consequently neither its variance convention nor a faithful probability
vector crosswalk can be asserted. Importance-backend and MCMC summaries are
not relabeled as this native Laplace/integral output.

The Python calendar OC already reports enrollment, selection, stopping,
toxicity/response and model-fit diagnostics. The source-defined, tractable
duration summary is added there without retaining all trial records. C++
posterior-kernel printouts, the native DF3+3 comparator, and exact native RNG
parity remain separate limitations.
