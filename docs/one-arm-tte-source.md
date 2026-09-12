# One Arm Time to Event Simulator source and validation

Catalog 98, version 3.0.9, is documented by the HTML help embedded in the
[official Windows installer](https://biostatistics.mdanderson.org/SoftwareDownload/FileDownloader/Index/470).
[Source hashes](one-arm-tte-sources.json) identify the archive and the individual
help files used. The installer was not executed: its embedded cabinet was
extracted using the system archive reader. Raw vendor files remain outside
the distributed Python package.

The help establishes the following mathematical and calendar behavior:

- Exponential patient event times have either a mean or a median parameter.
  Both independent inverse-gamma priors are on the selected parameter itself.
  With `d` events and total observed exposure `T`, the experimental posterior
  has shape `a+d` and scale `b+T` for mean parameterization, or `b+T*log(2)`
  for median parameterization. The historical prior remains fixed.
- Inferiority and superiority have separate, possibly negative additive
  margins. For a larger-is-better goal the desirable probability is
  `P(theta_E > theta_S + delta)`; for smaller-is-better it is
  `P(theta_E < theta_S + delta)`. Inferiority uses a strict probability
  comparison below its cutoff; superiority uses a strict comparison above
  its cutoff. Cutoffs may include zero and one.
- No stopping check occurs before minimum enrollment. Optional periodic
  checks lie on a grid starting at trial origin zero, which usually precedes
  the first Poisson arrival. Optional accrual checks occur before enrolling
  the next patient.
- Accrual ends on a stopping decision or at maximum enrollment. Periodic
  checks cease at that point. The fixed follow-up period starts at this
  actual accrual stop time, even after an early stop. The rules are assessed
  once at the end, including when follow-up is zero. Final classification
  may differ from the reason for stopping accrual.
- Each scenario generates Poisson arrivals and independent exponential
  event times using its true mean or median. The reported central interval
  uses sample quantiles at `(1-level)/2` and `(1+level)/2`.

The help does not resolve simultaneous contradictory rules, exact event-time
ties, or the sample-quantile interpolation algorithm. The Python interface
preserves both rule flags, processes observed events and monitoring before an
arrival at the same time, and uses linear sample-quantile interpolation.
These are explicit conventions, not claims of native executable parity.
NumPy random streams also differ from the Windows program's streams.

`tools/reference_one_arm_tte.R` independently integrates over the gamma
distribution of the inverse historical parameter. Twelve reference cases
cover both goals, both parameterizations, and margins -1, 0, and 1, with
historical prior `(4,8)`, experimental prior `(2,3)`, two events and exposure
5. For the mean case the posterior equals the historical distribution;
the zero-margin probability is exactly one half. For the median case the
posterior scale is `3+5*log(2)`, and the zero-margin larger-is-better
probability is approximately 0.3852909215.

A deterministic scheduling reference uses unit shape/scale priors,
minimum enrollment 2, maximum 4, arrivals `[0.2,1.2,2.2,3.2]`, event durations
`[0.5,100,100,100]`, periodic interval 1, pre-accrual checks, and follow-up 2.
With inferiority cutoff 0.5 and superiority cutoff 0.8, the first eligible
check is at time 2. It has two patients, one event, and exposure 1.3. The
desirable probability is `(2.3/3.3)**2`, so accrual stops for inferiority.
At final time 4, exposure is 3.3 and probability is `(4.3/5.3)**2`; neither
rule is met. There are no intermediate follow-up checks.

The original methodological paper is Thall, Wooten, and Tannir (2005),
*Monitoring Event Times in Early Phase Clinical Trials: Some Practical
Issues*, Clinical Trials 2:467–478. Its randomized multicomponent extension
is distinct from this single-arm simulator and is not implied by this port.

On September 12, 2026, ten focused tests passed in 1.98 seconds with a peak
RSS of 133.1 MiB and zero process swaps. These include a 400-trial check
against the analytic Poisson-accrual mean calendar duration. A separate
100-trial, maximum-40-patient workflow from the built wheel took 0.283 seconds,
peaked at 112.7 MiB, and reported zero swaps. It used both monitoring modes,
minimum enrollment 5, zero margins, and fixed final follow-up. Nonzero-margin
quadrature is more expensive than this analytic zero-margin example.
Affected Ruff/mypy checks and source/wheel builds also passed. The full
repository test suite was not run for this checkpoint.
