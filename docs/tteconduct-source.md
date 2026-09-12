# TTEConduct source and numerical reference

Catalog 63 is based on John Cook's **TTEConduct 2.0 Users Guide**,
November 6, 2006, available from the
[official software site](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/TTEConduct/TTEConductUsersGuide.pdf).
The downloaded PDF has SHA-256
`c5bbe9d078276cd6a9fb8769401f8d138e20108df413969d7d187ebba498cdd4`.
The guide describes the exponential/inverse-gamma model of Thall, Wooten,
and Tannir (2005), *Monitoring Event Times in Early Phase Clinical Trials:
Some Practical Issues*, Clinical Trials 2:467–478.

The historical mean has an inverse-gamma distribution with shape and scale
`alpha_S, beta_S`. The experimental mean's posterior parameters are
`alpha_E + events, beta_E + total_time_on_test`. The historical distribution
is not updated by experimental observations. The required improvement is an
additive nonnegative `delta`, in the same time units as both means. The guide's
early stopping criterion is strictly
`P(mu_E > mu_S + delta | data) < cutoff`. Reaching the maximum patient count
also stops accrual. Time on test includes censored follow-up and ends at an
observed event for patients who experience the event.

Section 5 uses historical prior `(60, 295)`, experimental prior `(3, 10)`,
`delta=1`, `cutoff=0.03`, and maximum 40 patients, with time inputs in months.
The first six minimum time-on-test values displayed by the desktop program
are 0, 0, 105, 216, 330, and 449 days.

`tools/reference_tteconduct.R` independently integrates over
`Z = beta_S / mu_S ~ Gamma(alpha_S, rate=1)`:

```text
P(mu_E > mu_S + delta)
  = integral gamma_density(z; alpha_S)
      * gamma_CDF((beta_E + T)/(beta_S/z + delta); alpha_E + events) dz.
```

Base R `integrate` and `uniroot` produce continuous boundaries of
0, 0, 3.43772033817485, 7.06782436029872, 10.8382394395421, and
14.7214695552867 months. R also records independent posterior probabilities
at 1 and 20 months of total exposure. These values are stored in
`tests/fixtures/tteconduct-reference.csv`.

Taking the ceiling after multiplying the continuous roots by `365.25/12`
reproduces all six published day values. This conversion is an inference from
the example; the guide does not specify its exact day conversion algorithm.
The Python API returns continuous boundaries in the caller's chosen time
units and does not silently convert or round them. The guide's desktop
ten-years-per-patient table cap is replaced by an explicit search cap in those
same units.

The separate One Arm Time to Event Simulator (catalog 98, version 3.0.9)
advertises additional calendar simulation and superiority rules. Those
features are not implied by this conduct-table implementation. Its Windows
installer contains embedded help whose full scheduling semantics still need
verification.

Validation on September 12, 2026 ran eight focused tests, including the
independent R fixture, time-unit scaling by `1e-100` and `1e100`, and a search
cap/prior-scale ratio that would overflow float64. The combined run took
2.73 seconds, peaked at 132.8 MiB RSS, and reported zero process swaps with
one BLAS thread. Affected Ruff and mypy checks, source/wheel builds, and an
isolated wheel import with the documented monitoring workflow also passed.
The full repository test suite was not run for this checkpoint.
