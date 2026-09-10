# IPDfromKM reconstruction

`reconstruct_ipd` implements the modified iterative Kaplan–Meier reconstruction
used by MD Anderson's [IPDfromKM application](https://biostatistics.mdanderson.org/shinyapps/IPDfromKM).
It accepts **cleaned, digitized coordinates** and creates approximate patient
follow-up times and event indicators. These are synthetic records compatible
with the supplied aggregate information, not recovered original patient data.

```python
import numpy as np
from mdanderson_stats import reconstruct_ipd

ipd = reconstruct_ipd(
    np.arange(11),
    np.linspace(1, 0.5, 11),
    risk_time=[0, 4, 8],
    at_risk=[100, 72, 40],
)
assert len(ipd.time) == 100
assert ipd.event.sum() == 36
assert ipd.remaining == 24
assert abs(ipd.survival[-1] - 0.508055626652268) < 1e-14
```

Supply either `patients` alone or paired `risk_time` and `at_risk` vectors.
Coordinates must start at `(0, 1)`, have nonnegative nondecreasing times, and
nonincreasing survival probabilities in [0, 1]. Duplicate times are allowed for
vertical drops. Risk times increase strictly from zero; counts are nonnegative
integers that cannot increase. Each retained risk interval needs at least one
coordinate. Counts reported after the curve ends constrain final censoring as
in the source. Limits are 100,000 coordinates and 1,000,000 initial patients.

## Censoring and event estimation

Within each risk interval, uniformly spaced latent censor times determine
censor counts between curve coordinates. Rounded event counts update the
product-limit survival estimate. The interval censor count is iterated to match
the next reported risk count where possible. The reconstructed risk count is
carried into the next interval; the returned `reported_risk` and
`reconstructed_risk` expose discrepancies instead of concealing them.

The final interval uses the source's mean of previous interval censor counts,
scaled by its duration relative to the preceding risk-time spacing. This is the
native heuristic, including when earlier interval durations differ. With no risk
table and no total event count, intermediate censoring is assumed absent.
Remaining patients are censored at the final coordinate. Emitted censor times
are coordinate midpoints, as in the R code, rather than the latent uniform times.

`total_events` informs the final censor estimate but **does not enforce the
reconstructed event total**. In the example above, supplying `total_events=50`
still reconstructs 36 events. Inspect `ipd.event.sum()` against any reported total.
This behavior follows the numerical source.

Results include read-only individual `time`/`event` vectors, coordinate-level
`risk`, `events`, `censored`, right-continuous fitted `survival`, retained risk
times/counts, terminal `remaining` patients, and unrounded error summaries.
`remaining` patients are already included as terminal censors in the records;
they are not included in the coordinate-level `censored` counts.
Records are sorted by time, with events before censors at a tied time.

## Numerical choices and differences

- Integer rounding uses nearest-even, matching R. A final censor estimate within
  16 machine epsilons (relative to its magnitude, or 1 near zero) of an integer is
  snapped before truncation. This prevents losing a censor solely from changing
  time units. The untruncated estimate still sets the latent censor spacing,
  matching R's fractional final-interval behavior.
- Exhausted risk sets retain their last survival estimate instead of dividing by
  zero. Impossible event/censor counts and nonconvergent censor iterations raise
  errors. The iteration is bounded at 10,000 distinct censor guesses.
- Censor allocation uses sorted-bin searches rather than a nested patient/point
  scan. Midpoints use differences to avoid overflow from summing large times.
  Unrepresentable midpoint placement raises an error.
- RMSE divides the unrounded squared errors by the number of coordinates. R
  rounds intermediate survival/errors to three decimals and divides by `n-1`.
  Mean and maximum absolute errors are also returned without rounding.
- Duplicate-time coordinates are compared to the final, right-continuous
  survival at that time, so the upper end of a vertical drop can have nonzero
  error even when the reconstructed drop is exact.
- Input cleaning is explicit: malformed/nonmonotone coordinates are rejected.
  The native outlier removal, cumulative-minimum cleaning, interactive image
  digitizing, graphical reports, KS diagnostic and secondary survival analyses
  are not yet ported. Catalog entry 151 remains **partial**.

## Validation and sources

Focused checks compare event, risk, censor and survival estimates against the
numerical body of CRAN IPDfromKM 0.1.10 `getIPD.R`, stopping before its dplyr-based
reporting. Ordinary no-risk-table, multi-interval and total-event-input cases
match; 30 additional exponential-curve cases match the original R counts exactly
and its finite survival estimates to floating-point tolerance. In all 30 cases,
R produces an undefined terminal survival after exhausting the risk set; Python
retains the last product-limit estimate. Tests independently recalculate the
product-limit curve from emitted records, cover complete/no-event curves and
vertical drops, and check equivalent time units at factors `1e-200` and `1e200`.

Reference source: [CRAN package](https://CRAN.R-project.org/package=IPDfromKM),
[versioned R source](https://github.com/cran/IPDfromKM/tree/16ea3e163b8ad409e51e035154c52803dcb1c28b),
and Liu, Zhou and Lee's [method paper](https://doi.org/10.1186/s12874-021-01308-8).
The reference package declares GPL-2; its files are retained only as ignored
research inputs. Source hashes are recorded in `ipdfromkm-sources.json`.
