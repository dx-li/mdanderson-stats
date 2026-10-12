# Recovered mTPI R final selection

`select_mtpi_legacy_mtd` implements the final-estimation mathematics in
`mTPI-middle.R`, distributed in MD Anderson's official
[TPI version 2.1 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/TPI/TPI_V2.1PID439.zip).
The archive is mTPI; this is not the original 2007 TPI rule or mTPI-2.

```python
import numpy as np
from mdanderson_stats import select_mtpi_legacy_mtd

result = select_mtpi_legacy_mtd([3, 9, 6, 0], [1, 1, 3, 0], target=0.30)
assert result.dose == 2  # one-based; None means no selection
np.testing.assert_allclose(result.isotonic_mean[:2], [0.145219988941394] * 2)
assert np.isnan(result.isotonic_mean[3])  # beyond highest treated dose

# Supply recorded controller state; counts alone do not determine it.
stopped = select_mtpi_legacy_mtd([3, 3], [3, 3], no_selection=True)
assert stopped.dose is None
```

Counts are complete-outcome patient/DLT vectors in increasing dose order,
with at most 100 doses and 200 patients. `target` is a probability in (0,1).
No physical dose units or dose spacings enter this estimator.

The posterior for final estimation is Beta(y+.005, n-y+.005), distinct from
allocation's uniform prior. An inverse-posterior-variance weighted isotonic fit
uses every dose from 1 through the highest treated dose, truncated just below
`first_excluded_dose` if supplied. Even untried lower doses contribute their
prior. Candidates start at the lowest treated dose. The returned `isotonic_mean`
contains the fit; `selection_score` additionally adds dose index times 1e-10
at dose 2 and above. The candidate nearest the target is selected; exact residual
distance ties choose the lowest dose. `eligible` records the candidate interval.
All arrays are read-only and retain the full input shape; unused fits are NaN.

`first_excluded_dose` corresponds to the original controller's `toxdose`, with
one-based values 1 through dose count + 1. `None` defaults to no exclusion.
`no_selection=True` corresponds to its `nodose`, which may reflect excessive
toxicity **or underdosing**. This estimator does not infer either flag or apply
an additional posterior-overdose screen. Trial conduct requires an independently
validated safety policy. The existing `MTPIDesign.select_mtd` and `simulate_mtpi`
retain their documented paper-based safety and selection rules.

The R loop `2:tdose` runs backward when `tdose=1`; Python corrects that indexing
defect by perturbing only existing doses. Empty candidate ranges return `None`
instead of evaluating reversed R index ranges. No-trial and stopped-trial outputs
contain NaN fits and no eligible doses. Noncontiguous treatment counts are accepted
as estimator inputs, but are not a claim about reachable native trial paths.

Ten synthetic states are checked against unchanged original R final-estimation
statements (3e-14 relative tolerance), including unequal sample sizes, pooling,
untried lower doses, exclusions, offsets, and extreme outcomes. An independent
exact-rational max-min characterization checks weighted isotonic fits across all
60 response configurations with sample sizes (2,3,4). See the
[audit and reproducible reference command](../research/mtpi-legacy-selection-audit.md).
These validate the estimator, not the entire native controller, Excel add-in,
random streams, or clinical validity. Current catalog72 workflow acceptance is documented in mtpi-desktop.md.


Exact residual ties are independently checked by original R fixture case11:
`n=(2,2)`, `x=(1,1)`, target `.5000000001`. After native offsets the scores
`.5` and `.5000000002` have exactly equal residuals; original R and Python both
select dose1. Hash-pinned original source line220 selects
`sort(abs(pp[mindose:tdose]-pT), index.return=T)$ix[1]`, not all minimizers.

The [current desktop workflow](mtpi-desktop.md) now adds the separately named
XLL estimator, `select_mtpi_desktop_legacy_mtd`, native controller compatibility,
all-scenario simulation and reopenable reports. This R estimator keeps its
existing offsets/ties; it is not silently redefined to match the XLL.
