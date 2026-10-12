# UAROET 1.8 saved inputs and observation accounting

`read_uaroet_desktop_bundle(files)` imports an explicit mapping from original
filenames to their text contents. It validates the documented saturated ordinal
model, prior means/SDs, selected utility, trial settings, scenarios and optional
patient records. It does not read arbitrary paths, execute the original program,
or infer missing scientific settings. The current desktop workflow remains
**partial**; this interface is a saved-input and counting contract.

```python
from pathlib import Path
from mdanderson_stats import read_uaroet_desktop_bundle, read_uaroet_desktop_study

folder = Path("my-uaroet-study")
files = {p.name: p.read_text() for p in folder.glob("UAROET_*.txt")}
study = read_uaroet_desktop_bundle(files, patient_text="1 1 1 0\n2 2 -1 1")
saved = study.to_json()
reopened = read_uaroet_desktop_study(saved)
assert reopened.to_json() == saved
counts = reopened.counts()
```

Original patient records contain ID, one-based dose, efficacy category and toxicity
category. Outcome categories are zero-based; `-1` means pending. Python study
records store zero-based doses. IDs increase strictly and start at one. Counts
separate complete joint outcomes from additional toxicity-only outcomes. Efficacy-only
and wholly pending patients contribute to assigned counts but not the likelihood.
`outcome_counts()` returns `(joint, toxicity_only, assigned)` for connection to
`fit_uaroet`; the existing fitter retains its documented positive truncated-normal
increment priors and independent Python sampling algorithm.

`counts()` also exposes complete-case efficacy/toxicity margins and the distinct
native likelihood count. `legacy_highest_allowed` is the native escalation ceiling:
one dose above the highest assigned dose becomes available only after at least one
patient there contributes toxicity information. This is bookkeeping, not an
assignment recommendation; no posterior safety filtering is performed. Native
observed-outcome gates count complete cases, even though toxicity-only observations
inform the likelihood. These counts are independently compared with 184 executed
original-instruction cases. Empty-data native bookkeeping returns dose zero; the
configured starting dose still belongs to the controller.

Original compressed prior files use threshold-major coordinates. The importer
maps these into Python's endpoint baselines followed by dose increments. It
retains the final association mean as metadata; the association prior remains
uniform on `(-1,1)` in the documented model. A per-coordinate SD file takes
precedence over the scalar default. Its association SD does not set a prior.
Coordinate mapping participates in 720 native logit/marginal comparisons. It does
not by itself establish identity with the original prior sampler.

Two repairs require explicit choices:

- Both bundled examples use `UseHybridGreedyArandAlgorithm`, which the recovered
  native parser does not recognize. Use `repair_example_typo=True` to map it to
  the documented `UseHybridGreedyArandAssignment` keyword.
- The three-dose bundle selects utility 2 but supplies only utility 1. Select
  `utility_identifier=1` explicitly if that is the intended design.

Both changes are recorded in `input_notes` and survive saving/reopening. The
recognized `UseHybridGreedyARandAssignment` capitalization variant needs no repair.
Unknown scientific setting keys, duplicate keys/cells, invalid probabilities,
nonmonotone utilities and malformed histories fail validation.

Supported bounds are 2–5 doses, 2–4 categories per endpoint, 100 scenarios and
500 patient records. The Python schema permits 1–500 planned patients as a bounded
extension; the guide's design menu specifies 12–500. Input probability rows may
differ from one by at most `1e-12`; transient normalization is used only when
computing Gaussian-copula scenario probabilities, preserving exact JSON round trips.

Simulation target MCSE settings are stored and validated, **not satisfied by
importing a file or running a fixed short chain**. Remaining current-version scope
includes prior calibration/ESS, precision-controlled inference, safe and explicit
legacy controllers, as-of conduct, calendars, simulation aggregation and reports.
See [recovery evidence](../research/uaroet-current-recovery-2026-10-12.md).

## Prior effective sample size

`uaroet_desktop_prior_ess(study, draws=10000, rng=rng)` draws the documented
independent normal/truncated-normal prior coordinates and matches each ordinal
marginal's mean and sample variance to beta shapes. It reports per-category
information plus efficacy, toxicity and combined means, minimum and maximum.
The calculation ignores patient observations. Association does not change these
marginals, so no copula integration is needed. This is prior-information ESS,
not MCMC sampling efficiency or prior calibration from elicited scenarios.

`uaroet_marginal_ess(efficacy, toxicity)` applies the same reduction to supplied
sample-by-dose-by-category probability draws. Zero-variance marginals have no
finite beta moment match and are rejected. Negative finite-sample estimates are
reported unchanged; they are not valid positive beta-prior information measures.
Thirty-six original reduction cases and an independent logistic-normal integral
check support these calculations.

Both native-recognized distribution keywords, `DistributionSave` and
`DistributionSaveOption`, now canonicalize to the latter. Supplying both is an
error; no preference is silently chosen. Capturing an export setting still does
not imply that the complete desktop export workflow is implemented.
