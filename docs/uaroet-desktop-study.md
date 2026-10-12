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

Continuous posterior precision is available through `fit_uaroet_desktop(study,
 rng=..., conduct=False)`. Simulation uses TargetMCSERatio (default0.03);
conduct uses TargetMCSERatio_Treat (default0.005). An explicit target overrides
these settings. The undocumented TargetMCSERatio_Sim alias requires an explicit
target until its precedence is recovered. The function honors1–20 chains and
burn-in once per chain, retaining at least the burn-in length before acceptance.
Each chain continues from its last parameters and association, with a separate
random stream. Joint/parameter means and screening summaries accumulate online;
only utility samples are retained across chunks. Lowest, upper-middle and highest
dose utilities must each meet the target in **every chain**. Batch means use
floor(sqrt(draws)) samples per batch; exact native batch partition/RNG equivalence
is not claimed. Between-chain split R-hat is reported separately and is NaN for
one chain. Neither diagnostic establishes convergence or clinical validity.

Always inspect `precision_met`: reaching max_draws can return False and must not
support a precision-qualified recommendation. Work-budget exhaustion raises.
A constant sampled chain is not counted as precise; an analytically constant
utility table is handled separately. The default work budget remains2million;
callers can explicitly raise it up to128million. Retained utilities are bounded
at20million values and each joint-posterior chunk at2million cells. No desktop
controller or calibration workflow is implied by this inference API.

`calibrate_uaroet_prior(study, scenario, repetitions=..., rng=...)` implements the
specified pseudo-sampling workflow:100 generated patients per dose, zero-mean
SD100 pseudo-priors with positive increments and Uniform(-1,1) association,
then an average of pseudo-posterior parameter means. Each replicate must satisfy
per-chain utility precision and the explicit between-chain R-hat gate (default
1.05). Failure raises without returning a partial prior. A callback can save
completed replicate diagnostics. Fewer than1000 repetitions are exploratory under
the guide; the result exposes `meets_guide_replication_minimum`, which is not a
clinical-validation flag. `result.apply(study)` preserves the chosen prior SDs.
`uaroet_calibrated_prior_text(result)` exports the native threshold-major ordering
and stored association mean; the association prior remains uniform.

`recommend_uaroet_desktop` combines saved patients, inference and the current
controller. Its default `safe` profile applies posterior toxicity/futility checks
without requiring observed bad events, never restores a toxic lowest dose,
corrects old-cohort length and zero-width cutoff schedules, and keeps first ties.
The explicit `legacy_1_8` profile retains the recovered event gates, toxic-lowest
fallback, initial-cohort bypass, old-cohort closure bug, saved-history counters,
and distinct hybrid/final tie rules. This profile is for compatibility research.
Both profiles return a computational `hold` when precision is insufficient or
multiple chains fail the explicit R-hat gate. These policies do not establish
clinical validity. Final selection requires assigned patients and chooses the
highest-mean eligible dose; request `final=True` once maximum accrual is reached.

A decision retains its exact immutable input study. `uaroet_assignment_report`
produces versioned JSON with input provenance, disjoint patient counts, decision
and inference diagnostics; `uaroet_assignment_csv` provides the dose summary.
Reports distinguish unevaluated posterior quantities and unavailable diagnostics.
`UAROETDistributionWriter(stream, study)` can be passed as `on_chunk` to inference
or conduct to stream all marginal categories and utilities. Call its
`validate_complete(chains=..., draws_per_chain=...)` after the fit. Output from an
interrupted fit is diagnostic only. CSV chain/draw/dose indices are1-based;
Python API dose indices remain0-based. These are portable Python output schemas,
not native report byte identity. Calendar simulation/aggregation integration and
final workflow acceptance are still pending; the application remains partial.
