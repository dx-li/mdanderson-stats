# ToxFinder saved-study completion evidence

The remaining TMML-to-prior gap is closed by `elicit_toxfinder_study_prior`.
The physician doses, reference scales, target/low/high probabilities and
interaction moments are mapped explicitly from saved settings. Missing or
inconsistent inputs fail before fitting. `simulate_toxfinder_study` and
`recommend_toxfinder_study` reuse the recovered native allocation/controller
with independently validated Python inference. A strict patient CSV writer
supports editing and reopening, including physical-dose conversion and quoting.
Reports capture design, prior, Python sampler settings and quality diagnostics.

## Original variance conversion

Original ToxFinder.jar SHA256
`39e72d4b535595f548b6602496227bc92a6d9107fbb3ce5560973342fd39897a`.
`VarianceAsSDValue.setGADAValue` takes the square root of the stored number;
`getGADAValue` squares the displayed number. `ParameterValue` serialization
calls that virtual getter. Thus `pr-alpha3-var`/`pr-beta3-var` in TMML are
variances, despite comments/display labels saying SD. The archived example's
values3 and0.9 must not be squared again.

`tools/reference_toxfinder_variance.py` calls these original Java conversions
while substituting only value storage/GUI validation-event plumbing. Three
fixtures confirm stored/displayed/saved values0.25/0.5/0.25,
0.81/0.9/0.81 and3/sqrt(3)/3 (floating-point roundoff retained).
`parameters_tox.wgc` makes the native `mcmc-trans`, `mcmc-maxiter` and
`mcmc-reeval` settings hidden. Python sampler controls remain explicit and are
recorded in reports; native metadata and seeds are retained without a guessed
mapping or false claim of identical adaptive MCMC behavior.

## Declared profiles and guide workflow

| Advertised workflow | Python contract/evidence |
| --- | --- |
| Physician prior elicitation | Saved four-dose inputs per agent, native stochastic calibrator, independent Gamma-density assessment and explicit probability MCSE/residuals. Source precheck/simplex/retry fixture evidence is retained. |
| Saved study and scenario | TMML import/export, all saved design fields, physical scaling, active FI+CKP/FI modes and explicit seeds/settings. Derived dataclass fields are rebuilt from saved settings. |
| Trial conduct | Original patient CSV import/edit/export; pending-outcome filtering, capacity bound, stage selection and alternating cohort branch. Reports show the actual observed/pending counts and physical recommendation. |
| Simulation and final analysis | Stage-one cohorts and stage-two paired cohorts; posterior fits and native information/contour allocation; complete scenario summaries and uncertainty measures. |
| Reports and reproducibility | Captured study, prior, sampler controls, calibration assessment when supplied, per-trial MCSE/R-hat, saved patient CSV and readable HTML. Prior moments can be saved/reopened without refitting. |

The study runner defaults to **refit after all final outcomes**. Original
pre-final-cohort posterior use requires explicit `final_policy="legacy"` and is
identified in the report. Original mislabeled SD/SEM and sentinel behavior are
not hidden scientific defaults; correctly labeled sample SD/MCSE are retained.
The stage-one invalid predecessor-memory access uses the documented safe boundary
correction. These are intentional, visible differences rather than claims of
complete native executable equivalence.

Original x86 fixture regeneration yields seven identical JSON sets; the information
set has two supplied-LAPACK floating-point differences, maximum absolute
1.46e-10 on an approximately249726 information value (relative below6e-16).
No decisions, control flow or integer outcomes differ. This is not native RNG
identity, and the original binaries remain excluded from distributed artifacts.
Clinical validity and distribution licensing are separate, unchanged matters.

Final full-suite, installed-wheel and artifact evidence is recorded in the
[completion validation manifest](native-workflows-validation-2026-10-12.json). Status promotion is based on the declared
scientific/user workflows, not GUI appearance or native random bitstreams.
