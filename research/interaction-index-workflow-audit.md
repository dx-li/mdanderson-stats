# Interaction-index workflow coverage

October 10, 2026: catalog entry 65 completes the recovered functional workflow.
The official [CIInteractionIndex_V2.0.1source.zip](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CIInteractionIndex/CIInteractionIndex_V2.0.1source.zip)
was retrieved and inspected; SHA-256 is
`e69f045d6c15595cc67e5055c94cc8ad119230e07e334a2def11708a0f1712f1`.
Original source/documents remain ignored research inputs and are not shipped.
The original archive contains the three inference routines, median-effect
plots, case studies and two simulation scripts.

| Source routine or example | Community implementation |
| --- | --- |
| `CI.known.effect`, Section 2 | Observed-combination log-delta inference with explicit response variance or native-verified residual-df pooled fallback |
| `CI.delta`, Section 3 | Correct fixed-ray coefficient-gradient log-delta intervals; the native mixture-variance defect is recorded below |
| `CI.simulation`, Section 3 | Normal-coefficient RMS comparator with retained coefficient tapes; source-study reporting floors the lower limit at .0001 |
| Median-effect plots and case studies | Optional figures and executable Table 2/3 examples with saved numerical records |
| `Simulation1_3_drugsV2.SSC` | Bounded three-drug generation and all eight summaries; exact source scenario constants, optional retained index/log-index QQ panels and saved JSON |
| `Simulation2_fixed_ratioV2.SSC` | Recovered ratio 2, dose/model/SD/grid defaults, all three interval comparisons, observed/MC length ratios, figures and saved replayable inputs/results |

## Original-function validation

`tools/reference_interaction_index_native.py` verifies the archive checksum and
runs the author's unchanged `CI_IIV2.SSC` function bodies under base R. Its
recording wrapper observes base-R normal draws; a separate calculation
reconstructs the actual coefficient tapes. Neither harness imports production
Python. Nine committed CSV tables use two synthetic designs, including unequal
5/6/4 curve sample sizes and ratios 2/.7. They retain native fitted indices,
log-delta limits, observed-combination limits, MC RMS/limits and true Scenario 2
indices. Python matches the original pooled/MC outputs within absolute and
relative tolerance `1e-12` with the same actual coefficient tapes.

The native residual pool is exactly total single-agent residual sum of squares
divided by total residual degrees of freedom. This independently verifies the
Python pooling choice, rather than a reference intentionally sharing an
unverified convention.

Original `CI.delta` has a defect: its final variance term is proportional to
`(1/D1 + ratio/D2)^2`, but needs inverse mixture dose squared as well. An
independent base-R covariance quadratic form computes the corrected intervals;
Python matches those within `1e-12`. The unchanged native limits remain in the
fixtures and tests require their substantive difference. Reproducing the native
defect would change the scientific result and is not the completion criterion.

## Published and simulation validation

The case-study data are Section 4.2 Tables 2/3 and Figures 4/5 of
[Lee and Kong (2009)](https://doi.org/10.1198/sbr.2009.0001). Base-R references
validate the fitted curves and propagated uncertainty. Published coefficients,
median doses and residual SDs agree within .0005; printed Table 2 intervals
within .001. These rounded published values do not support exact equality.

An independent base-R study reference checks all eight Scenario 1 summaries
across three five-replicate cells, including a negative raw-delta lower limit.
The archive resolves the seventh index as `1/.6`; it sets a QQ title to `1.67`
only after simulation. Existing paper-based defaults remain unchanged, with
`INTERACTION_INDEX_SOURCE_SCENARIOS` providing exact recovered inputs. Original
QQ panels use normal `ppoints`; retained bounded samples now provide those
panels. Python includes endpoints in coverage, while the original continuous
simulation uses strict inclusion; random generation is continuous and exact
endpoint equality has probability zero. This convention remains explicit.

Scenario 2 defaults run all 14 datasets and validate its 43 true indices against
original `Solve.II`. Focused checks cover numerical replay, saved records,
read-only arrays, plotting and preflight controls. All 73 affected interaction
and BayesFactorTTE checks pass together with warnings treated as errors.

PCG64 streams, corrected native defects, native script filenames, page artwork
and file bytes remain compatibility differences. Every recovered calculation
and diagnostic has a usable Python workflow. SYNERGY #18 remains partial for
its separate semiparametric bootstrap contract.
