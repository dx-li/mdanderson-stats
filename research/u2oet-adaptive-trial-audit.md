# U2OET adaptive trial integration audit

## Source rule and computational interface

The cached author guide is `research/raw/U2OET/guide.txt`, section 2.2. It
defines target MCSE/SD monitoring for the four corner utilities in each chain,
with simulation default 0.03 and trial-conduct default 0.005 (both within
0.001–0.05). Retained draws per chain are at least the burn-in count and may be
increased until the target is met. The separate standalone precision module
implements this monitor and explicitly documents its Python batch-means and
growth conventions.

The calendar simulation takes an optional `U2OETAdaptiveSettings`. No settings
preserves the old fixed-budget path. Adaptive settings explicitly supply the
target, initial retained draws, draw cap, append size and whole-trial workload
cap. A cap miss raises before any decision can use the under-precise posterior.
No under-precision override is provided.

## Cache, continuation and random streams

Every changed pair of complete/toxicity-only sufficient statistics invokes
the standalone adaptive fitter. Unchanged statistics reuse the previous
posterior and its precision diagnostics, exactly like fixed-mode posterior
caching. The standalone fitter continues each chain from its last parameter
vector, including association, and only burns in on the first chunk. It returns
its concatenated fit; the calendar immediately converts that to the existing
posterior summary and releases the full draws. Per-decision and final records
retain only success status, draws per chain and the chain-by-corner MCSE/SD
matrix.

The simulation keeps its recorded split data/posterior seeds. Adaptive
sampling consumes only the posterior stream; data-generation and assignment
streams remain separate. With adaptive settings, `design_json` records those
settings so operating-characteristic aggregation cannot combine different
precision rules. With no adaptive settings, design JSON retains its previous
field set and fixed sampler calls/RNG path are unchanged.

The calendar preflight uses the same pure chunk/work/live-cell plan as the
standalone fitter. It conservatively assumes up to `max_patients` distinct
posterior fits, including a distinct final fit, multiplies worst-case per-fit
sampler-work estimate by that bound, and rejects over-budget requests before
splitting the caller's RNG. The standalone per-fit retained-joint and live
array caps apply unchanged. These work estimates are guardrails, not guarantees
of elapsed time or total process RSS.

## Scope

The simulation route uses the existing PDS, CMI or PDS+CMI fit selected by
`model`. GAO remains separate. Corner utility precision is not a convergence
certificate and says nothing directly about the precision of every model
parameter, other utility cells, or efficacy/toxicity risk probabilities.


## Focused validation

Fourteen focused tests pass with warnings treated as errors: the real adaptive
calendar integration and cap failures, standalone continuation/precision tests,
fixed calendar behavior and GAO constructor compatibility. The successful
calendar example uses a CMI fit with one materially free coordinate, two chains,
512 starting draws and a 4096-draw cap. It exercises pending-data caching and a
distinct complete-data final fit. A low whole-trial work cap and adaptive warmup
above 10,000 are rejected before RNG consumption. An unused fixed-mode `draws`
value does not override adaptive storage limits. Targeted Ruff, formatting and
mypy pass. The measured test run peaked at 131.97 MiB RSS with zero swaps;
numerical processes ran serially. This is integration evidence, not a full
published operating-characteristic reproduction.

Root executed the public calendar example and replayed its two fits directly
through the standalone adaptive API using the recorded posterior seed. All
posterior summaries and precision diagnostics matched exactly for the three
cached pending-data decisions and the distinct final analysis. The pending
fit met the target at 512 draws per chain (maximum ratio .047138); the final
fit extended to 4096 draws (maximum ratio .046497). The run took 9.292 seconds,
peaked at 123.55 MiB RSS and reported zero swaps. This verifies the calendar
integration and stream/cache behavior against the separately validated fitter.
