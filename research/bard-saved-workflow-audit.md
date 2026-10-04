# BARD saved workflow checkpoint

The captured study and report APIs complete the guide's save/reopen and
portable BF-BOIN protocol/OC workflow. The saved specification reconstructs
the existing calibrated response model and calls the existing streaming
simulator. The report presents the actual captured settings and summaries;
it introduces no new posterior, dose-selection or calendar algorithm.
See the [input-schema audit](bard-study-audit.md),
[report audit](bard-report-audit.md) and
[remaining-scope crosswalk](bard-remaining-simulation-audit.md).

Five focused checks pass: two cover immutable inputs, JSON/file roundtrip,
seeded replay and input/work rejection; three cover scenario-order
independence, escaped complete reports and aggregate rejection before a run.
Scoped mypy and final Ruff/format checks pass. No CI configuration was changed
and no full local test suite was run.

An additional integration check round-tripped nondefault priors, eligibility,
profile weights, balanced-factor subset, endpoint joint probabilities,
timing, titration, expansion and the maximum unsigned 64-bit seed. The
reloaded study and report reproduce every compact summary field. A separate
zero/one-response case confirms that files contain finite calibration inputs
rather than infinite fitted intercepts and still replay identically. These
small serial runs used 133.06 MiB peak process RSS and zero reported swaps.
This checks transport and composition; the mathematical references for the
unchanged kernels remain in their existing audits.

The cached app guide advertises BF-BOIN for stage one. Its simulation, saved
input and report workflows are now available with explicit Python policies.
The paper additionally evaluates stochastic BARD-BLRM trials. Existing
BF-BLRM replay and supplied-outcome continuation do not perform that repeated
simulation. Its stage-one cap calibration and stage-two duration convention
also require explicit choices. Catalog 165 remains partial under the broader
method-coverage objective; native hidden defaults and exact document formats
are not claimed.
