# Multc Lean recovered compact boundaries and study aggregation

On October 9, 2026, continued inspection of the official Multc Lean 2.1 DLL
resolved posterior-to-vector control, prior/minimum/cohort precedence and
native study aggregation. The source remains the archive and DLL identified
in [the duration audit](multc-native-duration-audit.md). No original program,
machine code, default patient data or inspection runtime is distributed.

## Original instruction references

`tools/reference_multc_legacy_boundaries.py` verifies DLL SHA-256
`8ac45568e87530d4bb6732eb8807a22e4e8c5de52a5557a14f750c24867fa807`
and runs bounded x86-32 Unicorn execution. Each boundary/summary run has
100,000 instructions and one second; complementation has 1,000 instructions
and one second. Failure to return normally is an error.

* RVA `0x7480`: original compact-boundary loop and vector writes. Only the
  posterior predicate at `0x6340` and vector allocation/access services are
  substituted. The predicate reads the beta shapes actually written by native
  update instructions and looks up an independently generated base-R tail.
  Its decision is the strict comparison `probability > cutoff`.
* RVA `0x59b0`: original toxicity complement instructions. Only default-struct
  initialization is substituted. Fixed history becomes `1-history`; random
  historical and experimental beta shapes swap; the margin changes sign and
  the cutoff is retained. Unused struct fields are not model parameters.
* RVA `0x75b0`: original study prior screen, aggregation, averaging and PMF
  normalization. Scenario validation, vector services, RNG seeding, unrelated
  security-cookie verification, boundary calls and individual trial calls are
  substituted. Those trial outputs come from the independently executed
  original duration kernel with saved explicit variates, not production Python.

These are **control/aggregation references**. They do not run the original
posterior integrator, Windows GUI, CLR setup, study file conversion or random
generator. The production posterior comparisons are separately validated
against R and the published tutorial; those claims must remain separate.

`tools/reference_multc_legacy_boundaries.R` uses base-R `pbeta` for fixed
history and direct integration over historical beta density for random
history, split at shifted-support boundaries. It imports no production Python.
The saved table contains 7,496 posterior probabilities. Each queried production
probability agrees within the comparison's documented 1e-8 integration scale,
and all stopping comparisons agree, including exact uniform-prior cutoff ties.

Twenty-two synthetic designs cover fixed and beta histories, fractional prior
shapes, shifted margins, cohort sizes 1/2/3/6, minimum below a cohort, minimum
cohort multiples, minimum at the cap, disabled endpoints, support endpoints,
prior response/toxicity/both rejection and strict prior equality. Original
compact vectors match exactly. Fifty-four original duration replays use those
vectors: counts and draw consumption match exactly; duration tolerance is
2e-15 relative / 1e-14 absolute. Twenty-two study wrapper runs verify all five
means and the PMF, including four prior-screen exits without trial/RNG calls.
The mean tolerance is 3e-15 relative / 1e-14 absolute.

## Recovered contracts

The prior is checked before minimum enrollment. Rejection returns one vector
entry zero. Otherwise, the native loop advances one success index through
scheduled sample sizes, never revisiting earlier indices and excluding the
all-success count. The loop adds cohort size, then clamps to minimum enrollment.
The native parameter validator requires positive controls, minimum no larger
than cap, cap divisible by cohort, and minimum either below one cohort or a
cohort multiple. The Python design already enforces that schedule.

An endpoint without any crossing gets vector `(cap,)`. A nonempty crossing
vector whose last entry precedes the cap gets one cap placeholder appended.
The placeholder is not evidence of an adverse posterior. The duration kernel
exits at the cap before actual boundary checks. Toxicity vector indices count
**nontoxicities**. General monitoring uses ordinary toxicity counts and full
posterior boundaries; those interfaces are distinct.

The original wrapper averages duration, enrollment, response, toxicity and
balk counts and normalizes its sample-size histogram. Prior rejection zeros
the means and leaves the PMF unpopulated. Python explicitly provides a point
mass at zero for that case. Its additional MCSEs use sample SD/sqrt(replicates)
and are NaN for one replicate. Bounded PCG64 simulation retains child seeds,
owned numeric summaries and reconstructs patient ledgers on demand. It uses
separate outcome/exponential streams and does not claim original RNG parity.

## Remaining scope

The subsequent [managed-model/workflow audit](multc-lean-model-audit.md) resolves
native saved files/defaults and the composed protocol/report workflow, completing
entry 12. Native posterior-integrator, CLR/GUI and Windows RNG identity remain
compatibility differences rather than missing advertised Python calculations.
The existing saved Python study's calendar simulation keeps its explicit
observation-aware policy; the new legacy batch uses recovered latent-count
and balked-arrival behavior. No silent change of that study engine is made.

Reproduce the numerical table with base R, then use research-only
`pefile==2024.8.26` and `unicorn==2.1.4` for the instruction references:

```bash
Rscript tools/reference_multc_legacy_boundaries.R
uv run --locked --extra plot python tools/reference_multc_legacy_boundaries.py \
  --dll research/raw/source-recovery-2026-10-08/multc-lean-native/MultcLeanCalcs.dll \
  --check
```
