# GAO adaptive corner-precision audit

## Source contract and scope

The cached U2OET author guide's adaptive-precision section is summarized with
its exact source location and provenance in
[`u2oet-adaptive-precision-audit.md`](u2oet-adaptive-precision-audit.md). It
defines four dose-grid corner utility expectations, monitored separately by
chain using batch-means MCSE divided by posterior standard deviation. It gives
a target range of 0.001–0.05 and requires retained draws per chain to be at
least the burn-in count. It also describes PSRF as a separate diagnostic.

The guide does not determine batch length, extension schedule, finite cap,
zero-variance policy, or continuation-state handling. The new GAO wrapper
therefore reuses the documented criterion but labels these details as Python
policy. It wraps the explicit-prior 2017 GAO fitter, not the native executable,
and makes no claim about native prior-file defaults or exact native output.

## Continuation and resource policy

The Python GAO sampler retains the complete parameter vector per chain,
including the association Fisher-z coordinate. Its transition is elliptical
slice sampling over fixed independent normal-prior coordinates; it has no
adaptation or latent auxiliary state. The wrapper runs warmup once and starts
each subsequent chunk from that chain's preceding final vector. The same
Generator advances sequentially across calls.

Batch means are nonoverlapping with length `max(2, floor(sqrt(draws)))`; a
trailing incomplete batch contributes to neither batch-means variance nor
MCSE. Posterior SD uses all retained draws. Chunks are at most the requested
batch size, with a final short remainder absorbed if fewer than eight draws
would otherwise remain. If the cap has fewer than eight remaining draws, that
remainder is not sampled. Undefined ratios from zero posterior SD do not pass.
The wrapper inherits the fitter's 2–16 chain range, narrower than the guide's
1–20 control range.

Before RNG use, a conservative plan checks that the aggregate minimum
likelihood evaluations and dose/category grid work for all chunks fit the
requested budgets, including one pre-RNG start likelihood per chain. It also
checks four-million retained joint cells and twelve-million live cells. The
minimum work charges one accepted proposal per free-coordinate slice update,
including warmup. Remaining evaluation/work budgets are passed into each
chunk, where the sampler can explicitly fail if slice rejection requires more
work than remains. These are deterministic allocation/work estimates rather
than runtime or RSS guarantees. The returned split-Rhat is a
separate descriptive diagnostic, not an additional source-defined stop rule.

## Validation

Five focused tests pass with warnings treated as errors. They compare the wrapper's two-chunk result with
manual calls to the fixed GAO fitter using the same continuation vectors,
recompute all corner diagnostics from retained probabilities, reject an
aggregate-work overrun before RNG state changes, and check that constant
corner utilities cannot pass. They also exercise a four-chain free-coordinate
run with 500 warmup and 500 retained draws per chain, plus explicit exhaustion
during a slice update under a nearly minimum runtime budget. The focused run
took 2.24 seconds.
Ruff format/check and module-scoped mypy with silent imports pass. No native
executable comparison or RSS measurement was performed.
