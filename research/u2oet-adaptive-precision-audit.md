# U2OET adaptive precision source audit

## Primary source contract

The cached author guide is `research/raw/U2OET/guide.txt`, section 2.2 (the
U2OET trial software guide). It states that the program estimates batch-means
MCSE for four corner utilities, separately within each MCMC chain, and uses
the ratio of MCSE to posterior standard deviation. Its simulation target is
0.001–0.05 with a displayed default of 0.03; the trial-conduct target has a
displayed default of 0.005. It says retained draws per chain are at least the
burn-in count and may be increased until each chain meets the target. It also
reports PSRF for these corner utilities when multiple chains are used; PSRF is
a diagnostic, not the precision stopping statistic.

The guide does not give the batch length/overlap rule, the extension schedule,
a finite extension cap, zero-variance handling, or a restart-state protocol.
The implementation therefore labels these as Python policy: nonoverlapping
batch means with `floor(sqrt(draws))` batch length (minimum 2), incomplete
trailing batch omitted from MCSE only (not from SD or fit), fixed-size append chunks, explicit maximum retained
draws, and a nonpassing undefined ratio for zero posterior SD. The required
retained starting amount is at least the caller's burn-in count. The sampler
requires at least eight retained draws per call; a final chunk may absorb a
remainder under eight. If fewer than eight draws remain beneath the cap, the
cap is an upper bound and may not be reached exactly.

## Continuation and method scope

`fit_u2oet` state consists of the complete parameter vector per chain,
including association. Its updates are fixed transition kernels: two outcome
elliptical slice blocks, optional coordinate/link moves, and an association
proposal. There is no adaptive proposal tuning or other hidden state. The
adaptive driver passes the last retained vector of each chain to the next
call, runs warmup only in the first call, and preserves the same Generator
across chunks. It concatenates all retained posterior draws, so reported
diagnostics describe the complete adaptive run. Because random-number use at
chunk boundaries differs, same-seed equality with a single fixed-budget fit is
not claimed.

This adapter covers `fit_u2oet`'s PDS, CMI and PDS+CMI models, with its existing
2–16 chain limits. It does not cover GAO, which has a distinct implementation.
The guide's broader 1–20 chain control is therefore not claimed as implemented.
The returned split-Rhat field reuses the project's classical split-Rhat
summary and is separate from the precision stopping statistic.

## Resource policy

Before the first sampler call (and before consuming RNG state), the driver
validates data/model shapes and bounds a conservative worst-case count of
likelihood-cell evaluations using the sampler's 1000-proposal elliptical-slice
iteration limit, optional coordinate updates, all chains, warmup and the full
draw schedule. It caps the retained joint posterior at four million cells and
uses a combined live-cell estimate for joint, parameter, likelihood, utility,
concatenation/freeze copies, normalized diagnostic traces, batch means, and
split-Rhat workspace. Chunking limits per-call
sampler arrays, while the retained posterior and utility summaries remain
available in the result. These are workload estimates, not RSS or runtime
guarantees.

## Numerical validation

Four focused tests pass with warnings treated as errors. They check continued
chains against explicit fixed-sampler calls, short final chunks, constant
utilities, rejection before RNG consumption and batch-means diagnostics.
Targeted Ruff, formatting and mypy pass. The measured focused run took 1.93
seconds, peaked at 127.00 MiB RSS and reported zero swaps.

`tools/reference_u2oet_precision.R` generates independent base-R references
for two-chain four-corner traces of lengths 16, 37 and 64, including equal
and unequal constant chains. `tools/check_u2oet_precision.py` compares 252
SD, MCSE, ratio and classical split-Rhat diagnostics at utility scales
`1e-200`, `1` and `1e200`. The maximum absolute difference after undoing
the scale is `2.89e-15`. The comparison took .008 seconds, peaked at 123.27
MiB RSS and reported zero swaps. These validate the documented Python
diagnostics, not an unspecified native batching or continuation algorithm.

An end-to-end CMI fit used two chains, 100 warmup draws, 512 starting retained
draws and 512-draw extensions, with an upper cap of 4096 and a .05 precision
target. It used the existing logistic-normal reference configuration: three
doses per agent, ten observations at the central combination (three efficacy
responses, no toxicities), near-fixed slope and toxicity parameters, and a
standard-normal efficacy intercept. With seed 7771 the controller continued
to 1024 draws per chain and met every corner target. The largest MCSE/SD ratio
was .047268 and the largest corner split-Rhat was 1.000069. The efficacy
intercept and response-probability means differed from the existing independent
R quadrature references by 2.363 and 2.145 estimated MCSEs. The run took 1.026
seconds, peaked at 121.34 MiB RSS and reported zero swaps. This is a bounded
continuation and posterior-accuracy check, not validation of trial operating
characteristics or all model parameters' precision.

Numerical processes ran serially with numerical-library thread counts fixed
at one. Adaptive GAO fitting remains separate work. Subsequent
[calendar integration](u2oet-adaptive-trial-audit.md) connects this controller
to PDS/CMI/hybrid interim and final analyses.
