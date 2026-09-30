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
