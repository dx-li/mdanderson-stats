# 2010 U2OET GAO conduct reference

The independent source is Houédé et al. (2010), *Utility-Based Optimization of
Combination Therapy Using Ordinal Toxicity and Efficacy in Phase I/II Trials*,
cached as `research/raw/U2OET/gao2010.txt` (DOI 10.1111/j.1541-0420.2009.01302.x).
Section 3.3, equation (11), selects the dose pair maximizing posterior mean
utility over the candidate set. Section 3.4, equation (12), stops only when
the minimum over the entire dose grid of the posterior probability that the
most severe toxicity probability exceeds its limit is greater than `pU`.
Consequently safety is a global stopping rule, not a per-pair exclusion rule.
The same section states that untried escalation is limited to the three
coordinate-wise upper neighbors; there is no analogous restriction on
de-escalation. Previously used pairs remain available for selection.

`tools/reference_u2oet_gao2010_conduct.R` is a base-R reference over explicitly
supplied synthetic posterior joint-probability draws. It does not fit the
model or reproduce an MCMC sampler. The fixture cases cover interim candidate
masking, a utility-maximizing pair whose toxicity-tail probability is high
while another pair is safe, exact equality at the toxicity limit, exact equality
at the global stopping probability, a global all-pairs stop, and final
unrestricted grid selection. Utility and toxicity summaries use direct
expectations under each supplied joint draw; exceedance uses the strict event
`p_severe > toxicity_limit`. Batch MCSE uses per-chain nonoverlapping batches
of size `max(2, floor(sqrt(draws_per_chain)))`, dropping an incomplete tail,
then the sample standard deviation of pooled batch means divided by the square
root of their count.

The interim rule for an untried mixed-direction move (one agent up and the
other down) is not explicitly settled by the paper's examples. The Python
implementation treats such cells as ineligible; this is an explicit
conservative implementation convention, not a claim of native software
parity. Ties use row-major dose-grid order as a deterministic Python
convention.

The exact toxicity-boundary fixture uses the binary-representable value
`0.25`. An initial decimal `0.3` fixture exposed different rounding after
joint cells were serialized: two stored cells summed to a value one floating
step above the intended threshold. The reference inputs were corrected;
the implementation retains the source's literal strict comparison without
an added tolerance. The outer stopping boundary is exactly `0.75`.

Regenerate the small CSV fixtures with:

```sh
Rscript tools/reference_u2oet_gao2010_conduct.R tests/fixtures
```

Root integration validation passed all 19 affected probability, fitter and
conduct checks with warnings as errors: 2.475 seconds, 137.89 MiB process
peak RSS and zero swaps. The four supplied-draw cases contain 2,304 joint
cells and 36 grid-summary rows; numeric comparisons use absolute tolerance
2e-12 and no relative tolerance. The cohort checks run the actual fitter
with fixed parameters and supplied outcome tapes, including toxicity-only
observations and invalid-start rejection before random-state consumption.
These checks complement the existing independent posterior-quadrature audit;
they do not establish native executable or operating-characteristic parity.
