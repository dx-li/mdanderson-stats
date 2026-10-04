# Interaction-index workflow coverage

The cached `research/raw/CIInteractionIndex/ReadmeCI.pdf` advertises three
inferential routines, median-effect plots, two case-study scripts and two
simulation-study scripts. Its original archive was not inspected.

| Source routine or example | Community implementation |
| --- | --- |
| `CI.known.effect`, Section 2 | Observed-combination log-delta inference, with explicit response variance or the documented pooled-error fallback |
| `CI.delta`, Section 3 | Fixed-ray log-delta intervals using the separate fitted coefficient covariances |
| `CI.simulation`, Section 3 | Normal-coefficient Monte Carlo comparator with retained draws and slope-reversal diagnostics |
| Median-effect plots and case studies | Optional plot helpers and executable Table 2/3 examples with saved figures and numerical records |

The case-study data come from Section 4.2, Tables 2/3 and Figures 4/5 of
[Lee and Kong (2009)](https://doi.org/10.1198/sbr.2009.0001). The cached BioC
table passages retain their embedded MathML, including the mixture-total-dose
convention. Table 2 models fractional survival; Table 3 models fractional
inhibition. The examples preserve those responses and use total mixture doses.

`tools/reference_interaction_index_cases.R` independently fits the six curves
with base R and propagates their coefficient covariances. Python agrees with
these full-precision references within absolute tolerance `1e-11`; the six
published fitted coefficients, median doses and residual standard deviations
agree within `0.0005`. The four published Table 2 observed indices and confidence
intervals agree within `0.001`. In particular, the first lower limit is
`0.202556...` from the printed observations versus the reported `0.202`.
The rounded source numbers do not support claiming exact equality.

This comparison supports the explicit residual-degree-of-freedom pooled mean
square used by Python. It does not establish an undocumented native pooling
denominator: the independent calculation intentionally uses the same stated
statistical convention. No covariance is pooled for the fixed-ray method.

## Remaining advertised simulation work

The cached Section 4.1 describes a three-drug repeated-sampling study in enough
detail to implement its generation, raw/log-delta coverage, interval lengths
and classification summaries. Its printed `1.67` interaction index and `0.625`
response do not establish the exact unrounded constant used by the original
script; a Python study must record the value actually used.

For the second, fixed-ray simulation, the cached text omits the composition
ratio from inline mathematics. The marginal and mixture models alone cannot
determine it. Its exact source-specific demonstration therefore remains
unimplemented. The general fixed-ray and Monte Carlo APIs already accept
explicit proportions; choosing those proportions would not recover the
missing demonstration contract. Catalog entry 65 remains partial.
