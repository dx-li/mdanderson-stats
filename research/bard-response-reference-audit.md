# BARD response-model independent reference

## Source contract recovered

The cached primary text is `research/raw/BARD/paper.txt`. In the simulation
methods (main-text pp. 17–18; extracted lines 767–823), the response model is an
additive logistic regression over three binary prognostic factors with
coefficients `1.7`, `-1.5`, and `0.4`; scenario- and dose-specific intercepts
are listed in Supplementary Table S1 (lines 1398–1434). The main-paper Table 3
(lines 1245–1285) reports scenario-level response probabilities rounded to
three decimals. The paper says each factor is Bernoulli(0.5), but does not
spell out cross-factor dependence in that sentence; the reference treats the
three factors as mutually independent for its reconstructed profile weights
and identifies that as an assumption.

The cached user guide `research/raw/BARD/Guide.txt`, Remarks 1 (lines
245–276), defines categorical factor effects as odds ratios relative to level
1. In a main-effects logistic model, `exp(beta)` is the conditional OR holding
the remaining factors fixed. Averaging predicted response probabilities over
the other factors generally produces a different marginal OR. That difference
can reflect odds-ratio non-collapsibility and, when factors are associated, a
changed distribution of the other factors. Since the guide's
displayed probability notation does not fully specify how other factors are
averaged, this oracle reports both quantities; it does not infer a native
mapping between them.

## Independent calculation

`tools/reference_bard_response.R` uses base R only. Given a complete categorical
profile table, normalized joint probabilities, and positive reference-coded
ORs, it computes profile linear-predictor offsets as the sum of log ORs. It
then calibrates a scalar intercept by `uniroot` on the strictly increasing
weighted marginal response function. A one-factor binary case is also checked
against its closed-form quadratic root. A correlated two-factor joint table
checks marginalization separately from conditional ORs. The supplement's
eight-by-five intercept table and paper Table 3 response values provide a
published-coefficient reconstruction check.

This is independent of the Python calibration and profile-probability code.
Printed Table 3 rates and Supplement S1 intercepts are rounded, so the
reference records observed discrepancies and uses only the rounding precision
to set check tolerances. It does not tune coefficients to force exact printed
matches. It neither specifies joint toxicity-response dependence nor
reconstructs any native response optimizer, allocation process, random stream,
or interface behavior.

Run with `Rscript tools/reference_bard_response.R`; the script prints
calibration, conditional/marginal OR, and source-table discrepancy results.

The independent run passed. The binary closed-form check gives intercept
`-1.098612` for prevalence `.4`, marginal target `.35`, and conditional OR `3`.
In the correlated two-factor example (target marginal response `.42`), it gives
intercept `-0.963784`; factor 1's conditional OR is `2.5` while its marginal OR
is `3.600689`, and factor 2's conditional ORs `.6, 3` differ from marginal ORs
`.967824, 5.623540`. Across the eight scenarios and five doses, the largest
absolute difference between rates reconstructed from rounded S1 intercepts
and printed Table 3 rates is `0.0004946`; recalibrating from three-decimal
Table 3 rates differs from rounded S1 intercepts by at most `0.01121`.
The first discrepancy is consistent with the two tables' displayed precision;
the intercept discrepancy is recorded rather than tuned away.

The full-precision CSV output was compared with Python core commit `0c98e0e8`.
Across all 40 source scenario/dose predictions, the largest Python-to-R rate
difference was `4.45e-16`; recalibrated intercepts differed by at most
`8.93e-14`, and Python's largest marginal residual was `8.99e-15`. In the
correlated 2x3 case, the Python intercept was `-0.9637837289014297` versus
R's `-0.9637837289014656`; the conditional and marginal ORs agreed to below
`1e-14`. This independently checks the numerical calibration and profile
mapping, not native software behavior.
