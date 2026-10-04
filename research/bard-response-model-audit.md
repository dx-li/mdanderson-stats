# BARD categorical response-model audit

The official cached BARD guide, `research/raw/BARD/Guide.txt`, Remarks 1
(printed pages 12–13; extracted text lines 251–276), asks for categorical
factor-level probabilities and odds ratios relative to level 1. The cached
paper, `research/raw/BARD/paper.txt` lines 763–770, gives the response model
used in its simulations as an additive logistic predictor with three binary
factors and dose-specific intercepts; Supplementary Table S1 supplies
intercepts for published scenarios. This supports a callable logistic model
and calibration to marginal response rates.

The guide's factor marginals do not specify the joint distribution among
multiple factors. The implementation therefore takes a joint profile matrix
and its probabilities directly; it does not silently assume independence.
For profiles \(x_p\) with supplied joint weights \(w_p\), the calibrated
intercept at dose \(d\) solves

\[
\sum_p w_p\,\operatorname{logit}^{-1}\!\left(
  a_d + \sum_f \log(OR_{f,x_{pf}})
\right)=r_d.
\]

The supplied odds ratios are defined as conditional effects, holding the
other factors fixed. The code does not treat these as marginal odds ratios;
odds ratios are not generally collapsible over other covariates. Exact zero
and one margins are represented by infinite intercepts and deterministic
probabilities. Interior margins use a tail-aware monotone solve, with an
explicit failure when floating-point precision cannot resolve a nearly flat
inverse.

The implementation and tests are in `src/mdanderson_stats/bard_response.py`
and `tests/test_bard_response.py`. The companion scenario records supply
published intercepts and factor coefficients and can be evaluated with the
same probability function. This component does not provide the full BARD
two-stage operating-characteristic simulation or native Shiny input/output
parity; factor dependence must be supplied by the caller.
