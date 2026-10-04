# BARD published response scenario audit

The cached primary text is `research/raw/BARD/paper.txt`; the cached app guide
is `research/raw/BARD/Guide.txt`. This module transcribes scenario inputs; it
does not implement an allocation or OBD rule.

## Source crosswalk

| Python ID | Source values | Cached text anchors |
| --- | --- | --- |
| `five-dose-1` … `five-dose-8` | Table 3 DLT and marginal efficacy rows; Supplementary Table S1 intercepts | `paper.txt` lines 1238–1283 and 1418–1428 (paper Section 3, Table 3; supplement Section 1, Table S1) |
| `three-dose-1` … `three-dose-4` | Supplementary Table S4 DLT and marginal efficacy rows; Table S5 intercepts | `paper.txt` lines 1580–1610 (supplement Section 3, Tables S4–S5) |
| Shared response coefficients and factor marginals | Three binary prognostic factors are each described as generated from Bernoulli(0.5); coefficients are 1.7, −1.5, and 0.4 | `paper.txt` lines 763–773; model equation and coefficients at lines 1403–1413 and 1551–1561 |
| Scenario entry workflow | Users provide true DLT and population response probabilities by dose | `Guide.txt` lines 224–240 |

The model equation codes each factor through an indicator for level 2. The
scenario record therefore retains levels `(1, 2)` with equal marginal
probabilities. The text does not state that the three factors are mutually
independent. The stored marginal probabilities do not imply a joint law; a
caller that averages over all eight factor profiles using product weights is
choosing an explicit independent-factor convention.

The intercept-based conditional response probabilities and their chosen
profile marginalization are distinct from the table's printed marginal
probabilities. Intercepts are rounded to three decimals and printed truths to
three decimals, so small differences after logistic transformation and
marginalization are expected. Neither value is substituted for the other.

The paper also prints utility rows and highlights an OBD, but those are not
stored here. The source material does not provide a joint toxicity-response
generation law sufficient to reconstruct expected utility from the marginal
toxicity and response values alone. No true-OBD tie rule or policy label is
inferred by this data module.
