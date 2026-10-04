# Remaining BARD simulation workflow

The cached official BARD guide and complete paper, including its supplement,
identify work beyond the implemented allocation and final-selection functions.
Source URLs and hashes are recorded in [the provenance record](../docs/bard-sources.json).
This review used those cached files; no live application was accessed.

## Advertised simulation scope

The guide's Section 2 (printed pages 12–14) accepts dose-specific population
toxicity and response probabilities, categorical prognostic-factor probabilities,
and response odds ratios relative to each factor's first level. It produces
operating-characteristic tables for the two-stage design. Sections 1 and 3
also connect trial settings, stage-one boundaries and a saved protocol.

The paper's numerical-study section (printed pages 18–19; cached text around
lines 768–823) specifies a dose-specific logistic response model with three
binary prognostic factors. The coefficients are 1.7, -1.5 and 0.4. Only the
first two factors enter minimization, so the third provides an omitted-factor
balance comparison. Supplement Section 1 and Table S1, present in the same
cached paper around lines 1398–1434, provide intercepts for eight five-dose
scenarios. Supplement Section 3 and Tables S4–S5 provide four three-dose
scenarios. The factor coding in the supplement uses levels 1 and 2, with an
indicator for level 2; it should not be confused with a numeric covariate that
uses 1 and 2 directly in the linear predictor.

The paper defines per-factor imbalance as the absolute difference in the
proportion at a factor level between the two arms, and allocation imbalance
as the absolute difference in arm counts. It also reports sample size,
duration, and correct OBD selection under both noninferiority and utility.
These statistics need a full two-stage patient history and explicit true OBD;
they cannot all be recovered from a stage-one dose-selection average.

## Existing components and remaining work

The package supplies BF-BOIN stage-one simulation, BF-BLRM fitting and calendar
replay, patient-wise minimization, OBD selection, and a supplied-outcome
BF-BLRM continuation. The [response model](../docs/bard-response.md) now
calibrates conditional response probabilities under an explicit joint factor
distribution; [published scenario records](../docs/bard-response-scenarios.md)
preserve the recovered table inputs. These do not yet provide the guide's
complete BF-BOIN two-stage scenario simulation with retained patient covariates,
automatic stage-one carryover, both final OBD summaries and balance metrics.

Clinical eligibility and the dose pair are partly protocol inputs. The paper
explicitly permits eligibility to differ between stages and selects the pair
using clinical evidence. The guide supplies MTD and its adjacent lower dose
as its default pair. Automating clinical judgment is not a completion
requirement; retaining explicit eligibility and pair inputs is appropriate.

The guide asks for a target count per arm, while the paper's general formula
specifies a combined target including stage-one carryover. Neither inspected
description gives a full hard-quota enforcement algorithm. Stage-two calendar
timing and native joint outcome-generation conventions also need explicit
policies or stronger evidence. Those source uncertainties are separate from
the unfinished integration and operating-characteristic outputs. BARD remains
partial.

## Response-model component and next integration

The response-model calibration API solves each dose's logistic intercept so
that its profile-weighted conditional response probabilities match the entered
population response probability. Factor effects are log odds ratios relative
to level 1. The API accepts an explicit joint profile distribution;
constructing independent factors from their marginal probabilities is a
caller convention because the guide does not specify dependence.
The paper's printed intercept table provides a separate numerical cross-check,
with tolerance for its rounded coefficients. This component does not determine
the joint toxicity/response law, which remains an explicit simulator input or
policy. The next integration must retain each generated patient's factors and
conditional response probability through stage-one assignment, carryover and
stage-two minimization, without changing existing stage-one random streams
when the new model is absent.
