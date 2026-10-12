# Next bounded native-workflow completion contracts

ARAND's guide-defined simulation crosswalk is separate from the following work.
This note identifies concrete checks for the next implementation pass; it does
not promote CRM/BMA-CRM or ToxFinder and does not claim new native execution.

## CRM Suite132 and BMA-CRM81

The current saved-study runner explicitly accepts `PriorsAre_P=1`,
`DoseEscalationRule=eStrict`, and `MinPatientsToDeclareMTD=3`; other values fail.
Determine from each version's guide and original controls which alternatives
are advertised. Recover their input meaning and decision/final-selection
semantics, then implement and independently validate any required alternatives.
Do not infer that the three installer examples exhaust the application contract.

Simulation and conduct are different original paths: the conductor can perform
within-cohort safety checks, while simulation retains the cohort assignment.
A populated native patient-file workflow needs an explicit locale/date contract
and separate controller references. Existing numeric-time replay is useful but
does not validate guessed date parsing or the native conduct path. Saved patient
text is currently preserved without being used as prospective simulated data.

The older BMA-CRM guide's DA safety-wait/full-information stop convention must
remain a separately versioned profile. Current CRM Suite scheduling and native
hazard-prior recovery do not by themselves establish the older desktop contract.
Online entry133's automatic skeleton calibration is another distinct task:
resolve the Q equation, regression intercept and simulation-based candidate
selection before claiming its advertised calibration is complete.

Independent posterior inference need not reproduce a native random bitstream.
However, a native shortcut that changes decisions (rather than merely stopping
an already-converged sampler) is a source-contract question and needs bounded
branch evidence or an explicit, independently validated alternative profile.

## ToxFinder14

Stage-two information, both advertised allocation modes, paired cohorts, native
stochastic prior elicitation, observed-history filtering, TMML/CSV and reporting
already have substantial implementation and original-reference evidence. Do not
repeat recovery of the old "unknown information determinant" blocker.

Review the original advertised user journey from physician inputs through prior
calibration, a saved study, observed-history recommendation, simulation and report.
The TMML reader currently preserves elicitation/MCMC controls as metadata;
execution takes a validated prior and explicit Python sampler settings. Determine
which retained controls require a documented mapping or a complete executable
recipe. Validate that recipe and its quality diagnostics independently before
closing the application. Exact native RNG, GUI layout and cohort-log bytes are
not themselves missing statistical methods.

Retain source-versus-corrected choices: the native final posterior precedes the
last cohort/pair; native GUI "SD" columns actually use SEM and sentinels; the
stage-one out-of-bounds boundary has an explicit safe correction. Do not turn
these into hidden scientific defaults or remove their evidence.

## Preserved external dependencies

The six pending fitted models still need exact unavailable parameters/contracts;
read their existing evidence before another search. No approximate substitution
is authorized. The full distribution remains mixed-license. The concrete core/
legacy split proposal is unactivated and requires a project decision. PR creation
was Forbidden; no bypass, merge, registry release, deployment, outreach or access
change has been performed.
