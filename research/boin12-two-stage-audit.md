# BOIN12 two-stage source audit

The cached BOIN12 application help `research/raw/BOIN12/BOIN12Stop.txt`
describes an optional two-stage design: Stage 1 uses toxicity only for rapid
escalation; once at least S patients have been treated at one or more dose
levels, Stage 2 uses both toxicity and efficacy for optimization. It says
Stage 1 applies only the safety criterion, Stage 2 applies both safety and
efficacy, and recommends S=6. The cached app page `research/raw/BOIN12/index.html`
specifies S input range 6–12 and default 6. This implementation requires S
explicitly rather than silently adopting the app default.

The cached `BOIN12Stop.pdf` SHA-256 is
`34c5603b4c423f9f4db41a3dd0fca3682984bfa4a7d259e49400d3bea3099fd8`; the
cached `BOIN12Admissible.pdf` SHA-256 is
`aee120afaac19b7f70d5413b121dbb27d57f35f9c3d1f6989027dd8b772db443`.

Cached `research/raw/BOIN12/BOIN12Admissible.txt` defines safety admissibility
with a strict posterior-tail cutoff. Accordingly, a dose fails at equality as
well as above the cutoff. The existing BOIN12 posterior uses the marginal
Beta(1,1) toxicity model. Stage 1 computes this tail directly and reuses the
BOIN escalation/de-escalation rate boundaries, but does not call
`BOINDesign.next_dose`: that separate method has its own minimum-three-patient
safety convention. Stage 1 does not consult efficacy or utility for its
movement decision.

The help does not specify whether the triggering cohort is assigned in Stage
1 or Stage 2, how safety exclusion persists, how a trial stopping before S
selects an OBD, or the precedence of optional early stopping. Python policies
are: the triggering cohort is Stage 1 and the next assignment is Stage 2;
per-dose safety exclusions are sticky but are not propagated to higher doses;
safety stopping precedes `early_stop_patients`; and final OBD selection uses
all accrued joint outcomes even if the transition never occurs. The ordinary
BOIN12 final selector continues to apply its own posterior eligibility and
selection rules.

Primary references: cached BOIN12 help and application page above; Lin et al.,
“BOIN12: Bayesian Optimal Interval Phase I/II Trial Design for Utility-Based
Dose Finding in Oncology,” *JCO Precision Oncology* 4 (2020), 1393–1402,
<https://pmc.ncbi.nlm.nih.gov/articles/PMC7713525/>. No native two-stage
outcome fixture was located; numerical validation uses exact beta-tail
identities and deterministic cohort ledgers.

## Validation

Twenty-two affected BOIN12 tests pass with warnings treated as errors, with
129.23 MiB peak resident memory and zero swaps. Targeted Ruff formatting/lint
and mypy checks pass. An independent integration check evaluates 320 Stage 1
safety decisions against 70-digit decimal finite-binomial sums equivalent to
the integer-shape Beta posterior tails. It also checks efficacy invariance,
adjacent safe destinations, a deterministic stage-transition ledger, and work
budget rejection before random draws. The combined numerical audit with
BF-BOIN titration took 0.265 seconds after imports, peaked at 125.12 MiB and
reported zero swaps. Source backend random-stream parity is not claimed.
