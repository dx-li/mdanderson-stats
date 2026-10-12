# ARAND guide-scope completion assessment

The recovered version 5.2 guide defines ARAND as software for **simulating**
clinical trials with binary or time-to-event outcomes. It does not advertise a
clinical recommendation service. Its required outputs are per-arm selected,
selected-early, dropped-early and enrollment summaries, with empirical intervals,
plus average trial length and saved input/output. A source library's latent
patient-file APIs are not themselves an additional advertised application.

The Python native-controller profile now provides the following end-to-end
simulation contracts. Entry 62 is classified as implemented for these contracts after the independent
reference and installed-workflow checks. The final regression/artifact checkpoint
is recorded in the integration report. This is not original machine-code identity.

| Guide/workflow contract | Implementation and independent evidence |
|---|---|
| Binary and TTE models, up to ten arms, largest/smallest objective | Independent beta/inverse-gamma posterior APIs;52 independent R integrations, exact identities, censor/exposure and mean/median scaling tests. |
| Adaptive assignment, initial balancing, tuning/floor, stopping and follow-up | Named native controller;98 original instruction fixtures,4200 seeded status/minimum/constraint/per-arm comparisons, supplied-calendar end-to-end checks. |
| Separate native enable flags and arm-specific stopping | All eight gate combinations (256 cases);60 original per-arm cases exercise status and both final selectors. |
| Batch monitoring initialization and replications | Six construction/reset slices establish ceil(a) versus int(a). The actual native wrapper loop has six original x86 probes showing director-reset → trial-run → accumulation per replication. |
| Original simulation input files and saved Python studies | Hash-pinned original managed serializer validation/round trips for Binary/TTE; strict noncovariate Months XML mapping, retained entire source XML/history metadata; versioned Python JSON read/write and immutable inputs. |
| Reproducible scenarios and replications | Original SetSeed IL confirms base seed plus scenario index (five probes). Explicit independent NumPy streams record each replay seed. Four original clock-slice fixtures confirm a zero first candidate; independent Poisson/Gamma identities verify enrollment and terminal timing. |
| Per-arm selection, early selection, early drop, enrollment | Full original accumulation body0x59ac0 (12 fixtures) and actual native report0x18bf90 slices. The actual report counts statuses1+2+3 as drops, unlike the earlier base-template candidate's1+2. |
| Empirical enrollment intervals | Actual native x86 report probes establish suppression below100 and legacy sorted indices3/98 at100. The standard type7 default and explicit legacy-5.2.2 report option are independently checked. |
| HTML input/output and export | Self-contained report captures all study parameters and replay seeds. Original XML can be retained byte-for-byte as text; modified Python inputs use versioned JSON. Installed examples execute Binary/TTE import→simulation→report. |

## Boundaries that must remain explicit

- Original RNG bitstream and posterior-integrator bit identity are not claimed.
  Independent mathematics and NumPy sampling are a declared supported profile,
  not an approximation for an unavailable fitted model.
- The old Windows GUI, exact HTML styling/CSV bytes, dormant covariate/predictive
  models and clinical-history conduct are not silently claimed. Unsupported XML
  branches fail; preserved patient metadata is not used as simulated data.
- Native clinical-file date origins and pending-record behavior remain research
  work if a separate conduct feature is requested. Managed `time` and
  `CurrentTime` setters are no-ops; no GUI defect is inferred from synthetic
  records that assigned those properties.
- Legacy batch reset and percentile behavior are explicit compatibility choices.
  Undefined/overflow schedules and unresolved numerical tails fail safely.
- Scientific implementation, compatibility with declared source behavior, and
  clinical validation are separate claims. No clinical validation is asserted.
- Existing mixed-license terms remain. This scope assessment does not grant
  redistribution rights for the original application or remove bundle restrictions.

## Source-dispatch caution resolved

Managed thunks must be resolved through their CLR fixup MethodDef tokens; the IL
body following a thunk is not necessarily its target. The reproducible dispatch
manifest maps base and derived slots. Derived native simulation is0x18d3e0 at
slot12, native report0x18bf90 atslot16; accumulator0x59ac0 is shared atslot32.
The initial adjacency-based slot interpretation in c45b46b1 is superseded by
that complete token-resolved manifest and actual native report/loop execution.
