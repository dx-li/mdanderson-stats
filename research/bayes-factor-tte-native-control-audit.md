# BayesFactorTTE v1.1 recovered control audit

October 10, 2026: the original official
[BayesFactorTTE_V1.1_NoFX4.0.zip](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BayesFactorTTE/BayesFactorTTE_V1.1_NoFX4.0.zip)
was recovered, SHA-256
`7fa329b68f7ac60887571087b38820048c5bdd0ce7a95ca9780ce1df5ef89c5a`.
It contains `BayesFactorTTELib.dll`, the executable, numerical dependencies,
original sample input and user guide. Original artifacts remain ignored and
excluded from distribution.

`tools/reference_bayes_factor_tte_control.py` verifies the DLL checksum, decodes
original managed instructions and executes bounded original method bodies.
The fixture records the full DLL hash. Relevant native method RVAs are:

| Original managed method | RVA | Contract |
| --- | --- | --- |
| `Simulator.SimulateOneTrial` | `0x29a4` | Check after enrolling patients 2..N; end at final enrollment |
| `Simulator.InitEnrollandEventTime` | `0x2a30` | Alternate event/arrival exponential draws; truncate each to integer days |
| `Simulator.UpdatePatientLog` | `0x2a94` | Earlier-patient exposure; strict event-before-look comparison |
| `TrialConductor.GetDecision` | `0x2794` | Obtain BF, pass enrolled count to stopping rule |
| `Simulator.CheckStoppingRule` | `0x2c80` | Strict superior/inferior cutoffs; otherwise continue or inconclusive at maximum |
| `Simulator.GetBoundariesInDays` | `0x2b10` | First noninferiority day / first superiority day; finite scan horizon |
| `BiostatTimeSpan..cctor` | `0x3216` | 30.4375 days/month and 365.25 days/year |

Constructors/runtime objects, exponential RNG and Bayes-factor services are
explicit substitutions. Original controller instructions execute unchanged.
Three synthetic tapes cover final inconclusive, immediate inferiority and later
superiority. Fractional draws, zero gaps/durations and ties verify truncation
and ledger details. A fourth exercise executes the original integer search with
an external monotone synthetic BF `day/(events+1)`; its independent analytic
roots verify the Python integer convention, including exact integer roots.
The harness imports no production implementation.

This is original controller validation, not independent native integration or
RNG validation. The existing mathematical kernel has separate original-mean-
coordinate quadrature, published guide boundaries and unit-invariance checks.
Additional tests compare actual day-boundary neighbors with direct iMOM
evidence, positive-exposure ledgers with the continuous kernel, simulation
replay, immutable tapes, saved records and preflight guards.

Python supports zero-exposure event patterns created by native truncation in
this workflow only; the continuous-data API remains unchanged. Native int32
arithmetic overflow is corrected with explicit guards. The original one-patient
check-loop omission is rejected. Patient quantiles preserve sorted-index
selection, while reports retain full numerical precision. Continuous-root
inversion removes the arbitrary native search cap and incomplete lists;
quadrature error/CLR identity remain numerical compatibility limits.

Catalog entry 89 now completes the recovered functional workflow: evidence,
continuous/integer boundaries, explicit/native-scheduled conduct, bounded
simulation, original patient quantiles and saved numerical/HTML reporting.
Native GUI, .NET runtime, historical input bytes and RNG identity are not the
functional coverage criterion.

The original boundary exporter approximately disables sides within `1e-9` of
cutoffs 0/1, although its trial rule uses the supplied strict cutoff. Python
keeps the requested cutoff active and disables a side only at exact 0/1,
avoiding that exporter/decision inconsistency. Native TimeSpan millisecond
rounding for arbitrary fractional-month inputs is also a CLR compatibility
difference; default/example month conversions are exact under the recovered
30.4375-day convention.
