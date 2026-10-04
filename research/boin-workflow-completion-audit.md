# BOIN Python workflow completion

Catalog 120 is implemented under the functional Python completion standard
used by [Keyboard](../docs/keyboard-protocol-report.md) and the earlier
[report workflow audit](report-workflow-completion-audit.md). This review adds
no numerical method. It reconciles the status with the already integrated
calculation and reporting workflows.

| Source task | Python interface and evidence |
| --- | --- |
| Design by probabilities or direct boundaries | `BOINDesign` and `from_boundaries`; stable inversion with independent high-precision checks |
| Dose decisions, safety, optional modifiers and stopping | `next_dose`, sticky exclusions and source-defined actual-stay precision stopping |
| Final MTD and toxicity summaries | `select_mtd`, weighted isotonic estimates, source reporting intervals and separate safety tails |
| Fixed-cohort and accelerated-titration trials | `simulate_boin`, explicit grade-2 truth, capped enrollment and stopping-reason summaries |
| Conventional 3+3 comparison | `compare_boin_three_plus_three`, both enrollment-matching choices and retained trial accounting |
| Operating characteristics and overdose allocation | Selection/MCSE outputs and the source-defined strict planned-enrollment 60%/80% risks |
| Reproducible protocol output | `boin_protocol` in English/Chinese and `boin_design_report` with captured inputs, boundaries, scenario summaries and saved HTML |

The [method guide](../docs/boin.md), [source provenance](../docs/boin-sources.json),
[allocation-risk audit](dose-allocation-risk-audit.md) and
[report guide](../docs/boin-protocol-report.md) document the numerical references
and supported settings. Existing tests cover native boundary/MTD fixtures,
analytical trial probabilities, numerical inversion, optional conduct rules
and report-to-simulation replay. No further advertised calculation was
identified in this review.

The original animation, editable web session and Word template are not
reproduced. Python random streams differ from R, and explicit tie handling
can differ from R's artificial perturbation in pathological near-ties. These
are retained compatibility limits, not claims of exact application equivalence.
The separately cataloged desktop, combination and time-to-event programs keep
their own statuses and unresolved contracts.
