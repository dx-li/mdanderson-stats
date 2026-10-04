# BOP2 functional workflow coverage review

This review applies a functional Python workflow criterion: a user can specify
the documented design, inspect decision boundaries and operating
characteristics, and save a report of the calculation. It does not require the
original Shiny controls, animation, file bytes, language-specific formatting,
or an undocumented optimizer's exact search path.

## Online BOP2 (catalog entry 112)

The cached online application exposes binary efficacy/toxicity, ordinal and
multiple efficacy, joint efficacy/toxicity, and time-to-event analyses. The
current Python package has specified-design monitoring and report paths for
each advertised family:

| Source workflow | Python workflow |
| --- | --- |
| Binary efficacy or toxicity | `bop2_binary_design`, `optimize_bop2_binary`, and binary efficacy/toxicity builders in `bop2_protocol_report.py`; [guide](../docs/bop2-binary.md) |
| Ordinal or multiple efficacy | `bop2_paired_design` and ordinal/multiple builders in `bop2_protocol_report.py`; [guide](../docs/bop2-paired.md) |
| Joint efficacy/toxicity | `bop2_efftox_design` and `bop2_efftox_report`; [guide](../docs/bop2-efftox.md) |
| Time-to-event | `bop2_survival_design` and `bop2_survival_report`; [guide](../docs/bop2-survival.md) |
| Saved specified-design analysis | Six endpoint-specific builders in `bop2_protocol_report.py`; [report guide](../docs/bop2-protocol-reports.md) |

The report source audit maps each builder to the existing monitoring, exact
operating-characteristic, or survival simulation kernel rather than treating
the report as a new statistical algorithm. Endpoint guides document their
assumptions and validation examples; the report audit records the endpoint
category order, combined-error handling, and which outputs are exact versus
Monte Carlo. No additional source-defined numerical calculation remained
unimplemented in this review.

Native Word/Chinese report formatting, animation, and optimizer equivalence
remain unimplemented. The Python finite-grid calibration workflow is explicit
about its search and tie policy and does not claim to reproduce an
undocumented native optimizer. These are native-presentation or
implementation-convention differences, not evidence that the documented
endpoint calculations are absent.

## Retired BOP2 desktop (catalog entry 144)

The retired desktop page directs users to the online BOP2 application. The
Python coverage above supports the successor workflow. It is not a claim that
the historical desktop binary was separately inspected or that every feature
of that binary matched the current online version. Successor features added
after retirement cannot be attributed retroactively to the desktop.

The scoped implementation and source crosswalks are in
[`bop2-protocol-reports-audit.md`](bop2-protocol-reports-audit.md) and
[`bop2-desktop.md`](../docs/bop2-desktop.md). This document records a
functional-coverage assessment; it does not change catalog status or assert
native application parity.
