# BOP2 retired desktop entry

Catalog entry **144** is a retired distribution of BOP2. The
[official desktop page](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/144)
lists version 1.3.6.1, dated April 15, 2020, and explicitly directs users to the
[online BOP2 application](https://biostatistics.mdanderson.org/shinyapps/BOP2/).
The Python implementation uses the same public method family as that successor,
catalog entry **112**.

| Successor endpoint/workflow | Python coverage |
| --- | --- |
| Binary efficacy or toxicity | [Posterior monitoring, exact operating characteristics and finite-grid calibration](bop2-binary.md) |
| Ordinal or multiple efficacy | [Dirichlet monitoring, correlated operating characteristics and calibration](bop2-paired.md) |
| Joint efficacy and toxicity | [Separate monitoring schedules and three-null error control](bop2-efftox.md) |
| Time to event | [Single-arm survival monitoring, calendar simulation and calibration](bop2-survival.md) |
| Sample-size selection | [Expected-enrollment and minimax searches](bop2-sample-size.md) |

[Saved Python protocol reports](bop2-protocol-reports.md) connect all six
successor endpoint families to captured settings, boundaries and operating
characteristics. The functional Python workflow is implemented for the online
entry and for this retired entry through its official successor; see the
[completion review](../research/bop2-workflow-completion-audit.md).
Native report formats, animation and optimizer equivalence remain unverified.
The original desktop binary is no
longer supplied by its catalog page and has not been audited for equivalence.
Successor features added after 2020 are not evidence that the retired desktop
had those same options.

The retirement notice and current online endpoint list were checked on
September 27, 2026 UTC. The current app reports 1.4.29.0, updated September 16;
the detailed numerical source audit in [bop2-sources.json](bop2-sources.json)
records the earlier 1.4.27.0 snapshot. This mapping does not assert numerical
parity with the newer build or overwrite that historical provenance.
