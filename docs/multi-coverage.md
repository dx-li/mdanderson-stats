# MULTI coverage audit

Audited against `source/multi.doc`, the desktop MAIN menus, and
`S/multi-1.0/README.library` in the archived MULTI download.
Catalog entry 50 is implemented with the documented solver, RNG, input and
report-format substitutions. This audit covers the desktop methods and all named
functions in the S library README.

| Original capability | Python implementation | Validation |
| --- | --- | --- |
| MAIN data changes, procedure execution and seeds | MultiSession | Data replacement, fresh null estimates, all groups dispatched, RNG replay |
| Session report files | MultiSession.write_report, format_report, write_text_report | JSON snapshots; readable tables; 24 native PDISP report cases; specialized-field and failure tests |
| RDDATA / QLEX file and terminal text | parse_multi_data, read_multi_data | Native lexical fixtures, acceptance/warning comparisons, input-order and file tests |
| SWFIT / schwed.fit | schweder_fit, schweder_bootstrap | Native desktop/S fits, bootstrap tests |
| SWFIT coordinate file / schwed.plot | write_schweder_data, plot_schweder | Native coordinate comparisons, rendered PNG inspection |
| BFFIT nine procedures / bonferroni / sidak | multiple_testing, rom_critical_values | Native adjustments and cutoffs; independent formula checks |
| SHFIT sharpened procedures | sharpened_testing | Native SHL/SHC rejection comparisons |
| BMFIT / betamix.k / betamix | fit_beta_mixture_k, select_beta_mixture, EM/ML fitters | Native starts, both EM implementations, direct optimizer comparisons |
| BMFIT simulated CVM | beta_mixture_bootstrap | Native refits of fixed samples, uniform statistic checks |
| BMPVPB / BMURPB | beta_mixture_testing | 90 native cases covering rank and input sequence |
| betamix.prob | BetaMixture.null_posterior | Native BPVAL and independent Bayes checks |
| NPFIT / multi.np | nonparametric_testing, nonparametric_pvalues | Separate desktop/S native fixtures, high-precision regression checks |
| multi.os | order_statistic_diagnostics | Native OSFIT fixtures |
| rcpval | clustered_pvalues | Distribution/covariance checks and explicit random state |
| set.multi.seed | Per-call seed or Generator | Reproducibility tests; original RNG stream not reproduced |

Documented numerical differences and corrected undefined behavior are detailed
in multiple-testing.md and beta-mixtures.md. NumPy random draws and the new direct
optimizer are not exact reproductions of the old RNG and optimization paths.

## Original report content audit

| Source/report routine | Content retained in the readable report | Evidence |
| --- | --- | --- |
| RDDATA / MAIN | Dataset source, observations, original indices, data changes | Parser native fixtures and session replacement/report tests |
| SWFIT | Nominal alpha, null estimate, fit failure; coordinate output | Settings/result fields, native Schweder tests, existing CSV exporter |
| BFFIT / PDISP | Method, alpha, rank, observation, p-value, adjustment or Rom alpha, rejection markers | 24 PDISP output fixtures across eight adjustment methods; separate Rom label/threshold test |
| SHFIT / SHDISP | Method, alpha, null estimate, rejected observations or none | Shared decision table and explicit rejection count; native sharpened decisions validated separately |
| BMFIT | Algorithm, tolerances/selection criterion, candidate K, log likelihood, component proportions and shapes, CVM and simulated p-values, selected model or failure | Nested candidate/model/bootstrap fields; explicit beta-component count; native fit/selection fixtures and report integration tests |
| BMPVPB / BMURPB | Ordered observations, reciprocal-density diagnostics and decisions | Sequence-aware step/rank/observation columns; input-sequence report test and native decision fixtures |
| NPFIT / PDISP | Alpha, observations, fitted diagnostic scores and rejections | Shared score table, null estimate and fitted-rank density/bandwidth fields; native NP1P fixtures |
| S-library returned values | Estimates, posterior probabilities, order-statistic diagnostics and simulations | Public numerical result APIs; session reports for supported run routes |

Reports use Markdown tables and significant digits, not exact Fortran spacing,
terminal pagination or the source's clamp of displayed numbers above .999999 to
one. Sharpened reports include all observations with rejection markers/counts,
which also identifies the rejected subset. Mixture and nonparametric historical
reciprocal-density quantities are labeled as scores rather than null posterior
probabilities. Explicit Python calls replace terminal menus, filename prompts,
report sinks and retry dialogues. Readable reports preserve source settings and
result content; strict JSON reports retain full numerical precision and RNG states.

Supporting numerical libraries and terminal utilities are replaced by the package's
NumPy/SciPy kernels, validated formulas and Python I/O, not exported as unrelated
generic Fortran-library ports. Source numerical bugs, undefined behavior and
compatibility choices remain documented in the method guides. This completed entry
does not establish completion of the other 137 catalog entries.
