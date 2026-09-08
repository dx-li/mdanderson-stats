# MULTI coverage audit

Audited against `source/multi.doc`, the desktop MAIN menus, and
`S/multi-1.0/README.library` in the archived MULTI download.
Catalog entry 50 remains partial because the change-data session and report-file
workflow are still missing.

| Original capability | Python implementation | Validation |
| --- | --- | --- |
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

Remaining: the interactive change-data/session workflow and the report-file
writer that records procedure
settings, observations, and result tables. Python arrays and result objects expose
the numerical data, but those I/O capabilities have not yet been implemented.
The main statistical groups being covered does not establish completion of the
whole catalog entry or the other 137 catalog entries.
