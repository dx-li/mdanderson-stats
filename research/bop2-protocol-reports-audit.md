# BOP2 protocol report source audit

The cached online application snapshot `research/raw/BOP2/app.html` lists six
endpoint choices around lines 197–233: binary efficacy, binary toxicity,
joint efficacy/toxicity, multiple efficacy, ordinal efficacy, and time-to-event.
Its boundary, OC table, and report/download controls appear around lines
995–1016, 2236–2245, and 2299–2443. These establish a useful report workflow,
not an unimplemented statistical kernel.

The report builders in `src/mdanderson_stats/bop2_protocol_report.py` compose
existing calculations:

- Binary efficacy/toxicity: `bop2_binary_design` and
  `BayesianMonitoringDesign.operating_characteristics`; toxicity success is
  `complete_negative`, while its adverse-event input remains the observed
  toxicity rate.
- Ordinal and multiple efficacy: `bop2_paired_design`, `_cells`, and exact
  `BOP2PairedDesign.operating_characteristics`; category orders are CR/PR/other
  and 11/10/01/00 respectively.
- EffTox: `bop2_efftox_design` plus the same exact paired-category recursion;
  the joint category order is efficacy/toxicity 11/10/01/00, and the combined
  stop probability is not formed by adding overlapping marginal failures.
- Survival: `bop2_survival_design`, `total_time_boundary`, and
  `simulate_bop2_survival`; the displayed success uncertainty is the returned
  plug-in Monte Carlo standard error.

The cached application guides and papers are summarized with provenance in
`docs/bop2-sources.json` and the endpoint-specific BOP2 guides. The current
Python implementation has independently documented interpretation choices
for defaults and source ambiguities. This report records the actual supplied
settings and does not claim native optimizer parameter recovery or parity.
Native Word/Chinese report formatting and animation remain presentation gaps;
they are not represented as missing statistical calculations.
