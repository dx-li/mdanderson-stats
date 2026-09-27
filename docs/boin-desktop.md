# BOIN desktop method coverage

Catalog entry **99**, the [BOIN desktop program](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/99),
collects single-agent, delayed-toxicity and drug-combination trial designs.
The Python package provides these methods through the APIs below. It shares
implementations with the separately listed web applications rather than
maintaining a second statistical backend for the desktop entry.

| Desktop capability | Python interface | Coverage |
| --- | --- | --- |
| Single-agent dose finding | `BOINDesign`, `simulate_boin` | Boundaries, conduct, safety, isotonic selection, simulation and accelerated titration |
| Conventional 3+3 comparison | `compare_boin_three_plus_three` | Independent simulation and the documented sample-size matching options |
| Late-onset toxicity | `tite_boin_estimate`, `tite_boin_decision`, `run_tite_boin_trial`, `simulate_tite_boin` | Imputation, follow-up gates, calendar replay, simulation and final complete-data selection |
| Rolling Six comparison | `compare_tite_boin_rolling_six`, `tite_boin_rolling_six_report` | Common scenario inputs, independent serial simulations and comparative summaries |
| Combination single MTD | `BOINCombDesign`, `simulate_boin_combination` | Conduct, safety, weighted bivariate isotonic selection and ordinary cohort simulation |
| Combination MTD contour | `BOINCombDesign.select_mtd(mtd_contour=True)`, `next_subtrial` | Final contour estimation and interactive waterfall subtrial planning; full waterfall simulation remains open |
| Standardized follow-up calculator | `toxicity_followup_weights` | Per-patient ordinary or informative-prior weights, summed to STFT/WSTFT |
| Statistical protocol text | `boin_protocol` | Single-agent Markdown methods text in English or Chinese |

Detailed conventions and numerical references are in the
[single-agent BOIN](boin.md), [TITE-BOIN](tite-boin.md),
[Rolling Six](rolling-six.md) and [combination BOIN](boin-combination.md)
documentation. The [source record](boin-desktop-sources.json) distinguishes
desktop documentation from the separately audited R and web implementations.
The [time-to-toxicity comparison example](boin-time-comparison.md) demonstrates
the coordinated TITE-BOIN/Rolling Six workflow.

## Standardized follow-up

Only pending patients contribute to standardized total follow-up. Times and
the assessment window must use the same unit. With a uniform timing prior:

```python
import numpy as np
from mdanderson_stats import toxicity_followup_weights

weights = toxicity_followup_weights([30, 48, 75], window=90)
np.testing.assert_allclose(weights.sum(), 1.7)
```

For informative timing priors, supply `trimester_probabilities`; these describe
when a DLT occurs conditional on its occurrence, not overall DLT incidence.
`tite_boin_decision` already calculates the per-dose weighted totals from pending
follow-up times. No separate numerical engine is needed for the calculator.

## What remains

Entry 99 is partial. Full waterfall simulation, combination titration/run-in
workflows, combination and TITE-specific protocol generation, native Word/HTML
layouts and saved-project interoperability remain open. The existing interactive
waterfall planner is not a full simulator; the R simulator has additional
subtrial transitions and stopping conventions.

The desktop installer and its embedded help have not been executed or inspected
for numerical equivalence. The Python APIs follow the cited methods and audited
R/web contracts, whose defaults and scheduling conventions can differ from
desktop version 1.1.0. No desktop executable, native project file or unverified
simulation result is bundled.
