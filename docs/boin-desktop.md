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
| Combination MTD contour | `BOINCombDesign.select_mtd(mtd_contour=True)`, `next_subtrial`, `run_boin_waterfall_trial`, `simulate_boin_waterfall` | Final contour estimation, interactive planning, full no-titration replay and serial simulation |
| Standardized follow-up calculator | `toxicity_followup_weights` | Per-patient ordinary or informative-prior weights, summed to STFT/WSTFT |
| Statistical protocol text | `boin_protocol` | Single-agent Markdown methods text in English or Chinese |

Detailed conventions and numerical references are in the
[single-agent BOIN](boin.md), [TITE-BOIN](tite-boin.md),
[Rolling Six](rolling-six.md) and [combination BOIN](boin-combination.md)
documentation. The [source record](boin-desktop-sources.json) distinguishes
desktop documentation from the separately audited R and web implementations.
The [time-to-toxicity comparison example](boin-time-comparison.md) demonstrates
the coordinated TITE-BOIN/Rolling Six workflow. The
[waterfall guide](boin-waterfall.md) documents complete subtrial transitions
and numerical differences from the original R simulator.

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

## Saved desktop workflows

`read_boin_desktop_project` reads the current native XML/HTML envelope and retains
inactive/unknown metadata. `add_boin_desktop_patient`, `update_boin_desktop_patient`,
`delete_last_boin_desktop_patient` and `clear_boin_desktop_patients` support the
combination patient file. `recommend_boin_desktop` connects managed cohort/pending
rules to active single-MTD or contour decisions and final selection.
`boin_desktop_conduct_html` creates escaped, reopenable diagnostic reports.

```python
from pathlib import Path
import mdanderson_stats as m

study = m.read_boin_desktop_project(Path("my-study.html").read_text())
batch = m.simulate_boin_desktop_scenarios(study, batches=2, trials=100, seed=2026)
Path("simulation.html").write_text(m.boin_desktop_simulation_html(batch))
Path("methods-en.md").write_text(m.boin_desktop_combination_protocol(study))
Path("methods-zh.md").write_text(
    m.boin_desktop_combination_protocol(study, language="zh"))
```

Simulation requires an empty combination/contour study. Every batch retains its
seed and full observations. Complete-request resource bounds are checked before
running. Safe profiles preserve actual counts and planned enrollment; explicit
`desktop_legacy` profiles expose original phantom enrollment, discarded outcomes,
unsafe selection and cross-replication state. Reports show actual and reported
patients/DLTs separately. Legacy nominal binomial errors are not validated
uncertainty estimates because replications can share defective state.

For single-agent/TITE final data, `set_boin_desktop_single_counts` edits the native
row-major count array; `boin_desktop_single_counts` reads it. Call
`select_boin_desktop_single(study, outcomes_complete=True)` only once all outcomes
are complete. The array itself cannot establish that fact. The shared selector
uses mathematical tie ordering rather than the native 1e-10 perturbation.

## What remains

Entry 99 remains **partial**. Current saved-project combination conduct, final
selection, simulation, scenario batches, bilingual methods and HTML reports are
integrated. Single/TITE saved final selection is integrated, but saved simulation,
comparison and TITE calendar/protocol adapters still need validation. In particular,
the desktop TITE simulator starts its first arrival at zero and checks pending
outcomes on additional random arrival increments, differing from the existing
Python calendar's conventions. Do not silently substitute one for the other.

Source compatibility, correct mathematical implementation and clinical validity
are separate claims. Native RNG/Iso solver identity, historical GUI appearance,
and exact proprietary template layout are not promised. The original program
is not redistributable and none of its executable/source/template assets are
bundled. See the [current evidence and remaining contracts](../research/boin-desktop-workflow-checkpoint-2026-10-12.md).
