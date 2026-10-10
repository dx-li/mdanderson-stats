# Three-drug interaction-index simulation study

`simulate_interaction_index_three_drug_study` reproduces the source-defined
three-drug, single-combination-dose study (Lee and Kong, 2009, Section 4.1,
Scenario 1). It evaluates the existing observed-combination interval
calculations; it does not add a new estimator.

```python
from dataclasses import asdict
import json
from pathlib import Path

from mdanderson_stats import (
    simulate_interaction_index_three_drug_study,
)

study = simulate_interaction_index_three_drug_study(
    interaction_indices=(0.6, 1.0, 1.67),
    error_sd=(0.1, 0.4),
    replicates=20,
    rng=20261004,
)
json_text = json.dumps(asdict(study), indent=2, allow_nan=False)
Path("interaction-index-study.json").write_text(json_text, encoding="utf-8")
```

The source settings are three median-effect curves with slopes -1 and median
doses 1, 2 and 4. Each curve has six equally spaced single-agent doses from
**absolute dose 0.1** to three times that drug's median dose. The fixed
combination dose is `(1/3, 2/3, 4/3)`. For a true Loewe index `tau`, its true
combination effect is `tau / (1 + tau)`. Independent normal errors are added
on the logit-effect scale to all 18 single-agent observations and the one
combination observation, for 19 observations per simulated dataset. The
published settings use `tau=(0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.67, 2.5, 5)`,
error SDs `(0.1, 0.4)`, and 1,000 replicates per cell.

The printed candidate `1.67` is used literally. The paper's rounded effect
value `0.625` corresponds to the unrounded index `5/3`; pass `5/3` explicitly
if that is the value desired. No hidden substitution is made.

Each `InteractionIndexStudyCell` reports mean estimated index; coverage for the
raw-scale delta interval and log-scale delta interval; mean lengths of both
intervals on the original index scale; and the fractions of log intervals
entirely below one, containing one, or entirely above one. The raw interval
uses the direct delta-method standard error `estimate * log-scale SE` and may
have a negative lower endpoint. The log interval length is exponentiated back
to the original index scale before averaging, matching the paper's reported
`Len.ci.log` quantity. The Section 2 no-replicate fallback uses the package's
documented residual-degree-of-freedom-weighted residual variance convention;
the paper's word “average” does not settle that weighting choice.

The result records the actual seed, all scenario values, true coefficients,
dose grids, combination doses, replicate count and summaries. A supplied
nonnegative integer seed makes a run reproducible; when `rng=None`, the
generated uint64 seed is returned for replay. Interaction indices must be
unique values in `[1e-6, 1e6]`; error SDs must be unique values in `(0, 5]`,
and seeds must fit in uint64. The run is serial and streams one
19-observation dataset at a time. Candidate vectors are bounded and the total
number of scenario-replicate cells may not exceed 50,000. If a simulated
response rounds to exactly zero or one, or a fit or interval is not
representable, the run fails with the affected cell and replicate; responses
are never clipped, retried, or dropped. NumPy seed behavior is the community
API's reproducibility contract, not a claim of native random-stream identity.

The original archive was recovered on October 10, 2026. Its seventh true index
is `1/.6`; use `INTERACTION_INDEX_SOURCE_SCENARIOS` for the recovered values.
The existing printed `1.67` default remains unchanged. Set `retain_samples=True`
to retain bounded per-cell estimates for `.plot_qq(cell)` index/log-index panels,
and use `.write_json(path)` for captured settings, summaries and optional samples.

The separate [fixed-ray study](interaction-index-fixed-ray-study.md) now uses
the recovered ratio 2. The [workflow audit](../research/interaction-index-workflow-audit.md)
records original-function validation and the corrected native variance defect.
Python seeds reproduce Python streams, rather than the original S-Plus/R stream.
