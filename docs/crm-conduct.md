# CRM look-ahead and calendar decisions

These functions connect the CRM, BMA-CRM and DA-CRM statistical kernels to
decisions with partially observed patient outcomes. They implement the
[CRM Suite guide's look-ahead rule](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/BMA-CRMSimulatorHelp.pdf)
and reconstruct the information visible at a chosen calendar time.

## Decisions before every outcome is known

```python
from mdanderson_stats import fit_bmacrm, bmacrm_lookahead

observed = fit_bmacrm(
    [.1, .25, .5], events=[0, 0, 0], subjects=[6, 0, 0], target=.3
)
look = bmacrm_lookahead(observed, pending=[1, 0, 0], current_dose=0)
print(look.action, look.dose, look.reason)
```

The input posterior contains **only fully observed counts**. `pending` supplies
additional patients at each dose. The calculation fits every relevant completion
and acts only if all completions produce the same action and dose. It does not
fill pending outcomes with their expected values or reuse updated model weights
as a new prior.

Patients at the same dose are exchangeable for these count-based decisions, so
the enumeration size is `product(pending + 1)`. Only two fits are needed to
establish disagreement when the all-DLT and no-DLT completions recommend different
actions. Agreement between these extremes is insufficient: all intermediate
count combinations must also agree. The returned `completion_events` and
`decisions` retain the evaluated evidence without retaining a full posterior
for every completion.

| Reason | Meaning |
| --- | --- |
| `complete` | No outcomes are pending; the existing complete-data decision is used. |
| `invariant` | Every completion was checked and gives the same action and dose. |
| `outcome_dependent` | Two evaluated completions give different actions or doses; wait. |
| `work_limit` | The completion count exceeds the requested bound; wait without claiming agreement. |

`max_completions` defaults to 128 and cannot exceed 1,024. Refits share an
explicit likelihood-evaluation budget. Numerical integration failures raise
errors rather than producing a partial decision. With no pending patients,
the helper performs no additional integrations. Dose indices are zero-based,
and starting, safety and final-selection settings follow `bmacrm_decision`.

## Reconstructing patient information at a chosen time

```python
import numpy as np
from mdanderson_stats import crm_calendar_snapshot, crm_calendar_decision

snapshot = crm_calendar_snapshot(
    doses=[0, 0, 1],
    enrollment_times=[0, 1, 2],
    dlt_delays=[np.inf, 1, np.inf],
    window=3,
    at=2.5,
    dose_count=3,
)
recommendation = crm_calendar_decision(snapshot, [.1, .25, .5], target=.3)
print(snapshot.outcomes, snapshot.times)
print(recommendation.routing, recommendation.decision.action)
```

This is a **retrospective or simulation replay interface**. `dlt_delays` contains
the eventual observed DLT delay after treatment, or positive infinity for a
patient confirmed to have no DLT throughout the assessment window. Infinity
must not represent an unknown outcome, loss to follow-up or an unassessed
patient. For live records with incomplete ascertainment, supply the actual
observed follow-up directly to the statistical kernels instead.

At time `at`, patients enrolled later are excluded, later DLTs remain pending,
and completed DLT-free windows become non-events. Events at time zero or exactly
at the window endpoint are included. `row_indices` maps the retained patient
rows to the original inputs. The snapshot contains immutable patient outcomes,
follow-up times and dose-level treated, observed, toxicity and pending counts.
It allows up to 200 patients and 20 doses, with consistent time units throughout.

`current_dose` defaults to the last retained patient's dose. An explicit override
can be supplied when the intended current cohort differs. `final=True` requests
final selection explicitly; the caller remains responsible for choosing the
decision time and cohort boundaries.

## DA-CRM and completed outcomes

```python
from mdanderson_stats import dacrm_trimester_prior

da = crm_calendar_decision(
    snapshot,
    [.1, .25, .5],
    target=.3,
    method="dacrm",
    da_prior=dacrm_trimester_prior(3, [.05, .15, .8], dispersion=2),
    minimum_observed=2,
    rng=np.random.default_rng(381),
    draws=1000,
    warmup=500,
)
print(da.routing, da.decision.action)
```

While outcomes are pending, this route uses DA-CRM posterior sampling and the
CRM Suite dose-decision policy. A single skeleton, a matching hazard-prior
window, explicit `minimum_observed` and a random generator are required.
The alpha prior comes from `da_prior`; separate model weights or `prior_sd`
would conflict with this single-model specification and are rejected.

When all outcomes are known, the router uses deterministic single-skeleton CRM
integration and its complete-data decision policy, consuming no random draws.
This includes an empty initial trial. Ordinary CRM and BMA-CRM instead use
bounded look-ahead when outcomes are pending. The result's `routing` field makes
the inference branch explicit, and `evaluations` includes the work within the
call. The returned `posterior` is the observed-count fit for a look-ahead call;
hypothetical completed posteriors are represented by their decision evidence.

## Coverage

These are independent Python implementations of the documented statistical and
conduct rules. They do not claim native executable parity, hidden sampler
settings or random-sequence parity. Calendar snapshots describe corrected
outcome histories, not the dates when a database received or corrected those
records. Full cohort scheduling, operating-characteristic simulation and native
files/reports remain separate work. Catalog entries 81 and 132 therefore remain
partial, and online entry 133 requires its own source audit.
