# Saved STPLAN community studies

Use a saved study to keep named scalar power calculations and bounded inverse
plans together, replay them from a portable JSON specification, and write the
results as JSON or standalone HTML. The workflow composes the existing
`STPLAN_METHODS` forward functions and `stplan_solve`; it adds no power formula.

```python
from pathlib import Path

from mdanderson_stats.stplan_study import (
    STPLANForwardCase,
    STPLANInverseCase,
    STPLANStudySpecification,
)

spec = STPLANStudySpecification(
    "phase II planning",
    (
        STPLANForwardCase(
            "current design", "stplan_normal_one_sample_power",
            {"difference": 0.5, "sd": 1.0, "sample_size": 30, "alpha": 0.05},
        ),
        STPLANInverseCase(
            "target design", "stplan_normal_one_sample_power", "sample_size",
            target_power=0.8, bounds=(2, 1000), parameters={"difference": 0.5, "sd": 1.0},
        ),
    ),
)
spec.write_json("phase-ii-study.json")
study = STPLANStudySpecification.from_json(
    Path("phase-ii-study.json").read_text(encoding="utf-8")
).run()
study.write_json("phase-ii-results.json")
study.write_html("phase-ii-report.html")
```

Forward cases require a scalar power result. Each inverse case names one
computable argument (or a supported tied-size pair), target power, explicit
search bounds and fixed forward inputs. Optional integer policy, index,
allocation weights, power tolerance and evaluation cap are recorded in the
specification. K-sample binomial inputs accept bounded probability and size
vectors; whole-total planning records the fractional group sizes returned by
the existing solver. It does not round them or invent an integer allocation.

Names and inputs are copied into immutable case specifications. Before any
calculation starts, the study checks every case's method, field names, scalar
and vector shapes, required fields, bounds, named-case uniqueness and aggregate
work budget. The saved JSON is capped at 1 MiB; a study holds 1–20 cases, each
group vector has 2–100 values, combined inverse limits are 20,000 evaluations,
and the conservative aggregate work ceiling is five million terms. The existing
forward methods and solver still validate scientific domains and solver-specific
policies during execution (for example, integer candidate ranges and the minimum
evaluation budget for continuous roots). An invalid input, unbracketed or
unattainable target, or numerical failure raises an error; a failed solve is not
retried or omitted.
Each inverse case defaults to a 1,000-evaluation cap; lower or raise that
explicitly while keeping the combined study under its limit.

The calculations are deterministic, so no random seed is required. Saved study
specifications capture caller-specified bounds and solver controls. Result JSON
records effective forward inputs, requested and achieved power, computed value
and bounds; HTML escapes names and values. Files are written atomically to an
existing parent directory.

This is a Python community format, not a native STPLAN report or session file.
Bounds and any selected branch remain caller decisions; the report does not
claim native automatic-bound defaults, native report-template parity, or an
integer allocation for proportional K-group totals. The separate
[critical-region significance planners](stplan-significance.md),
[historical-control planner](stplan-historical-planning.md), and
[matched-pairs initial-size estimates](stplan-matched-pairs.md) remain directly
available and are not silently converted into these case types.
