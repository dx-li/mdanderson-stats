# PoP community workflow source crosswalk

The pinned PoPdesign 1.1.0 source is cached under
`research/raw/PoPdesign/source-1.1.0/PoPdesign/R/` in the authoritative
repository. `select.mtd.pop.R` constructs the weighted isotonic estimates,
applies the posterior safety screen, and returns the selected MTD and
estimates. `plot.pop.R` plots `p_est`, colors the selected estimate, and draws
the target line. It does not compute credible intervals; the contradictory
interval claim appears in help text only. The Python selector already provides
the estimate, selected original dose, and eligibility mask, so the new plot
composes that result without another statistical calculation. It preserves
original dose positions instead of reproducing the native compressed-index
plotting defect for untreated or excluded doses.

The existing `run_pop_protocol` accepts explicit scenarios but had no portable
input roundtrip. `PoPScenarioInput` adds a bounded versioned JSON form of the
design settings, scenario labels/truth vectors, simulation settings, and seed;
its `run()` method calls the existing report workflow. The file contains no
code and does not alter simulation or RNG semantics.

This closes the executable MTD display and reusable scenario-input workflows
for a community Python user. It does not implement flowchart image output,
native Word formatting, app-edited objects, or the undocumented `plus3` UI
hooks. It adds no credible intervals and makes no R seed-parity claim.
