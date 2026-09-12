# BF-BOIN

BF-BOIN adds backfill enrollment at lower doses to a BOIN dose-escalation
trial. The Python decision layer accepts separate counts for completed DLT
assessments and all assigned patients. Pending patients contribute to enrollment
caps, but must not be counted as completed non-DLT observations.

`BFBOINDesign.backfill_eligibility` evaluates activity, toxicity, assigned-count
caps and posterior exclusions, then identifies the highest available lower
dose. Pass a boolean response-observation vector for the individual doses;
the method propagates activity to higher doses. Re-evaluate eligibility after
new observations. Empirical backfill closures may reopen, while previously
eliminated doses must be carried forward explicitly.

`BFBOINDesign.next_dose` evaluates a completed escalation cohort. Supply a
`backfilled` boolean vector identifying doses that have received backfill so
that conflicting lower-dose data can affect movement. `n_stop` applies when
the resulting action stays at the current dose and its assigned count reaches
the threshold. It is separate from the per-dose backfill cap `n_cap`.

`BFBOINDesign.select_mtd` reuses BOIN's posterior safety screening and isotonic
estimation using the completed outcomes from both enrollment components.

The implementation follows the published method's activity and pooled-data
rules. The independently developed CRAN `bfboin` package differs on several
details and is not the MD Anderson app backend. See the
[reference audit](bf-boin-reference.md) and
[implementation notes](bf-boin-source.md) for those distinctions.

[Calendar-time simulation](bf-boin-simulation.md) supports delayed observations
and auditable patient histories. Post-escalation expansion, accelerated titration,
and generated app reports remain to be implemented. This decision layer does
not manage enrollment clocks or automatically determine cohort completion.
