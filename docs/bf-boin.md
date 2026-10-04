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

`BFBOINDesign(stay_at_one_of_three=True)` enables the guide's optional 1-DLT-of-3
rule for targets from 0.20 through 0.279: at exactly three patients, 0/3
escalates, 1/3 stays, and at least 2/3 de-escalates. Python applies this
modification to individual dose actions before resolving backfill conflicts
with the usual pooled-rate rule.
`BFBOINDesign(deescalate_at_two_of_six=True)` enables the separate guide option
for targets from 0.28 through 0.33: at exactly six patients, at most 1/6
escalates and at least 2/6 de-escalates. The latter is applied as an explicit
six-patient action override; the ordinary BOIN design remains unchanged. Both
flags can be recorded in the protocol report. If both are enabled, their target
ranges do not overlap, so construction rejects the incompatible combination.
The guide does not explicitly specify this interaction, so the ordering is an
explicit Python policy. Empirical backfill closure still uses the raw observed
and adjacent pooled rates; safety exclusions take precedence.

Ordinary posterior safety elimination starts at three evaluated patients.
BF-BOIN's optional `extra_safe=True` rule requires **more than three** evaluated
patients at dose 1 and a posterior tail strictly above the offset cutoff, as
specified by its guide. `boundary_table()` reports the effective cutoffs,
including ordinary elimination at three patients. Previously the BF wrapper
inherited ordinary BOIN's extra-safe count threshold of at least three; that
boundary behavior is corrected for BF-BOIN only.

`BFBOINDesign.select_mtd` uses BOIN's weighted isotonic estimation with the
completed outcomes from both enrollment components and BF-specific safety
exclusions. With `bound_mtd=True`, admissible fitted estimates must be strictly
below the de-escalation boundary. This corrects the prior inherited inclusive
comparison; the ordinary BOIN API retains its existing convention.

The implementation follows the published method's activity and pooled-data
rules. The independently developed CRAN `bfboin` package differs on several
details and is not the MD Anderson app backend. See the
[reference audit](bf-boin-reference.md) and
[implementation notes](bf-boin-source.md) for those distinctions.

[Calendar-time simulation](bf-boin-simulation.md) supports delayed observations
and auditable patient histories. Its opt-in [post-escalation expansion](bard-expansion.md)
implements the fixed `c - 1` continuation documented for BARD's BF-BOIN path.
Optional [accelerated titration](bf-boin-titration.md) follows BF-BOIN Guide
Remarks 2 and is available in the calendar simulator. [Saved protocol reports](bf-boin-protocol-report.md)
capture the actual design, calendar choices, scenario truths and compact
simulation summaries. Native document templates and unspecified native summary
formulas remain separate scope. This decision layer does not manage enrollment
clocks or automatically determine cohort completion.
