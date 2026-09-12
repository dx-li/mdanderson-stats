# BF-BOIN implementation notes

`BFBOINDesign` implements the deterministic decision layer of the MD Anderson
Backfill Bayesian Optimal Interval design. It accepts evaluated patient and DLT
counts, plus `assigned` counts that include pending assignments. Backfill caps
are applied to `assigned`, so a dose at `n_cap` is closed even when its latest
outcomes are not yet evaluable.

Backfill eligibility is recalculated from the current evaluated data. A lower
dose is eligible when its activity is recorded, it has not reached the cap, and
it is not in the temporary empirical closure suffix. Closure requires both the
dose's own rate and its adjacent pooled rate to exceed the BOIN de-escalation
boundary. Activity/response must be recorded before a dose can be backfilled.
Empirical closure is temporary: if later evaluated data become safe, the dose
can reopen. Posterior
BOIN safety elimination is returned separately through `eliminated` and remains
the safety decision used by final MTD selection.

When a backfilled lower dose conflicts with the current action, BF-BOIN pools
counts from the highest conflicting backfilled lower dose through the current
dose. A pooled rate at or below the escalation boundary escalates; a pooled
rate above the de-escalation boundary searches cumulative lower pools and
de-escalates to the highest safe dose. The implementation uses the primary
paper's pooled conflict rules and does not include calendar-time simulation.

Final MTD selection delegates to the ordinary BOIN isotonic estimator and
posterior safety rule via `BFBOINDesign.select_mtd`.
