# MTADF author isotonic simulation audit

The cached rendered author program is
`research/raw/mtadf-author-reference/targetAgentDF.r`. In `isotonic()`
(lines 31–48), the program solves
`pbeta(0.3, alpha, 0.5-alpha) = 0.22`, uses the resulting fixed
`Beta(alpha, 0.5-alpha)` prior, pools posterior overdose probabilities with
increasing PAVA, and defines the admissible-dose count as at least one.
Consequently its operating-characteristic procedure never stops enrollment
for safety.

Lines 54–82 start each trial at dose 1 and add complete binomial cohorts.
After a cohort, the next-dose branch uses `adm` carried from before that
cohort, then updates `adm` from the new toxicity counts. The replay records
both caps, so the one-cohort lag is visible and independently testable. The
separate `df.isotonic()` function at lines 114–152 starts from a fresh cap;
these are distinct author procedures and the simulation does not silently
replace its lagged rule with the fresh rule.

The author simulator uses unit-weight `ufit` calls on the observed prefix to
select subsequent cohorts (lines 72–79), and a full-dose fit to select the
rightmost fitted maximum after dividing efficacy counts by `n+0.0001`
(lines 83–86). It then caps that index by the fresh admissible count. This can
select an untried dose on a final efficacy tie. The Python result preserves
this behavior and provides cohort allocation and pre/post-cap ledgers.

Independent inspection of the cached Iso 0.0-15 implementation shows that its
unconstrained midpoint-mode branch is equivalent to independently fitting
each split by increasing/decreasing PAVA for prefixes of at least two points.
Its one-point call is undefined: the candidate loop is skipped and the later
indexing uses invalid split positions. Python assigns that first-prefix case
the identity fit to make the replay operational; this is not native parity.

The Python simulator draws independent toxicity and efficacy potential
binomial counts for every cohort-dose cell and observes only the assigned
dose. This has the intended independent dose-specific margins and enables
replay through the same conduct routine, but consumes NumPy randomness in a
different order from the author's sequential R draws. No random-stream or
printed-table parity is claimed. The existing paper-policy `mtadf_decision`
and `simulate_mtadf` remain unchanged.

Focused checks cover the lagged-cap distinction, cohort-count conservation,
the rightmost all-dose tie, and repeatability under a fixed NumPy seed. They
do not validate every operating-characteristic scenario or establish
clinical utility. The raw author R source and independent reference fixture
The raw author R file and Iso implementation remain in the local evidence
cache, outside the package distribution; the generated reference routine and
small fixture are maintained separately from this module.
