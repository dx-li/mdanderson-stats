# BOIN desktop workflow checkpoint

Catalog99 remains partial; counts remain **98 implemented /34 partial /6 pending**.
This checkpoint integrates combination conduct, final selection, full simulation,
scenario batches, bilingual methods, escaped/reopenable reports, and single/TITE
saved final counts. It does not close the full desktop application.

## Evidence added after b22f8b8

- 144 executions of the active combination **conduct** final selector, with
  weighted means and native displayed marginal-limit transformations. A six-cell
  exhaustive isotonic oracle substitutes for original Iso::biviso; native Iso
  floating-point identity is not claimed. A 1e-10 score tie tolerance stabilizes
  near-pooled ties after the dose perturbation. Legacy single-MTD selection can
  choose an eliminated cell; safe selection excludes it and reports the difference.
- All48 original complete simulation wrapper cases now match Python, plus192
  extended executions covering extra-safe and custom upper bounds (including
  the48 baseline configurations). Outcome and tie services are deterministic
  host services, not native RNG claims. Single-MTD phantom top-up, endpoint jump
  and cross-trial mutable state are explicit `desktop_legacy` behavior. Waterfall
  can discard newly observed failed-subtrial counts; both actual and reported
  patients **and toxicities** are retained separately. Safe simulation retains
  observations, honors enrollment limits and excludes unsafe final selections.
- 240 additional original contour executions cover adverse edited histories.
  Legacy behavior can assign a globally eliminated cell. Safe conduct stops for
  safety instead of issuing that dose. Existing576 managed-controller cases and
  1728 GUI-valid R cases per conduct mode continue to pass.
- 960 original single-agent/TITE final-selection cases agree with the shared
  mathematical selector. Native TrialData is a flattened two-row integer array:
  all patient counts then all DLT counts. Reader/editor preserve this convention.
  Final selection requires explicit complete-outcome confirmation; the saved
  count array itself cannot establish that all outcomes are observed. Native
  1e-10 perturbation and exact near-tie identity are separately qualified.
- Multi-scenario batches preflight complete request limits: at most100 runs,
  two million retained trial-dose cells and one million planned patient draws.
  Distinct uint64 seeds and full arrays permit replay. Aggregates pool actual
  trial counts. Legacy nominal binomial error does not account for defective
  cross-replication dependence and reports explicitly reject it as validated
  uncertainty. These are implementation/Monte Carlo claims, not clinical validity.
- Independent English/Chinese combination methods include effective settings,
  conduct flow and integer tables. They distinguish patient conduct from the
  simulation titration prelude and waterfall's n=6 simulation-only override.
  Reports escape labels and retain reopenable XML; inconsistent saved inputs
  are rejected at report pairing. No proprietary protocol/template is bundled.

Focused BOIN tests: **5,742 passed**. A fresh external wheel environment with
plot/image extras passed the same **5,742 tests in25.80seconds**, plus all four
original saved defaults and **seven independent-review regressions**. **41,052 full-suite tests passed in877.50seconds**, with warnings treated as errors.
The [validation manifest](boin-desktop-workflow-validation-2026-10-12.json) records
source-tree identity and checkpoint scope. Final artifact hashes are in the
preservation commit message. The first warnings-as-errors run exposed unclosed CSV
fixture handles; fixed loaders close reads before parsing. That interrupted run
is not acceptance evidence. Production code was unchanged by the loader fix.

## Recovered next contract: desktop TITE calendar

256 base and2048 extended original executions are now preserved in `fixtures/boin-desktop/tite-calendar.csv`.
The extended cases additionally cover cohort sizes1/3, pending fractions.35/.65 and uniform/late trimester priors. The hash-pinned driver runs original get.oc.tite.R and its final selector, while
supplying explicit constant arrival gaps and synthetic outcome/delay vectors.
End-state instrumentation only observes local arrays; no numerical source
expression is rewritten. These are reference fixtures, **not an implemented
Python calendar equivalence claim**.

Native first enrollment is at zero; other cohort members enter after random
arrival gaps. After each cohort, decision time advances by another random gap;
suspension advances on more gaps, rather than jumping to an outcome ascertainment.
Final cohorts wait a full assessment window. Stored assessment units convert
using30.436666666666667days/month. Pending threshold is MaxPendingOutcomePercent/100;
analysis trimester weights are distinct from the simulated event distribution.
The native scheduler also uses future generated outcomes in one pre-observation
lowest-dose stopping branch and can leave its stop code unchanged. That defect
must be explicit legacy behavior, never a silent scientific default.

## Precise remaining contracts

1. Integrate saved single-agent simulation and native3+3 comparison choices,
   including titration-cap meaning, convergence stop and extra-safe gating.
2. Implement and validate the separate desktop TITE calendar/profile against
   the2304 traces, then connect saved assessment units, timing distributions,
   pending gates, STFT calculator, Rolling Six comparator, all-scenario batches
   and bilingual TITE methods/reports. Existing web129 calendar conventions are
   not automatically desktop99 equivalents; its separately unidentified regret
   metric remains a separate unresolved contract.
3. Broaden single-MTD edited-history references and document any deliberate
   safe-vs-native exclusions. Exact RNG, Iso floating identity and native GUI
   layout are distinct compatibility claims, not inferred from fixture matches.
4. Run final full-code/wheel/examples/reference/provenance acceptance after those
   workflows close; only then reconsider catalog99 status.

The four review fixes remain intact: ARAND unresolved-tail rejection with absolute
quadrature error, ToxFinder exact resource preflight, CRM final pending-consensus
uncertainty, and ToxFinder MAP diagnostic propagation. Seven installed regression
instances passed here; the full acceptance manifest keeps their evidence explicit.
Original BOIN assets are ignored and excluded from wheel/sdist; the original
program's redistribution prohibition is retained in THIRD_PARTY_NOTICES.md.
The package remains mixed-license. No status promotion, license split, PR bypass,
merge, release, deployment, author contact or settings change occurred.

Additional next-stage fixtures now include192 complete single-agent simulator
executions and96 Rolling Six calendar executions. Native single-agent titration
stays and tops up at min(cap+1,number of doses), unlike the shared simulator's
cap convention. Native Rolling Six may report max(entry)+window earlier than
its actual final decision clock; compatibility and actual time must be separate.
The managed UseDefaultBounds flag also suppresses explicit saved numeric bounds
when true. The current adapter still needs that flag mapping; default files have
matching stored values, so the existing default-example checks do not cover
contradictory inactive saved bounds. This is a precise next-stage correction,
not a claim that the complete desktop contract is accepted.

The original managed STFT calculator was executed at RVA0xd108 for84 synthetic
cases; independent Python weights differ by at most2.23e-16. Its saved follow-up
values are always **days**, even when the displayed assessment window is months.
Eight R reference files regenerate byte-identically.
