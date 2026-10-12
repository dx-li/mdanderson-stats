# Acceptance review corrections

Four review findings were verified and corrected without promoting another
catalog entry (97 implemented/35 partial/6 pending):

1. **ARAND integrated numerical safety.** Replay now retains posterior absolute
   quadrature error in `ArandNativeEvidence`. Positive-power allocation rejects
   unresolved eligible tails before transformation; run-in/equal allocation and
   single eligible arms do not transform those tails. The two-arm survival prior
   scales1e300/1e-300 reproduce the reported failure path. Regressions exercise
   powers.001,.5,1 and the valid power0 path. Original controller fixture services
   can still omit numerical errors; this preserves their explicit supplied-input
   compatibility scope.
2. **ToxFinder resource preflight.** The exact stage schedule now determines a
   lower bound before any simulation, including chain initialization and at least
   three objective calls per free-coordinate slice update. A20/40 cohort2 design
   at500/250/two-chain defaults needs at least28,622,124 counted evaluations for
   two trials; the20-million default is rejected promptly with explicit guidance.
   Larger budgets remain bounded at200 million total/20 million per fit; actual
   objective calls enforce the remaining budget during each fit. Sampling quality
   defaults are unchanged. Retained values retain their separate2-million cap,
   with explicit smaller-batch/distinct-seed guidance. Fifty60-patient trials
   would retain6.35 million values and fail before sampling. The representative
   two-trial run completed60 patients each using57,240,270 total counted evaluations.
3. **ToxFinder allocation MAP visibility.** Allocation, cohort and patient results
   preserve the original two-pass MAP fit. Reports display convergence, iteration
   counts and objective calls. A regression forces genuine one-iteration simplex
   exhaustion and verifies false flags survive both simulation and patient reports.
   The original optimizer result remains explicit compatibility behavior.
4. **CRM final-consensus uncertainty.** `BMACRMLookAhead.high_uncertainty` preserves
   any final completion's three-subject qualification. Recommendation and HTML
   retain the warning even when dose/action are invariant. The reported flat
   skeleton/one pending patient/final-cap case passes as a regression.

Independent numerical depth was also improved: the existing R posterior test is
explicitly described as a scalar alpha1 slice. A separate six-dimensional
randomized gamma-prior importance integral (eight scrambled32768-point Sobol
replicates) checks six log-parameter means and four risks against four slice
chains with2000 draws/1000 warmup. In the recorded moderate-data case, maximum
split R-hat is1.000367 and maximum difference is1.95225 combined estimated SE.
This is supporting evidence for that specified case, not universal mixing or
clinical validation. See the two `toxfinder-*-check-2026-10-12.json` records.

Current EffTox XML and named desktop selection were implemented separately, with
500 original-instruction admissibility/selection cases plus posterior integration
and saved-file tests. Current pending loop21 cases are source evidence with
explicitly supplied enumeration/inference services, not full conduct closure.
See `efftox-current-workflow-contracts-2026-10-12.md` for exact remaining contracts.

Full regression, fresh-wheel/extras/examples, reference regeneration and artifact
checks are being recorded in the subsequent validation checkpoint. No merge,
release, outreach, license split or settings/permission changes were performed.
