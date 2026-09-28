# BARPO completed-outcome trial workflow

The official guide, retained in ignored `research/raw/BARPO/BARPO.pdf`, describes
equal-randomization burn-in, adaptive allocation, scheduled monitoring, minimum
and maximum enrollment, and arm stopping. Its hash and formula references are
in [barpo-reference.md](../docs/barpo-reference.md). The guide does not describe
a delayed-outcome simulator; the saved app's delay inputs control animation.

Existing posterior, threshold and allocation kernels already cover the four
methods. The new trial workflow will reuse those kernels. Native details not
established by the inspected guide are equal-randomization block construction,
simultaneous stopping priorities, partial final cohorts and DBCD target
construction. DBCD must continue to take its target explicitly.

## Independent trial-path references

`tools/reference_barpo_trial.R` implements seven small experiments in base R,
using direct Beta-density integration for best-arm probabilities. These are
independent mathematical references under the following explicit Python
conventions, not captured native simulations:

- Two Beta(1,1) arms, four burn-in patients in a balanced block of four.
- An assignment uniform draws from the remaining block counts, making the
  block permutation replayable from the same tape as adaptive assignments.
- Adaptive probabilities are fixed for each two-patient cohort. All outcomes
  become available before its next update. Monitoring occurs at 4, 8 and 12.
- Early stopping begins at four. At the final look only final efficacy is
  assessed on eligible arms; prior declarations remain available separately.
- Under arm stopping, resolved arms close and the others continue; under
  trial stopping, any new terminal declaration ends accrual.
- BARCP uses tau=.7, BARN2N uses n/(2N), BARMTV uses posterior variance and
  assigned counts, and DBCD uses target=(.35,.65), tau=.7 and tau1=1.2.

The fixtures contain 72 patient/allocation rows, 36 arm-analysis rows and
14 arm summaries. Four experiments cover all allocation methods, one yields
a final efficacy declaration, and two use deterministic response truth (0,1)
to distinguish arm closure at n=8 from whole-trial termination at n=4.
No same-arm contradictory declaration is present. Python comparison is pending.

Simulation summaries must distinguish any efficacy declaration from a false
declaration; FWER requires an explicit definition of which arms are null.
The guide's average allocation curve assigns zero after an arm/trial stops.
