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
No same-arm contradictory declaration is present. Python replay matches all
72 patient assignments/outcomes, 36 arm-look rows and 14 summaries. With
`absolute_tolerance=1e-11`, maximum allocation-probability disagreement was
2.73e-12 and monitoring-probability disagreement was 1.12e-16. The tighter
requested quadrature precision supports this comparison; the default 1e-9
tolerance gave an allocation difference of 2.7e-10 in the strong-signal case.

An additional two-arm control check used truth `(0,1)` and a balanced four-patient
burn-in. The experimental superiority probability was .95, and enrollment
stopped at four when that sole experimental arm closed; the control remained
excluded from efficacy and futility declarations. The eight-case check took
0.092 seconds after import, used 114.9 MiB peak process memory and recorded
zero swaps.

Simulation summaries must distinguish any efficacy declaration from a false
declaration; FWER requires an explicit definition of which arms are null.
The guide's average allocation curve assigns zero after an arm/trial stops.
The simulator reports both phase-specific rates and the union of early and
final declarations, so overall trial error is not obtained by adding the two
phase-specific familywise rates.

The native DBCD target was rechecked against guide page 6 and the saved app
HTML. The guide defines the transformation of the current desired allocation
estimate, but never specifies how that estimate is constructed. The saved app
exposes tuning parameters and allocation floors without a target-vector input.
The Python API therefore continues to require an explicit target; posterior-best
or Neyman targets must not be silently substituted as native behavior.

## Integrated validation

The four new focused tests plus the existing BARPO core checks passed (16 checks
in total). They cover aggregate replay/accounting, cumulative early-or-final
declarations, independent-seed rejection, complete-block DBCD initialization,
and tape-size rejection before conversion. Focused Ruff and root-checkout mypy
checks passed for the two new modules; the package's public exports were checked.

Both public guide examples executed successfully, including a 20-trial seeded
simulation. Additional integration assertions checked permanent arm-closure
positions, cumulative versus final-only rates with early monitoring disabled,
and allocation-probability conservation. This used 116.9 MiB peak process memory,
zero swaps and 0.225 seconds after import. Full-suite testing, packaging and
GitHub publication were not rerun for this checkpoint.
