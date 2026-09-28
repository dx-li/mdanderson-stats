# BARD two-stage continuation audit

The cached primary paper (`research/raw/BARD/paper.txt`, pages 11–12) defines
the target total across the two selected doses to include eligible stage-one
patients. Additional enrollment is the target less both carryover counts.
Different stage-one/stage-two eligibility criteria require subsetting; the
exact eligibility criteria themselves are not specified.

The paper selects two doses using totality of evidence (safety, efficacy,
PK/PD and tolerability), saying the higher is often the MTD. An adjacent pair
is not mandatory. The new `continue_bard_trial` therefore takes an explicit
ordered pair. It uses the existing BF-BLRM completed patient ledger, caller
eligibility and factor rows, the existing minimization allocation and final
OBD helper. A missing MTD or permanent all-overdose result cannot be reopened
by this continuation. This transition gate is an explicit Python contract.

The source simulation does not include interim stage-two safety/futility
monitoring. Complete potential outcomes are therefore a sufficient input for
the allocation/final-selection workflow. No stage-two delay law is invented.
The official app asks for a number per dose arm but does not establish hard
quota enforcement; the implementation follows the paper's total-target formula.
The already documented prior/weight/tie and noninferiority-sign choices in
[`docs/bard.md`](../docs/bard.md) continue to apply.

`tools/reference_bard_integrated.R` independently carries forward the earlier
validated stage-one R ledger, updates covariate imbalance and joint outcomes,
then calculates Dirichlet mean utility and Beta-tail screens. Allocation and
tie probabilities equal one in these references to isolate bookkeeping from
random-stream differences. Four cases check:

- Seven eligible carryover patients plus three new assignments reach target 10.
- Selecting doses two/three excludes other-dose patients and retains six
  carryover patients, then adds three new assignments.
- A short candidate tape enrolls two of four required new patients and returns
  no final OBD.
- Seven carryover patients already meet the target; no candidate is randomized.

The integrated Python replay uses the actual stage-one model/calendar driver,
not a fabricated result object. All eight new patient records, 32 arm/category
count rows, six posterior rows and four enrollment/selection summaries agree
with the independent R results. Maximum posterior/utility error is `4.27e-14`.
The root comparison takes 0.0184 seconds after imports, at 119.63 MiB peak RSS
with zero reported swaps. The R generator takes 0.143 seconds with warnings
treated as errors.

Four focused worker tests pass in 1.63 seconds, with targeted Ruff/format and
mypy checks passing. Input dimensions, categorical representability, outcome
types, inclusive target and work estimates are checked before allocation
randomness. The result retains inclusion/exclusion identities, separate stage
counts, assignment seeds and target shortfalls. No CI expansion, native app
execution or large operating-characteristic run was performed. BARD remains
partial while titration, expansion, stage-two timing and native parity are open.
