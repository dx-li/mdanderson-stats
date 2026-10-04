# Two-arm binary automatic success-cutoff calibration

## Source contract

The cached primary paper is `research/raw/BayesianCalibration/paper.txt`.
Supplement §S2.2.2, printed pages 48–50, defines independent treatment and
control binomial counts, independent beta design and analysis priors, the
treatment-minus-control risk difference, and the effectiveness region
`theta_T > theta_C` (generalized in the existing Python API to a supplied risk
difference margin and direction). It defines the posterior decision
probability by numerical integration, success by the strict inequality
`p_a(x_T,x_C) > c`, the product beta-binomial predictive mass, the design-prior
posterior effective probability `q_d`, and the operating-characteristic sums.
In particular, PID is the ineffective-truth mass among successful trials
divided by Bayesian power.

The cached application help is `research/raw/BayesianCalibration/calibration-help.txt`,
SHA-256 `7d2c730bcbcd6b9957308aac0f169e86208b6ab4059d5045273e51998954c8b5`.
Lines 1–14 advertise prespecified cutoffs and calibration to target PID, with an
optional range defaulting to `[0.6, 0.999]`. The help does not specify its search
algorithm, tie handling, or numerical policy. The Python search therefore does
not claim native optimizer behavior.

## Bounded search convention

`prepare_binary_two_arm_success` already enumerates both response counts and
retains posterior probabilities, their analysis quadrature errors, effective
and ineffective design-prior masses, and frequentist null masses. It bounds
each arm to 1,000 patients and the joint count pairs to 40,000. The new search
validates the requested model and resource limits before preparing the table,
then sorts the flattened states once and computes compensated suffix sums. It
checks every distinguishable threshold state; it does not assume PID is
monotone when the analysis and design priors differ. A final chosen threshold
is evaluated again by `BinarySuccessTable.evaluate` for the complete returned
record.

The decision rule is strict, so a zero-error state with posterior probability
equal to the cutoff is excluded. For positive posterior integration error, the
existing evaluator rejects cutoffs in the closed interval
`[p_a - error, p_a + error]`. The search widens each interval outward by one
representable step before merging, then considers cutoffs outside the
conservative intervals. It checks every such state and selects the smallest
feasible cutoff among these conservative candidates; it does not claim every
guard-safe floating-point value was searched. It preserves the evaluator's
special `c=0` rule. When an uncertainty interval obscures a candidate state,
the search raises an unresolved-calibration error instead of claiming that no
mathematical feasible cutoff exists.

The table stores integration errors for analysis posterior probabilities but
not errors for the design-prior truth probabilities used in PID. The selected
target comparison therefore applies to the computed PID and is not a rigorous
upper bound on the exact PID. Existing limitations for rare float64
probabilities remain.
