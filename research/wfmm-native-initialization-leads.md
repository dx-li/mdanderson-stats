# WFMM native initialization leads — October 9, 2026

This is source triage for the remaining WFMM initializer/proposal contract;
it adds no production estimator and does not change entry #70's partial status.
The existing Python initializer explicitly uses REML. The recovered native
3.1 guide labels `omega_MLE` as **profile maximum likelihood**, so native parity
must not be inferred by relabeling the Python REML estimator.

The checksum-verified Linux `wfmm1` executable has SHA-256
`0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e`;
archive provenance and regeneration are in the
[variance-prior audit](wfmm-native-prior-audit.md). Symbol lookup and original
initialization disassembly identify these concrete locations:

| Virtual address | Symbol / inspected connection |
| --- | --- |
| `0x46b110` | `WFMM::WFMM_Init`; calls `ANOVA_MOM` at instruction `0x46b233` |
| `0x4870e0` | `WFMM::ANOVA_MOM`, iterative MOM initialization |
| `0x48ab80` | `WFMM::Bisect`, optimizer control; calls `loglik1` within its body |
| `0x49d960` | `WFMM::loglik1`, native likelihood evaluator |

The guide's initialization output table distinguishes `omega_MOM`,
`omega_MLE` and `se_omega`; the last supplies automatic variance-component
proposal calibration. Its input table gives `propvar_omega=1.5`,
`omega_MOM_maxiter=100`, `omega_MOM_convcrit=1e-3`, and native
`minVC`/`VC0_thresh` floors of `1e-6`. These are documented native inputs,
not evidence that the full estimator/proposal formula has been reproduced.

For the eight-curve, intercept-only, identity-transform synthetic input in
`tools/reference_wfmm_variance_prior.py`, a retained original `_Init.mat` probe
has MOM residual values exactly equal to the independently calculated RSS/N:

| Coefficient | Native MOM = RSS/8 | RSS/7 (REML) | Native profile initialization output |
| --- | ---: | ---: | ---: |
| 1 | 2.484375 | 2.8392857142857144 | 2.494323656093732 |
| 2 | 2.4375 | 2.7857142857142856 | 2.446731009336839 |
| 3 | 3.9375 | 4.5 | 3.9497915978004206 |
| 4 | 2.4375 | 2.7857142857142856 | 2.446731009336839 |

The profile outputs agree with the already committed original-program
`identity` case in `tests/fixtures/wfmm-native-variance-prior.json`, whose
SHA-256 is `458efd0457226f65ec54d4826356145cb66be39a4759f00753c0aa543685c48b`.
The MOM values come from the retained native probe, not a new executable run.
The discrepancy between profile output and the exact intercept-only RSS/N
optimum is unresolved numerical/optimizer behavior, not a new mathematical
MLE formula. The single-case MOM equality does not establish the mixed-design
iteration, residual-stratum estimator, flooring policy or proposal calibration.

Next, extend the checksum-verified native reference harness to retain MOM and
proposal-calibration outputs for diverse synthetic designs and trace the
likelihood/optimization control. Compare profile ML with an independent direct
likelihood before deciding which source behaviors need reproducing or correction.
Keep variance-prior shape/scale mapping, empirical-Bayes shrinkage, compressed
transform layouts and general pipeline files as separate contracts. Original
binaries/disassembly and synthetic MATLAB probe files remain ignored local
research inputs; no original program is redistributed.
