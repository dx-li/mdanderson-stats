# BOP2-DC randomized paired-endpoint exact OC audit

This tranche adds exact finite-grid calibration conditional on a caller-supplied fixed arm-assignment tape. It does not add a randomization law or an operating-characteristic average over allocation schedules.

Each truth supplies one four-category probability vector per arm in the order both endpoints, endpoint 1 only, endpoint 2 only, neither. The recursion increments the two endpoint-success counts jointly for the assigned arm, so the supplied within-arm association is retained. Posterior monitoring uses the shared paired randomized design's two independent arm-difference comparisons per endpoint, including its reported quadrature-error intervals and composite rules.

The optimizer evaluates the supplied Cartesian cutoff grid only. Scalar grid values broadcast to both endpoints; explicit two-column rows allow asymmetric endpoint values. Effective truth is checked against the source clinical-go composition; caller-declared futile truth is not forced to equal an LRV or lie below it. CGR counts interim graduation plus final go. FNGR counts interim no-go plus final no-go. FGR applies the same go events at the futile truth. Optional FCR is the larger final-consider probability across the two supplied truths. Exact feasibility and selection apply only to these supplied truths and this fixed tape; no composite-null guarantee, Monte Carlo error, or continuous-grid optimum is claimed.

For each look, posterior tail/error values are cached over endpoint success-count pairs and reused across candidate cutoffs and truth scenarios. The OC state is a four-dimensional lattice of paired binary margins. At each look, only nonzero-mass states are sent in bounded batches through the core's 16-corner decision rule; no full state-by-candidate decision tensor is constructed. Preflight accounts for state transitions, candidate classification, paired posterior-tail quadrature, retained candidate evidence, the old/new/multiply-scratch DP lattices, cached tails and bounded classification buffers. The exact lattice grows rapidly; requests above the documented limits fail before recursion rather than switching to simulation.

The reported exact work estimate is a conservative operation-count budget, not a runtime guarantee. Posterior comparison integration reports estimated quadrature errors rather than rigorous bounds; a candidate state whose decision changes over those error corners raises an arithmetic error rather than receiving a midpoint-only action.

Source boundary: BOP2-DC paper sections 2.1.4 and 2.2 define the paired Dirichlet endpoints and composite action rules; section 2.4 defines the arm-difference comparison. The implementation reuses those source-backed monitoring rules and adds only finite-grid recursion/calibration around a fixed allocation tape. Native randomization, native calibration search order, and other trial implementation details are not claimed.

## Independent numerical validation

The standalone base-R oracle uses an integer-shape finite-binomial-sum Beta
CDF, density integration and exhaustive four-category outcome tapes. Four
fixtures each enumerate 256 tapes for 16 cutoff candidates. They cover
multiple efficacy, efficacy/toxicity with reduced toxicity, matching marginal
truths with different association, and a zero-limit infeasible grid. All inputs
and generated reference tables are in
`tests/fixtures/bop2-dc-randomized-paired/`; they are mathematical examples,
not supplement settings or recommended clinical designs.

The Python checker verifies 388 case-specific posterior states, all 27,104
reached path/look decisions, 64 candidate metric rows, 256 per-scenario/candidate/
look OC rows, both objectives and infeasibility, and 24 representative public
replays. Public replays also match terminal enrollment and the reached look
schedule. The largest posterior difference is `6.2506744e-10`, within the
reported Python quadrature error plus `3e-12` reference allowance. Maximum
candidate-metric and OC differences are `1.3323e-15` (checks use `2e-12`).
The multiple-efficacy fixtures select candidate 0 for CGR and 8 for futile ESS;
efficacy/toxicity selects 9 and 8. Changing association while preserving
marginals changes decision probabilities, including a .0076 FNGR difference.

The amended R run took 22.159 seconds, with 214.75 MiB child-process high-water
RSS and zero swaps. The corrected Python comparison took 4.480 seconds,
154.70 MiB and zero swaps. Ten integrated focused core/OC/simulation tests
passed in 1.94 seconds, with 141.95 MiB and zero swaps. Numerical processes
ran serially with BLAS/OpenMP thread limits of one. No full repository suite
or new CI workflow was added.

The documented 40-patient, 100-trial simulation took 0.687 seconds including
imports/design construction, peaked at 103.47 MiB, and reported zero swaps.
It cached 130 marginal count pairs, returned expected enrollment 36.9, and
terminal probabilities `(graduate, stop_no_go, final_go, final_consider,
final_no_go) = (.13, .01, .41, .43, .02)`. Separate focused checks reproduce
12 nondegenerate seeded paths and an eight-trial degenerate truth with exact
early stopping. These are bounded numerical checks, not a precision study.

From the repository root, the reference can be regenerated and checked with:

```sh
Rscript tools/reference_bop2_dc_randomized_paired_calibration.R
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tools/check_bop2_dc_randomized_paired.py
```

Root adapted only reference-tool default paths and formatting after the worker
run, then reran the committed-form Python checker successfully against all
saved tables. Root review also checked endpoint-axis indexing, single-scenario
batch shape, absorbing lattice lifetime and the marginal-cache work bound.
