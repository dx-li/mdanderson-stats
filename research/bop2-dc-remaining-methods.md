# Remaining BOP2-DC method coverage

This audit concerns catalog #156, not the separate BOP2 #112 application.
Binary efficacy has exact operating characteristics and finite-grid calibration;
paired endpoints have exact correlated-outcome operating characteristics and
finite-grid calibration; survival has calendar simulation and Monte Carlo
finite-grid calibration. The Normal endpoint now has posterior monitoring,
replay, simulation and independently checked finite-grid calibration.
Randomized binary comparisons now include fixed-allocation monitoring, replay
and exact operating characteristics, with optional early graduation. Randomized
Normal and exponential-survival comparisons now have monitoring, replay and
bounded serial simulation with independent R evidence. All three single-endpoint
randomized models now have finite-grid calibration: exact conditional binary
recursion, and Normal/survival common-path Monte Carlo calibration with independent
holdouts. Their dedicated calibration audits record independent R evidence.

The cached primary preprint, `research/raw/BOP2-DC/paper.txt`, also establishes
these source-backed targets:

- Section 2.1.2 (printed page 5, text lines 166–202) specifies a single-arm
  Normal endpoint. For `Y ~ Normal(theta,sigma²)`, the prior is
  `theta|sigma² ~ Normal(theta0,sigma²/n0)` and
  `sigma² ~ InvGamma(a2,b2)`. Its conjugate update has precision `n0+n`,
  shape `a2+n/2`, location `(n0*theta0+n*ybar)/(n0+n)`, and scale
  `b2+SSE/2+n0*n*(ybar-theta0)²/(2*(n0+n))`. The marginal Student-t tail
  supplies the two posterior threshold probabilities. Monitoring, replay and
  simulation now have independent R evidence in `bop2-dc-normal-audit.md`.
  Finite-grid calibration and independent holdout checks are documented in
  `bop2-dc-normal-calibration-audit.md`.
- Section 2.4 (printed pages 12–13, text lines 466–495) specifies randomized
  comparisons. Fit arm models independently and compare
  `theta_experimental-theta_control` with both clinical thresholds. The source
  also defines optional interim superiority stopping. This requires posterior
  difference probabilities, arm allocation and trial conduct; independent
  single-arm decisions do not substitute. The binary-arm implementation is
  independently checked in `bop2-dc-randomized-binary-audit.md`; continuous and
  survival workflows are checked in `bop2-dc-randomized-normal-audit.md` and
  `bop2-dc-randomized-survival-audit.md`. Single-endpoint randomized calibration is now independently checked.
  Section 3.2 gives simulation
  examples and an example 2:1 allocation, which is not a universal default.

## Randomized multiple/co-primary endpoints

The paper also explicitly covers multiple/co-primary endpoints in randomized
trials (abstract lines 33–34 and introduction lines 89–97). Section 2.1.4
(lines 239–279) gives the joint Multinomial–Dirichlet endpoint model; §2.2
(lines 369–411) specifies OR/AND combination of endpoint decisions. Section
2.4 extends the previously described models to independent arms and posterior
experimental-minus-control comparisons. Section 3.2 (lines 663–667) explicitly
reports randomized multiple-endpoint and efficacy/toxicity simulations in
Supplement Tables S3–S4.

The randomized paired implementation now provides independent four-cell
Dirichlet arm models, marginal posterior differences, OR/AND endpoint decisions,
absorbing replay, exact joint-outcome operating characteristics and finite-grid
calibration. Bounded serial simulation supplies operating characteristics when
the four-dimensional exact state lattice exceeds its resource limits. See
the [randomized paired guide](../docs/bop2-dc-randomized-paired.md) and
[source mapping](bop2-dc-randomized-paired-source.md).

Two binary indicators can also represent a larger categorical outcome after
exactly aggregating counts, prior shapes and truth probabilities into their
four indicator combinations. More than two decision endpoints, nonbinary
utility-weighted posterior criteria, delayed paired observations and calibration
beyond the bounded exact recursion remain separate scope.
The cached main paper does not contain the full supplement scenarios, so
table-level parity must not be claimed without obtaining those settings.
The optional paired graduation rule is a documented composition of the scalar
randomized superiority rule with the endpoint OR/AND rule, not independently
verified native paired-graduation pseudocode. These qualifications and native
UI, report and RNG parity keep the public catalog partial. Effective truths
must satisfy the clinical-go composition; no additional LRV restriction is
invented for caller-declared futile scenarios.
