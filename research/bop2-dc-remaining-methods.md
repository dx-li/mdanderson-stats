# Remaining BOP2-DC method coverage

This audit concerns catalog #156, not the separate BOP2 #112 application.
Binary efficacy has exact operating characteristics and finite-grid calibration;
paired endpoints have exact correlated-outcome operating characteristics and
finite-grid calibration; survival has calendar simulation and Monte Carlo
finite-grid calibration. The Normal endpoint now has posterior monitoring,
replay and simulation; its calibration is being implemented separately.

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
  Finite-grid calibration remains open.
- Section 2.4 (printed pages 12–13, text lines 466–495) specifies randomized
  comparisons. Fit arm models independently and compare
  `theta_experimental-theta_control` with both clinical thresholds. The source
  also defines optional interim superiority stopping. This requires posterior
  difference probabilities, arm allocation and trial conduct; independent
  single-arm decisions do not substitute. Section 3.2 gives simulation
  examples and an example 2:1 allocation, which is not a universal default.

These are source-backed methods, not report-format or UI differences. The
public catalog remains partial until coverage is established. The source's
effective-truth and clinical-utility guidance should be documented without
inventing additional restrictions on caller-declared futile scenarios.
