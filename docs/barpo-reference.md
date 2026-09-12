# BARPO references and validation

Primary software: [Bayesian Adaptive Randomization with Posterior Probability](https://biostatistics.mdanderson.org/shinyapps/BARPO/),
J. Jack Lee, Ying-Wei Kuo and Nan Chen. The retrieved app identifies itself as
PID 948, version 2.0.3.0, updated January 6, 2026.

The [official support guide](https://biostatistics.mdanderson.org/shinyapps/BARPO/BARPO.pdf)
describes conjugate binary posteriors, monitoring with or without a control,
and four randomization methods. Pages 5 and 6 were inspected visually because
text extraction obscures the square root and exponents in the formulas.
The guide was retrieved September 12, 2026; its SHA-256 is
`b28b07358de80e575c6dce3db265465ba04db4db81c33b803ad750cf32fc1232`.
The original PDF and HTML are retained in ignored `research/raw/BARPO` and
are not redistributed with the package.

`tools/reference_barpo.R` independently integrates three Beta densities in
base R to calculate best-arm probabilities and comparisons against control.
`tests/fixtures/barpo-posterior.csv` contains those results and conjugate
posterior variances and threshold probabilities. All three priors are
Beta(1,1), with response counts `(1,4,8)`, nonresponse counts `(3,2,1)` and
assigned counts `(5,8,12)`.

`tests/fixtures/barpo-allocation.csv` contains twelve allocation scenarios:
four methods, each with no floor, one control floor and simultaneous arm
floors. The maximum enrollment is 50. BARCP uses power 0.7, BARN2N uses
25/(2*50), BARMTV uses posterior variance divided by assigned count plus one,
and DBCD uses explicit target `(0.2,0.3,0.5)` with powers 2 and 0.5.

These are independent mathematical references, not captured app outputs.
The guide supplies a proportional rescaling rule for a control floor.
Proportional redistribution with simultaneous lower bounds is an explicit
Python policy; exact app parity for competing floors is unverified. DBCD's
desired target is supplied explicitly because its construction is not fully
specified in the guide.
