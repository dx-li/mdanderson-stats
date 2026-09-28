# BaCIS input and plotting source contracts

Inspected September 28, 2026. The official app links
[BaCIS_InputTemplate.zip](https://biostatistics.mdanderson.org/shinyapps/BaCIS/BaCIS_InputTemplate.zip).
The archived 714-byte ZIP has SHA-256
`a04609359259c8ea603764d95cba5a73b04e4f5e4b7198e162c3fc0a6a5245e6`
and contains one 1,377-byte CSV, `BaCIS_InputTemplate.csv`.
Only in-memory ZIP inspection was needed. No source application was executed.

## Input template

The CSV is a labeled-row configuration with a versioned title, blank second
row and trial-name field. It contains subgroup count, response/sample-size
vectors, manual/automatic switches for tau1 and the classification cutoff,
all classification/borrowing priors, efficacy cutoff, native MCMC iterations
and seed, followed by graph settings. Automatic tau1/cutoff values are literal
`NA`. The template's title version is 1.0.0.0; the inspected app reports
1.0.1.0, so these versions should not be silently conflated.

The supplied example uses responses [2,3,1,7,8], sample sizes [25]*5, automatic
tau1/cutoff, tau2=.001, low/high centers .1/.3, tau4=.1, gamma shape 50 and
**rate 10**, efficacy cutoff .92, native MCMC count 1000 and seed 1234.
Graph settings include automatic generation, one color per subgroup, two
cluster colors, a prior color and maximum y=22.

CSV input/output can preserve these fields independently of executing a fit.
Native MCMC iteration count is not automatically equivalent to Python retained
draws per chain plus warmup; importing a file must not silently reinterpret it.
R color names such as `green3` and `cyan2` also need explicit translation before
use in Matplotlib. The source CSV/parser backend is unavailable, so tolerance
for reordered labels, unknown rows or malformed input is not established.
No parser or native output-file equivalence is claimed by this audit alone.

## Classification density plot

Pinned bacistool 1.0.0 `bacisPlotClassification.R` reruns first-stage inference
and overlays R kernel-density estimates of the latent theta samples. It uses
the range [-100,100], a fixed maximum density .04, Arm labels and a five-color
palette. The mathematical posterior is instead available exactly as two
weighted half-normal densities, as recorded in the existing theta audit.

An analytical plot can use the exact density and avoid MCMC/KDE noise. Its
two sides can have different limits at zero, so they must be separate line
segments. A fixed y=.04 can clip a valid high-precision density; autoscaling
the density axis is an intentional presentation correction. This is the
native classification-plot purpose with exact curves, not reproduction of an
individual random KDE or the app's other response/prior plots.
