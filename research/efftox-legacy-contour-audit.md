# Legacy EffTox contour contract

[Thall and Cook (2004), pp. 686–687](https://www.johndcook.com/efftox.pdf)
define a monotone target curve `t = a + b/e + c/e**2`. Their desirability is
`rho(p)/rho(q) - 1`, where `p` intersects this curve on the line from the ideal
point `(1,0)` through the evaluated pair `q`. Thus, given `q=(e,t)`, the
intersection is `p=(1-z*(1-e), z*t)` and utility is `z-1`. The paper's displayed
homotopy maps `p` to `q`; reversing that mapping without inverting its ratio
would reverse the intended utility. Utility is zero on the target and tends
to positive infinity at the ideal point.

The [2006 report, pp. 2 and 6](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/EffTox/NewTradeOffFunctions.pdf)
records the switch from inverse-quadratic contours through version 2.8 to Lp
contours in version 2.9. It also says the old fit minimized error and could
miss the three elicited points. Its objective and constraints are unavailable;
exact interpolation for compatible monotone points must not be called native
optimizer parity. The report's informal translation description also differs
from the explicit radial construction in the original paper.

`tools/reference_efftox_legacy.R` independently solves interpolation coefficients
and the ray intersection with base R. It covers the published Pentostatin
targets, lower-degree cases, a zero derivative at the lower endpoint, probability
boundaries and radial transformations with known utility. This is a mathematical
reference for the Python implementation, not a native-kernel comparison.

## Integration validation

The public `EffToxLegacyContour` matches all 60 base-R rows with maximum
absolute utility error `4.27e-14`. Five contour checks and three simulation
checks passed together in 1.35 seconds, including the reproduced tiny-toxicity
endpoint failure and scalar-shape preservation. Targeted lint, formatting and
type checks passed. The public guide example also ran successfully.

A 100-by-100 probability grid had finite scores increasing with efficacy and
decreasing with toxicity. Vectorized scoring took 0.0071 seconds; the combined
reference/example/grid process peaked at 112.5 MiB RSS with no process swaps.
These timings describe the measured small workload, not all possible inputs.
