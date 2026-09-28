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
reference for implementation in progress, not a completion claim.
