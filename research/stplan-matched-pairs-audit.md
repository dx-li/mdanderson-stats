# STPLAN matched-pairs procedure audit

The archived `ampd` procedure is a substantive paired-binary design missing
from the current Python methods. It is separate from the active matched
case-control method: the latter samples a case/control pair and conditions on
discordant exposure, whereas `ampd` compares paired all-or-none responses and
uses the pilot 2×2 outcome table to estimate its variance.

The cached STPLAN 4.5 archive is identified in `docs/stplan-sources.json` as
`DSTPLAN_V4.5v.zip` (SHA-256
`05cb3eb2e2aef84958205f6c0748b9310ea93648854766c3e77261c67877b648`). The
relevant source is `SOURCE/ampd.f`, `qmpd.f`, `pmpd.f`, `psimpd.f`, and
`fmpd.f`. `SOURCE/stplan.f90` comments out its `CALL ampd` main-menu entry;
the archive describes it as an older method but does not state why it is
disabled. The equations cite Olli S. Miettinen (1968), “The Matched Pair
Design in the Case of All-or-none Responses,” *Biometrics* 24, 337–352.

`PSIMPD` with known preliminary counts defines `N=Z11+Z10+Z01+Z00`,
`a=Z10+Z01`, `b=Z10-Z01`, `r=(a+delta*b)/(2N)`, and
`psi=r+sqrt(r^2-delta*(b-delta*(Z11+Z00))/N)`. `PMPD` uses

```text
D² = psi² - delta²*(3+psi)/4
z = (-Phi^-1(1-alpha/sides)*psi + |delta|*sqrt(n*psi))/D
power = Phi(z)
```

The Python API adopts the source's one-sided behavior and source's two-sided
alpha/2 with power only in the direction of truth. Unlike the routine's
interactive numeric range checks, Python validates the approximation's
positive variance geometry and rejects nonrepresentable calculations. It
uses a scale-normalized expression for `psi` and the equivalent
`D/psi=sqrt(1-(delta/psi)^2*(3+psi)/4)` to avoid overflow in squared counts.
The source's pilot inverse permits total counts and sample size from 1 through
`1e10` inclusive, and delta, significance, and power from `1e-8` through
`0.99999999`; its one-sided significance is halved for a two-sided test. Python
uses the common STPLAN API domain `alpha<0.5`, supports arrays of up to 200,000
cases, and accepts larger finite table weights where normalized arithmetic is
representable. It explicitly fails when `alpha/sides` underflows to zero.

`QMPD` solves delta, n, alpha, or power with the other three fixed. Its
sample-size formula squares a signed expression; Python's no-pilot initial
size wrapper checks that the positive branch is valid and verifies the
forward power after solving. `stplan_solve` supplies the existing bounded
inverse mechanism for pilot-table difference, sample size, and alpha, and
holds the four pilot counts fixed. It deliberately does not add a second
root-finding framework. Continuous sample size is returned by default;
integer attainment is an explicit option in the existing inverse API.

For source mode 5, with no paired table but estimated group response
proportions `theta1`, `theta2`, `PSIMPD` uses
`psi=theta1+theta2-2*theta1*theta2` and `QMPD` recommends a preliminary size
of `n/4`. Mode 6 uses `.9` and `.1`, recommends `n/6`, and is called
“ultraconservative” by the source. The implementation preserves this label as
attribution only; this plug-in psi is not a guarantee over arbitrary joint
response distributions. The output pilot sizes are fractional source
recommendations, not rounded enrollment instructions.

The public work is covered by `tests/test_stplan_matched_pairs.py`, including
the literal source equations, the generic inverse registration, scale
invariance of the paired table, no-pilot mode outputs, and invalid variance
geometry. The guide distinguishes this approximate legacy method from exact
matched case-control inference and from current STPLAN menu status.
