# ToxFinder physician-prior elicitation audit

The primary source is Thall et al. (2003), “Dose-Finding with Two Agents in
Phase I Oncology Trials,” DOI 10.1111/1541-0420.00058, §3.1 and §3.3.
The author-hosted PDF is
<https://odin.mdacc.tmc.edu/~jjlee/clinical_trials2012/Thall/References/11_Biometrics%20Two%20Agents%202003.pdf>.

Section 3.1 defines `G(a,b)` as Gamma(shape `a`, scale `b`), with mean `ab`
and variance `ab^2`; the single-agent odds are `alpha*x^beta`. In §3.3, the
four elicitation answers are translated to equations (7)–(10), with
`zj=dj/AD` and `g(eta)=eta/(1+eta)`:

1. `Pr[g(alpha*z1^beta) < q_low] = confidence`.
2. `E[alpha] = a1*a2 = target/(1-target)`.
3. `E[alpha*z3^beta] = a1*a2*E[z3^beta] = q_high/(1-q_high)`.
4. `Pr[g(alpha*z4^beta) > target] = confidence`.

The paper's defaults are `q_low=.05`, `q_high=.60`, and `confidence=.99`; it
explicitly allows replacing the 5% and 60% values. Given the target and
`z3>1`, equation (3) analytically gives beta scale as a function of beta
shape, provided the resulting Gamma moment-generating function exists. This
reduces the remaining fit to two probability constraints in alpha and beta
shape. The implementation uses bounded least squares on log shapes and a
fixed-order probability-space Gauss-Legendre quadrature for the beta
expectations. Optimizer starts/bounds and quadrature tolerances are Python
conventions because the paper specifies only numerical solution, not a solver.
It fails if constraints cannot be met within the declared probability
residual tolerance.

Table 1 uses target `.30`. Gemcitabine elicited doses are `(600,1200,1400,2000)`
and reports alpha moments `(.4286,.1054)` and beta moments `(7.6494,5.7145)`.
Cyclophosphamide doses are `(350,600,700,800)` and reports
`(.4286,.0791)` and `(7.8019,3.9933)`. The article's sentence around
`d(3)=d*` conflicts with the four questions, equations, and Table 1; the
implementation follows those consistent equations/table (`d2=AD`, `d3>AD`).

The prior builder requires interaction Gamma moments explicitly. The source's
general recommendation is alpha3/beta3 means `(1,.05)` and variances `(3,3)`;
its case study instead uses beta3 mean `1`, variance `.9`. No global default is
claimed. Existing ToxFinder fitting and first-stage calculations consume the
resulting `ToxFinderPrior`; stage-two information selection and complete
native conduct remain outside this batch.

The test fixture uses the two Table 1 elicited dose sets and solves the literal
four equations for each. For each fitted agent, a separate adaptive integral
over the beta density checks the two probability equations independently of
the implementation's fixed probability-space quadrature. Separate checks
evaluate the published rounded Table 1 moments themselves against those same
probability equations.

That independent adaptive integral also exposed an internal source mismatch:
the Table 1 rounded moments yield probability pairs `(0.986614, 0.992176)` for
Gemcitabine and `(0.977774, 0.978817)` for Cyclophosphamide under the literal
equations/defaults, rather than `(0.99,0.99)`. The solver therefore validates
its equation residuals independently and does not assert that its fitted
moments equal Table 1.
