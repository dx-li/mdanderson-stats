# Legacy noncentral-t reference audit

This audit establishes unchanged C/F77 evidence for the `cdftnc` and
`cumtnc` interfaces. Existing F95 Python noncentral-t functionality supplies
validated overlapping behavior, including explicitly bracketed df inversion.
It does not yet cover the legacy signed noncentrality and wider input contract.

## Executable contracts and misleading documentation

The executable supports modes 1–4 for p/q, t, df and pnonc. The header's legal
range 1..3 is stale, while an invalid high mode reports bound=5 rather than the
actual maximum four. Input t and pnonc are unrestricted finite signed values;
df is positive finite. Noncentrality is the normal numerator's signed mean,
not the squared-shift parameter used by noncentral chi-square and F.

All inversions require p in [0,1-1e-16], including both endpoints. Despite the
header describing a complementary pair check, q is entirely ignored: it is
neither checked nor used, and its input value survives unchanged. A Python
legacy interface must state its own returned-q convention explicitly.

Computed t spans [-1e100,1e100] and noncentrality [-1e4,1e4]. **Computed df
spans [1e-100,1e4] in both executables**, not the header's [1e-100,1e10].
When a df search reaches its upper endpoint, status 2 nevertheless reports
bound=1e100, not the actual searched endpoint 1e4. A failed lower noncentrality
search similarly reports bound=0 instead of -1e4. Source inspection, rather
than trusting those returned bounds, establishes the search contract.

Inversion uses P alone, starts at five, and has absolute tolerance 1e-50 and
relative tolerance 1e-8. Df need not be monotone, so a global endpoint search
can miss roots. Explicit brackets will remain necessary for selecting roots.

The tail routine reduces |pnonc|<=1e-10 to central t and |t|<=1e-10 to a normal
probability. Otherwise it forms t*t and pnonc*pnonc, computes beta coordinates,
and sums a series with convergence criterion 1e-7. Intermediate beta thresholds
can create incorrect flat regions, while overflow and rounded coordinates
can lose the intended distribution limit.

## Native evidence and validation

`tools/reference_dcdflib_nc_t.py` compiles the unchanged two C sources and header,
and all 64 F77 sources. Fixtures retain archive/source hashes, compiler commands
and versions, driver text and empty source-adaptation lists. Drivers initialize
status to -999 and bound to zero; bound is undefined on success. No original
numerical source or executable is bundled.

Each language records 171 ordinary calls: 45 tails, 45 t inversions, 36 df
inversions and 45 noncentrality inversions. The grid spans t=-2,-0.5,0,0.5,2;
df=0.5,2,10; and pnonc=-3,0,3. All calls complete. Twenty-four df inversions
succeed and twelve return status 2; all other ordinary calls return status 0.
Five invalid, four ignored-q, six boundary and eight wide cases per language
are also retained. All calls use a three-second timeout and distinguish timeout
or process failure from returned status; none of these fixture calls timed out.

The 378 tests compare ordinary behavior with the existing Python implementation,
using F(-t,df,-nc)=1-F(t,df,nc) to cover signed noncentrality. Thirty df=2 native
tails receive independent normal-CDF closed-form checks. Native df=2 absolute
errors reach about 1.03e-8; Python tails are checked separately at much tighter
relative tolerance. Successful inverse outputs are evaluated forward, with
explicit failure assertions for the incorrect zero-quantile cases. Local df
brackets recover roots missed by the native global search.

## Independently established failures

For every positive df, F(0,df,nc)=Phi(-nc). Its quantile is zero, up to the
rounding of the supplied probability. For nc=+/-3, both native languages instead
return quantiles with magnitudes roughly 0.014–0.078 across the grid. These
values miss the requested probability by more than 1e-5; they are not small
parameter-rounding errors. The tests retain these false successes explicitly.

For df=2, writing r=|t|/sqrt(t*t+2), the CDF equals
Phi(-nc)+r*exp(-nc*nc/(t*t+2))*Phi(nc*r) for t>=0, with the reflected negative
term for t<0. At t=-1,nc=10, native P is Phi(-10), approximately 7.61985e-24,
more than ten times the independent closed form. The reflected t=1,nc=-10
upper tail has the same defect. The existing Python quadrature repair agrees
with the closed form.

At t=1,df=1e200,nc=3, native P is Phi(-3), approximately 0.0013499, while the
large-df limit is Phi(-2), approximately 0.0227501. The tests do not merely
assume convergence: Chebyshev bounds V/df within 1+/-1e-6 for V~chi-square(df),
and the resulting conditional-normal bounds exclude the native answer.
The retained df=1e-308 case instead agrees with the small-df limit Phi(-3).
These distinct regimes need different handling in the legacy implementation.

For t=1,nc=3,P=0.03, native df inversion returns status 2 at df=1e4 and reports
bound=1e100. Two roots actually exist within its search domain, approximately
0.03926845 and 1.65497763. Explicit Python brackets recover both. By contrast,
for t=1,nc=1,P=0.4999995, a valid F95-domain root near 199470.93 genuinely lies
outside the legacy executable's 1e4 upper limit. A faithful port must distinguish
these situations instead of adopting the header's wider df range.

At t=0,nc=0,P=0.5, df is unidentified, yet native inversion returns its arbitrary
initial value five with status 0. The existing Python interface rejects this.
The p=0 central quantile fixture returns status 1 with lower bound -1e100;
positive finite-df t distributions have no finite exact-zero quantile.

The [wider legacy API](dcdflib-nc-t.md) now implements both entry points and
all four modes, with independent checks and recorded batching measurements.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
