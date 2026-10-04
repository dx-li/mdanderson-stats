# CONFINT repeated-calculation report

`ConfintSession` records repeated calls to the existing named CONFINT
calculations. Each call returns the ordinary numerical result and a new
immutable session containing the procedure name, effective keyword settings
(including defaults), and every returned result field. Keep the returned
session to add the next calculation.

```python
from mdanderson_stats import ConfintSession

session = ConfintSession()
session, normal = session.calculate(
    "confint_normal_probability",
    sample_size=45,
    max_length=1,
    population_sd=2,
    confidence=0.9,
)
session, survival = session.calculate(
    "confint_survival_probability",
    hazard=0.2,
    accrual_rate=5,
    accrual_time=10,
    followup_time=0,
    max_length=0.5,
)
session.write_report("confint-calculations.txt")
```

The report labels each calculation by its method and retains the inputs and
full result, including survival error bounds, clipping flags, tails and
convergence details when returned. Confidence level and assurance probability
are recorded separately. Method labels distinguish total CI width, population
SD, the pooled-variance mean-difference interval, Wald difference of
proportions, and exponential-survival hazard-versus-mean quantities. A `None`
result is displayed as an unattainable limit within the procedure's search.

All 18 named Python calculations can be logged. Inputs are scalar except for
the explicit two-value `bounds`/`hazard_bounds` settings and a short tuple of
event counts for the fixed-event survival table (up to 129 rows). Array
broadcast grids are deliberately not accepted. The session is serial and
limited to 100 calculations and a two-megabyte rendered report; the existing
statistical functions retain their own calculation and validation limits.

`report()` returns the readable text and `write_report()` writes that text as
UTF-8 using an atomic replacement. This is a community calculation log for
repeated analysis and review. It does not reproduce CONFINT's interactive
terminal layout, native saved files, or byte-identical Fortran output.
