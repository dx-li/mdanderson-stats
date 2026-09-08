# EXPSURV exploratory survival analysis

Catalog entry 28 is partial. The survival-curve and inverse-survival core is
implemented, together with interactive cut-point exploration. Linked scatterplot
matrices, event charts,
survival box plots, interactive model-alignment views, data generation and
remaining input/output workflows are still pending.

The source is EXPSURV version 1 from the MD Anderson catalog, distributed as
`EXPSURV_V1.tar.gz`. It contains an XLISP-STAT source file, a TeX user manual and
a readme. The source/readme permits redistribution; original code remains local
reference material and is not bundled. Source SHA-256:
`b3807d82f5e18b9d6c564642bcb6b0f785103847578e1d9d5a19581a846a2492`.

```python
from mdanderson_stats import exploratory_survival

fit = exploratory_survival([1, 2, 3, 4], [1, 1, 0, 1])
print(fit.time, fit.survival)
print(fit.at([0, 1.5, 4]))
print(fit.survival_quantile([0.75, 0.5, 0.25]))
print(fit.survival_quantile(0.5, method="source"))
```

`exploratory_survival` computes Kaplan–Meier survival. Inputs are stably sorted
without mutation. Status is binary (one=failure, zero=censoring), and omitted
status means all failures. The default groups tied observations and includes
tied censoring in the common risk set. `legacy=True` retains KMEST's sequential
per-record product and order within ties. Unlike KMEST, Python explicitly sorts
input rather than requiring callers to have applied COSORT first.

Results include event times, survival, risk counts, events and censor counts,
along with the exact left/right corners used by KM-PLOT (`step_time` and
`step_survival`). `.at()` is right continuous, with survival one before the first
observation and flat extension beyond follow-up. Zero-time failures retain their
initial vertical jump. Result arrays and query outputs are read-only.

`.survival_quantile(q)` takes a **survival level**, not a cumulative probability.
Its default step method returns the first time survival is at most q, with q=1
at time zero. An unreached level returns NaN rather than a fabricated estimate.
`method="source"` implements QUANT's linear interpolation from the bracketing
survival levels, including its unusual last-time choice at an exact plateau.
For survival [.75, .5, .5, 0] at times [1, 2, 3, 4], the default q=.5 result is 2;
the source result is 3. This interpolation is not the ordinary inverse of a step
function and is selected explicitly, independently of the tie convention.

Validation uses exact rational per-record products on censored, uncensored and
tied samples; independently computed grouped risk sets; right-continuous queries;
plot corners; interpolation and plateau examples; scaling; time-zero events;
immutability; and invalid input checks. No XLISP-STAT interpreter was available
in this environment, so these are source-formula and independent mathematical
checks, not claims of executing the archived program. Curve construction uses
NumPy sorting/aggregation/products and queries use binary searches; no measured
speedup is claimed for this increment.


## Cut-point exploration

```python
from matplotlib import pyplot as plt
from mdanderson_stats import survival_cutpoint, plot_cutpoint

data = survival_cutpoint([3, 1, 4, 2], [1, 0, 1, 1], [2, 1, 2, 3])
comparison = data.compare(2)
print(comparison.lower_indices, comparison.upper_indices)
view = plot_cutpoint(data, cut=2)
plt.show()
```

`survival_cutpoint` retains copied, aligned observations. `.compare(cut)` fits
separate survival curves for `covariate <= cut` and `covariate > cut`, preserving
original row indices. The comparison is immutable and has no plotting dependency.
An empty group has a None curve and an empty index array, rather than invented
survival estimates. Any finite cut is allowed for numerical comparisons.

With the optional plot extra, `plot_cutpoint` returns a controller containing a
figure, linked density/survival axes and a Matplotlib slider. Keep the controller
alive while interacting. The slider has 50 positions from the minimum to maximum
covariate, initially at index 25 as in the archived function. An explicit initial
cut or `view.set_cut(value)` may use any value inside the observed range. Updates
change the density marker, both curves, and group-count labels together. Empty
groups remain visible as n=0 labels with no curve. `view.close()` disconnects the
callback and closes the figure. No global backend is changed; use an interactive
Matplotlib backend for dragging, or call `set_cut` programmatically and export
with `view.figure.savefig(...)`.

The two linked panels replace the source's separate windows. The covariate
density guide uses a Gaussian kernel with `std(ddof=1)*n**(-1/5)` bandwidth,
or an explicit positive `bandwidth`. It is evaluated once in bounded chunks and
reused when moving the cut. This is an explicit replacement for XLISP-STAT's
implicit `kernel-dens` runtime dependency, not a claim of identical density
defaults. A constant covariate cannot support a varying slider and raises an
explicit error; the numerical `compare` method still supports it.

Tests verify equality at the cut, unsorted row alignment, exact group membership,
empty endpoints, immutable snapshots, source tie mode, callback updates to all
artists, initial slider synchronization, invalid-update state preservation,
density normalization/symmetry and figure cleanup. Both an interior-cut figure
and an empty-upper-group endpoint figure were rendered and visually inspected.
This is exploratory group comparison, with no automatic cut optimization or
inferential test added by the plotting workflow.
