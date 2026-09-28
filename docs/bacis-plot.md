# BaCIS classification posterior density plots

`plot_bacis_classification_posterior` displays the latent-theta density for
each subgroup from an existing `BaCISClassification`. It uses the exact
analytical posterior, so plotting does not refit the model or generate a
Monte Carlo sample. Install the package's existing `plot` extra to use it.

```python
from mdanderson_stats import bacis_classify, plot_bacis_classification_posterior

classification = bacis_classify([2, 3, 1, 7, 8], [25] * 5)
ax = plot_bacis_classification_posterior(classification)
```

The plot returns Matplotlib axes without showing or saving them. The default
new figure uses the native horizontal range [-100,100], with Arm labels and
the archived five-color palette. Its vertical range is calculated from the
densities to avoid clipping. Labels and colors may be supplied per subgroup.

For another latent prior precision or an automatic horizontal range:

```python
ax = plot_bacis_classification_posterior(
    classification, latent_precision=1, xlim="auto",
)
```

The latent prior variance is the reciprocal of `latent_precision`, default
.001. Automatic range means four prior standard deviations on each side of
zero. This display parameter changes the theta scale, not the probabilities
of membership in the two response clusters. The theta axis is not a response
probability axis.

`xlim` accepts `"native"`, `"auto"` or a finite increasing pair. With
`xlim=None`, a newly created axes uses the native range; a supplied axes keeps
its current horizontal view. Explicit limits override that view. Supplied
axes keep their other limits and labels. A finite increasing `grid` overrides
automatic grid construction; bounded grid sizes prevent large plot allocations.

The posterior has two half-normal components. Their densities can have
different limits at zero, although the cumulative distribution is continuous.
The helper draws the negative and positive branches separately, including
their one-sided zero limits when the requested range reaches zero. It does
not draw a false line between the two heights.

The original `bacisPlotClassification` plots smoothed kernel-density estimates
from MCMC samples. Exact analytical curves intentionally differ from a random
smoothed realization, especially near zero. The
[source audit](../research/bacis-file-audit.md) records the native behavior and
the [theta guide](bacis.md) explains the analytical distribution. Other app
response/prior graphs and native PDF report layouts remain separate work.
