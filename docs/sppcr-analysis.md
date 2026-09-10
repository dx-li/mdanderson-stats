# SPPCR reusable analysis workflows

`sppcr_analyze(data, *, rng, replicates=1000, saturation="half")` analyzes an
`SPPCRData` returned by batch/FileMaker/interactive input or the data factory.
It preserves original allele identities and omitted-allele diagnostics, fits the
observed counts and bootstraps from their cell fractions.

`sppcr_simulate(request, *, rng, replicates=1000, saturation="half")` takes an
`SPPCRSimulationRequest` from truth entry or programmatic construction. It always
generates the first experiment from supplied truth, then fits it and bootstraps
from truth or observed fractions according to `bootstrap_from_truth`. Simulated
alleles have source-style ordinal labels 1..A, and all remain present even if unseen.

Both return `SPPCRAnalysis(data, bootstrap, request)`. `request` is `None` for
observed-data analysis. Counts, original/replicate fits, uncertainty, boundary flags
and population summaries remain available through the existing immutable records.

```python
from io import StringIO
import numpy as np
from mdanderson_stats import read_sppcr_truth, sppcr_simulate, format_sppcr_analysis

request = read_sppcr_truth(StringIO("2 3\n100\n2 5 3\n.25 1\n1.7\n1 2\ny\ny\n"), StringIO())
analysis = sppcr_simulate(request, rng=np.random.default_rng(42))
reports = format_sppcr_analysis(analysis)
# reports.report contains truth parameters followed by the analysis.
# reports.simulations contains all replicate rows because the request selected it.
```

## Random state and source behavior

These workflows and `sppcr_bootstrap` accept either a `numpy.random.Generator`
for fast vectorized binomial generation or a `RandlibGenerator` for historical
SPPCR float32 binomials. The latter inherits the [legacy safeguards](sppcr-random.md),
including well-count limits and rejection of interior probabilities that round to
an endpoint. Explicit RNG objects are required; no global RNG or clock reseeding
is used. In truth simulation, one continuous stream drives the initial experiment
and then all bootstrap replicates.

The original application calls its bootstrap-choice flag before the first draw,
which can sample uninitialized observed counts when the flag is false. Python
always draws that first experiment from truth. It also avoids the source's
repeated clock reseeding and retains the documented boundary/numerical repairs.
Historical sampling therefore does not claim an identical full native transcript.

Input identities, truth consistency, configuration and unit representability are
validated before initial draws. Once sampling starts, failures propagate and
consumed randomness is not rolled back. For example, an observed-fraction legacy
bootstrap probability can fail its float32 safety guard after the first experiment
has already been drawn. The default half-count saturation policy applies to both
observed and replicate fits; `saturation="raise"` can fail after generation.

The shared data/report model stores genome input DNA and numerical model DNA.
Truth model DNA is halved for this metadata, then verified to double back exactly;
extreme subnormal values that cannot round-trip raise before sampling. Numerical
probability/generation APIs remain available for such designs without this report
metadata conversion. This conversion does not double the truth model DNA used
in the fit.

## Reports

`format_sppcr_analysis(analysis, *, write_simulations=None, precision=10,
multiplier=1.959964, max_report_characters=1000000,
max_simulation_characters=10000000)` returns `SPPCRReports(report, simulations)`.
It composes existing validated reports without fitting, sampling or file IO.
The report limit covers the combined truth-parameter and analysis text.

By default, replicate output follows the truth request's `write_simulations` flag,
or is absent for an observed-data analysis. An explicit boolean overrides output
selection; the truth-parameter section still records the original request.
The formatter verifies truth/data design correspondence and the requested bootstrap
model against sampled probabilities, allowing the legacy float32 representation.
All replicate rows and unavailable/boundary diagnostics are retained.

Tests compare both RNG paths and both bootstrap choices with explicit generation,
fitting and summary calls, including subsequent RNG draws. They also cover
never-seen alleles, input omission labels, all-zero data, invalid state before
sampling, legacy endpoint guards, report composition and total output limits.
The numerical kernels and historical sampler retain their independent native and
mathematical validation; this workflow adds integration evidence.

SPPCR remains partial: main-menu/CLI behavior, file routing, repeated application
sessions and the final source-interface completion audit remain.
