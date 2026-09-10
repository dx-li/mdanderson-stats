# SPPCR menu runner and command line

`run_sppcr(input_stream=None, output_stream=None, *, rng, ...)` runs repeated
analyses through the original menu choices:

- **0:** exit.
- **1:** read a FileMaker numeric-export file.
- **2:** read a batch-format file.
- **3:** enter observed data interactively.
- **4:** enter truth parameters and generate an experiment.

Input files use the bounded read-only file API. A file-path response of `back`
returns to the menu; `quit` exits. Numeric input and parameter corrections reuse
the validated CDFLIB/SPPCR dialogues. The source banner's initial pause is omitted.
Each action gets fresh data and choices, avoiding the source's repeated-allocation
and uninitialized simulation-output flag problems.

The required RNG can be a NumPy Generator or RandlibGenerator. One continuous
caller-controlled stream is used across analyses. The defaults are 1000 bootstrap
replicates, half-count saturation handling and dropping never-seen input alleles;
truth simulation retains all supplied alleles. Existing numerical repairs and
legacy RNG guards still apply.

```python
from io import StringIO
import numpy as np
from mdanderson_stats import run_sppcr

output, reports, simulations = StringIO(), StringIO(), StringIO()
result = run_sppcr(
    StringIO("2\nexperiment.dat\n0\n"),
    output,
    rng=np.random.default_rng(42),
    report_stream=reports,
    simulation_stream=simulations,
    write_simulations=True,
)
```

## Results and output ownership

`SPPCRRun(reason, completed, rejected, last_analysis)` records `reason="exit"`
or `"eof"`, the completed and rejected analysis counts, and the last completed
analysis (or `None`). EOF during input returns this outcome without analyzing a
partial record. File IO, parsing and numerical-analysis errors are displayed and
return to the menu. Invalid numeric responses use their bounded correction loops;
exhausted console limits propagate. Output/formatting failures propagate rather
than being retried as input errors. Numerical errors may consume RNG state after
sampling begins, as documented by the analysis APIs.

Each completed report is written to `output_stream` and, when distinct, optional
`report_stream`. Requested replicate text goes to `simulation_stream`, or to the
main output if none was supplied. `write_simulations=None` follows the truth
request and defaults to false for observed data; an explicit boolean overrides
that choice. Supplying a simulation stream alone only selects its destination.
Reports and replicate tables carry `SPPCR experiment N` headings across repeated
analyses. These are consecutive labeled sections, not one rectangular CSV table.

All supplied streams remain caller-owned. The input object must differ from output
objects. The runner closes only input files that it opens itself. It never infers
answer-file names or overwrites report files. Callers can use
[staged file publication](sppcr-files.md) for a completed analysis, or supply their
own report streams. [Interactive file saving](sppcr-output.md) is available with `ask_save=True`,
using completed reports and explicit per-file choices.

Limits default to `max_steps=1000` menu iterations, `max_attempts=3` correction
attempts, `max_records=10000` records per input request, and
`max_line_length=10000`. File input and each combined report are bounded to
1,000,000 characters; each replicate report to 10,000,000. These are per-analysis
limits, not a cumulative transcript budget. Stream memory/storage ownership stays
with the caller. Precision defaults to 10 significant digits (1..17), with interval
multiplier 1.959964. Configuration is validated before input or sampling.

## Command line

```sh
python -m mdanderson_stats.sppcr --seed 42
python -m mdanderson_stats.sppcr --legacy --seed 12345 --seed2 67890 --replicates 1000
```

`--seed` is required. Modern sampling uses it as a NumPy seed; legacy sampling uses
it as RANDLIB's first seed and `--seed2` as the second (default 123456789). There is
no clock seed. `--replicates`, `--max-steps`, and `--unseen-alleles drop|retain`
configure the runner. Mutually exclusive `--write-simulations` and
`--no-simulations` override output selection; otherwise the menu request controls it.
The CLI uses standard streams; reports can be redirected by the invoking shell.
`--save-reports` enables interactive output-file selection after each analysis.

Exit status is 0 for normal exit/EOF without rejected analyses, 1 if any analysis
was rejected (even if a later analysis succeeds), and 2 for invalid configuration,
exhausted limits or propagated IO/numerical errors. A normal EOF can have zero
completed analyses; consult output or use the structured Python outcome when
completion counts matter.

Integration tests run all four modes with both RNGs, consecutive truth/observed
analyses, EOF at different input stages, missing/malformed files, file navigation,
stream ownership, output failures and bounds. Real subprocess tests verify modern
and legacy CLI reproducibility and exit statuses. Underlying input dialogues and
numerical kernels retain their separate native reference tests; the repaired full
menu is not claimed to reproduce defective native application behavior.

SPPCR remains partial pending the final source-interface completion audit.
