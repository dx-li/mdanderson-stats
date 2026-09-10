# SPPCR file workflows

`read_sppcr_file(path, *, file_format="batch", unseen_alleles="drop",
max_characters=1000000, max_runs=50, max_alleles=None)` opens one UTF-8 file read-only,
reads at most `max_characters + 1` characters, closes it, rejects oversized content,
and calls the corresponding validated parser. Choose `file_format="filemaker"`
for the native numeric export format. Allele defaults are 50 for batch and 25 for
FileMaker. Explicit larger limits work as with the parsers; native compatibility
above native limits is not promised. Paths may contain spaces; input files are
never modified. Invalid options are rejected before opening; decoding, IO and
parsing errors propagate.

```python
import numpy as np
from mdanderson_stats import (
    read_sppcr_file,
    sppcr_analyze,
    format_sppcr_analysis,
    write_sppcr_reports,
)

data = read_sppcr_file("experiment.dat")
analysis = sppcr_analyze(data, rng=np.random.default_rng(42))
reports = format_sppcr_analysis(analysis, write_simulations=True)
paths = write_sppcr_reports(reports, "experiment.ans", "experiment.sim")
```

The read path composes previously reconciled native formats with normal Python
stream ownership. Native `open_file_mod` opens even read inputs with READWRITE;
Python only requires read access. Existing STATTAB file dialogue remains available
for interactive file selection; the [SPPCR menu](sppcr-console.md) now routes all four input modes.

## Report publication

`write_sppcr_reports(reports, report_path, simulation_path=None, *, overwrite=False)`
returns the absolute report and optional simulation paths. It consumes completed
`SPPCRReports` and never fits data or consumes RNG state. Simulation text and its
path must both be present or both absent, so requesting output cannot silently
lose replicate results. It uses the paths exactly as supplied, without inferring
extensions or creating missing directories.

Before writing, it validates options, encodes all text to UTF-8, rejects output
aliases (including existing hard links and symbolic links), and checks destination
conflicts. It stages both outputs in their destination directories, flushes and
syncs their contents, and then publishes them. Temporary files are cleaned up on
success and failure. Each destination becomes visible as a complete file.

The default uses atomic exclusive publication through a hard link, so an existing
file or a competing writer's new file is never truncated. Filesystems that do not
support that operation raise an IO error. `overwrite=True` explicitly selects
atomic replacement. Replacement of a symbolic link replaces the link itself and
preserves its target. Existing file permissions are not copied onto the new staged
file; publication uses the temporary file's permissions.

**The two outputs are not one transaction.** A failure before publication preserves
both destinations. A race or IO failure during later publication can leave the
earlier output successfully published. No rollback or backup is attempted, and
concurrent writers are not coordinated. File contents are synced, but this API does
not promise crash durability of directory metadata or a cross-file atomic commit.

Tests exercise complete batch/FileMaker file-to-analysis-to-report workflows,
read-only inputs, character and dimension bounds, UTF-8 failures, default exclusive
creation, explicit replacement, output alias detection, missing-directory and disk
sync failures, symlink replacement, and a competing writer at publication time.
Assertions check actual file bytes, preserved destinations, and temporary-file
cleanup rather than only mocking successful writes.

The [output dialogue](sppcr-output.md) adds independent overwrite/append choices
and menu integration. SPPCR remains partial pending the final source-interface
completion audit.
