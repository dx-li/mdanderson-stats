# SPPCR interactive output-file selection

`sppcr_output_dialogue(console, reports, *, default_name="sppcr",
protected_paths=(), max_attempts=3, max_append_characters=10000000)` takes completed
`SPPCRReports` and returns `SPPCRSavedFiles(status, report, simulations)`. Status is
`"saved"` with absolute paths or `"declined"` with no paths. The caller retains
ownership of the console streams.

The dialogue asks whether to save, then selects an answer path and (when requested)
a simulation path. Blank input chooses the basename of `default_name` with `.ans`
or `.sim` in the current directory, matching the original application's basename
convention. Explicit paths can select another directory. Directories must exist.
`quit` or `back` cancels saving; existing paths offer the source choices:

- `q`: cancel saving.
- `r`: retry filename selection.
- `o`: overwrite.
- `a`: append.

The two paths may use different modes. Input/protected paths, named active console
streams, and the other output cannot be selected, including existing filesystem
aliases. Invalid paths or conflicts have bounded retries. Output-path text follows
the console string contract, including `#` comments.

Neither file is changed while choices are being collected. Cancellation or EOF
after selecting overwrite for the first path still preserves that file. Existing
content for append is read as UTF-8 with a character bound; all output text is
encoded before staging. All files are staged before any publication. This repairs
the native helper's truncation-before-confirmation behavior.

Append preserves the existing text literally, without inserting separators. It
publishes a staged snapshot of old content plus the new report; it is not a
concurrent append protocol. Concurrent modifications between reading and replacement
can be lost. Replacement replaces a symbolic link itself, preserving the old target.
New paths use exclusive publication, even when another output uses overwrite or
append. Each file is published atomically, but the pair is not one transaction:
a late publication failure can leave an earlier complete output saved. Errors
propagate, with the [file layer's cleanup and durability limits](sppcr-files.md).

## Menu and CLI integration

Set `run_sppcr(..., ask_save=True)` to add this dialogue after each completed
analysis has been displayed. The default remains false for existing callers.
File input modes use the input basename; other defaults are `interactive.ans` and
`generated.ans`, with corresponding `.sim` names. The input data path and named
caller-supplied report/simulation streams are protected against selection.

`SPPCRRun.last_files` records the last completed analysis's save outcome. If EOF
occurs during file selection, the runner returns reason `"eof"`, retaining the
completed analysis and count, with `last_files=None`. No file has been changed at
that point. Publication errors propagate and are not silently retried as analyses.
Repeated analyses collect new choices; a prior overwrite or append choice never
implicitly authorizes the next analysis's writes.

```sh
python -m mdanderson_stats.sppcr --seed 42 --save-reports
```

The CLI flag enables the same dialogue; all original menu modes remain available.
The answer and optional replicate files contain the complete per-analysis reports,
without the multi-analysis headings used on shared console output streams.

Tests verify new default paths, all four existing-file choices, independent append
and overwrite modes, protected inputs/active streams, output alias retries,
cancellation and EOF before publication, append bounds, menu outcome preservation
and actual CLI file saving. Existing file-layer tests continue to verify staging
failures, exclusive-publication races and temporary-file cleanup.

SPPCR remains partial until its final source-interface completion audit is finished.
