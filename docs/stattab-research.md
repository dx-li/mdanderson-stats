# STATTAB research cross-reference

Catalog entry **23, STATTAB**, remains pending. The CDFLIB90 archive unexpectedly
contains `CDFLIB90/source/cdflib90_1.2/DOC.TEX`, whose title identifies it as the
STATTAB 2.0 manual dated March 2002. It is not a duplicate of CDFLIB90's manual.
The [CDFLIB90 completion audit](cdflib90-completion.json) retains its exact hash.

The catalog advertises a separate `STATTAB      _V1.3.zip` download. Its executable,
source and version differences must be inspected before implementing STATTAB;
the embedded 2.0 manual alone does not establish the 1.3 program's behavior.

The manual identifies responsibilities still tracked under STATTAB:

- A twelve-distribution menu and parameter workflow, including help, `?` for the
  unknown parameter, `.` for an omitted complement and `=` for a retained value.
- Lists of parameter values and customized result tables, including the `T`
  list selector and all eight list-editing actions.
- Two-sided p-values for normal/t and many-sided p-values for chi-square/F.
- Individual binomial and Poisson probabilities using truncated integer inputs,
  alongside their continuous-extension CDF calculations.
- Report-file output and annotated interactive examples.

The package's completed distribution, console, lexer, number-list and root
components provide reusable foundations. Their existence does not implement
or validate the STATTAB application workflow, additional probability outputs,
formatting, state reuse or examples. Those remain part of the full catalog goal.
