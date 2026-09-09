# CDFLIB90 completion audit

Catalog entry **21, CDFLIB90**, is implemented with documented Python semantics.
This conclusion covers its F95 1.2 library and bundled legacy C/F77 library. It
does not claim bitwise equivalence, reproduce native process termination or
discard the numerical limits documented for individual routines. The overall
MD Anderson catalog conversion remains in progress.

## Scope and evidence

| Requirement | Verified evidence |
|---|---|
| Complete archive | [Inventory](cdflib90-archive.json): 106 regular members, pinned archive and member hashes, extraction checked byte-for-byte |
| F95 distributions | All 12 modules and 48 public entry points; separate native fixtures and independent mathematical tests for each family |
| F95 support | Seven modules: sorting, strings, 27 constants, 35 mathematical procedures, root state/search, distribution descriptors/validation and console/list/format/message operations |
| Actual public visibility | All 19 unchanged F95 modules compile and link with a 252-name import probe, including transitive public USE associations |
| Legacy C/F77 | All 73 external C functions and 66 F77 routines/entries mapped; 24 distribution names and 49 support names, with documented C/F77 aliases and seven C-only helpers |
| Python availability | Runtime checks of all distribution and support mappings against the package namespaces; source-declared public names compared with the crosswalk |
| Alternate source | Primary/backup binomial sources separately exercised in 102 native cases each; 25 false-success backup inversions repaired |
| Documentation | All 17 members assigned an explicit disposition; manual errata and the separately cataloged STATTAB manual reconciled below |
| Installation and notices | Native build scripts replaced by the Python build/CI; CDFLIB90 LEGALITIES retained byte-for-byte in the packaged notice |

The [completion verifier](../tools/audit_cdflib90_completion.py) records its
compiler version, flags, driver hash, checked exports and document dispositions
in [machine-readable evidence](cdflib90-completion.json). It rejects missing or
extra declared mappings. Fortran's default-public mathematical and auxiliary
modules re-export constants and other support interfaces; Python exposes their
documented namespaces instead of duplicating every transitive alias.

An export probe establishes visibility and coverage, not numerical accuracy.
The numerical evidence remains the per-interface native fixtures, independent
Decimal integrals/series, distribution identities, inversion checks, protocol
tests and documented defect regressions linked from the
[coverage crosswalk](cdflib90-coverage.md). Existing tests exercise all supported
Python versions, while isolated-wheel checks verify installed imports and notices.

## Documentation reconciliation

The archive contains four formats of the CDFLIB90 user guide: PDF, PostScript,
LaTeX and extracted text. The PDF has 31 pages; its module list and relevant
formula pages were visually checked alongside the text and LaTeX sources.
The remaining documentation comprises the F95 readme, installation instructions
and legalities, plus C/F77 readmes, distribution summaries/full descriptions,
historical acquisition instructions and a distribution-only C header.

Executable source and independent mathematical validation take precedence over
contradictory manual text. Material reconciliations include:

| Archived discrepancy or convention | Python contract and evidence |
|---|---|
| Beta density omits the minus-one exponents; t density omits its negative exponent | Standard beta and t distributions, confirmed by native outputs and independent identities/integrals |
| F density support printed as [0,1] | Nonnegative F coordinate with each interface's documented finite bounds |
| Per-family CHECK_INPUT prose contradicts the general description | Always validate Python inputs; no unsafe unchecked mode; errors become exceptions |
| Gamma SCALE shown as a conventional scale in a density formula | Executable source multiplies x by SCALE; Python names this parameter `rate` |
| Noncentral F manual lists only two groups | F95 source also solves noncentrality; legacy source additionally solves both df parameters using explicit root brackets |
| Noncentral routine headings omit `nc_`; inverse t includes an extra T argument | Names and signatures follow the executable public interfaces and documented Python return-value mapping |
| Discrete CDFs use continuous extensions | Fractional-count inversions retained; integer quantiles are not silently substituted |
| C/F77 summary bounds and ignored-q rules differ from F95 | Separate legacy implementations and fixtures preserve/reconcile those contracts |
| Stateful root helpers, status outputs and STOP | Independent state/results, explicit failure flags and exceptions; both native tolerance conventions are documented |

Detailed bounds, endpoint choices, nonidentifiable parameters, multiple-root
behavior and precision limits remain in the individual method documents. A
finite input can still raise when no acceptable representable inverse or verified
numerical result exists. Completion is not a guarantee of uniform accuracy over
every float64 value.

The archive's root `DOC.TEX` is **STATTAB 2.0 documentation**, not a fifth format
of the CDFLIB90 manual. Its title, menu workflow and additional p-value/individual
probability outputs identify the separate catalog entry 23. Those responsibilities
remain explicitly tracked in [STATTAB research](stattab-research.md); STATTAB is
not marked complete by this audit. No CDFLIB90 library interface is dropped by
that cross-reference.

## Reproduction

From a checkout with the pinned ignored research archive/extraction:

```text
uv run python tools/audit_cdflib90.py
uv run python tools/audit_cdflib90_completion.py
uv run --extra plot pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
```

The first command verifies archive membership, hashes and per-interface evidence.
The second verifies the complete declaration crosswalk and compiles the public
import probe. The test suite establishes numerical and behavioral validity;
formatting, type checks and the build validate the deliverable. Original native
source and temporary compiler products are not included in the Python wheel.
