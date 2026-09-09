# CDFLIB90 archive inventory and remaining coverage

The [machine-readable inventory](cdflib90-archive.json) accounts for **106 regular
files** in the catalog archive. `tools/audit_cdflib90.py` pins the archive SHA-256,
compares every extracted file with its archive bytes, records member hashes and
sizes, rejects unknown file roles, and inventories source declarations. It also
records directory members; there are no other member types in this archive.
Original source and executables are not bundled in the Python package.

This is a complete file inventory, **not a claim that every archived interface
is implemented or numerically equivalent**. CDFLIB90 remains partial. A source
file's declarations and a matching Python function name are not validation of
all its domains, inversions, endpoint policies or errors.

| Archive material | Files | Coverage disposition |
|---|---:|---|
| F95 distribution modules | 12 | Seven implemented and validated; five pending |
| F95 support modules | 7 | Public/support contract review remains open |
| Binomial editor backup | 1 | Distinct source variant; compare during binomial port |
| F95 build files | 2 | Replaced by the package build and CI workflow |
| Legacy C implementations | 2 | DCDFLIB 1.1 contracts and independent native validation pending |
| Legacy public C header | 1 | 73 external function prototypes inventoried |
| Legacy Fortran source | 64 | 66 declared entry points inventoried, including two ENTRY statements |
| Documentation, installation and notices | 17 | Reference material accounted for; legal terms retained |

## Distribution crosswalk

Each F95 distribution module explicitly exports four interfaces: its `cdf_*`,
`cum_*`, `ccum_*` and `inv_*` functions. The inventory checks all **48 names**.
The C/F77 libraries expose CDF solvers and paired-tail routines with older names.
All 24 legacy distribution names below are present in both source inventories.
The legacy routines do not introduce a thirteenth distribution, but their
contracts and numerical behavior still need comparison with the F95 port.

| Distribution | F95/Python suffix | Legacy CDF / tail names | Current F95 interface coverage |
|---|---|---|---|
| Beta | `beta` | `cdfbet`, `cumbet` | Implemented |
| Binomial | `binomial` | `cdfbin`, `cumbin` | Pending |
| Chi-square | `chisq` | `cdfchi`, `cumchi` | Implemented |
| F | `f` | `cdff`, `cumf` | Pending |
| Gamma | `gamma` | `cdfgam`, `cumgam` | Implemented |
| Noncentral chi-square | `nc_chisq` | `cdfchn`, `cumchn` | Pending |
| Noncentral F | `nc_f` | `cdffnc`, `cumfnc` | Pending |
| Noncentral t | `nc_t` | `cdftnc`, `cumtnc` | Pending |
| Negative binomial | `neg_binomial` | `cdfnbn`, `cumnbn` | Implemented |
| Normal | `normal` | `cdfnor`, `cumnor` | Implemented |
| Poisson | `poisson` | `cdfpoi`, `cumpoi` | Implemented |
| Student's t | `t` | `cdft`, `cumt` | Implemented |

The [method notes](cdflib90.md) document the implemented contracts, native F95
fixtures, independent identities and source repairs. Those fixtures do not
independently validate the legacy C/F77 implementations. The inventory therefore
keeps legacy contract status open even where a corresponding Python distribution
exists. Future work must establish shared behavior, document differences, and
preserve any additional substantive functionality before closing that scope.

The archived DCDFLIB readme explicitly warns that F and noncentral-F CDFs need
not be monotone in either degrees-of-freedom parameter and may have multiple
solutions. Their future inversions cannot assume the monotone shape-search
contract used for beta/gamma. The discrete-family ports must also preserve the
archive's continuous extensions rather than substitute integer quantiles.

## Public numerical and supporting interfaces

The C header declares **73 external functions**, plus two static implementation
helpers. All detected external names have matching definitions in the two C
source files, and no detected external definition lacks a header prototype.
The 64 F77 files declare 66 routines/entries: `dinvr.f` additionally exports
`dstinv`, and `dzror.f` additionally exports `dstzr`. Counting only filenames
would miss those solver-configuration interfaces.

The name differences between the C and F77 inventories are `erf`/`erf1`,
`gamma`/`Xgamm`, and seven C translation helpers: `fifdint`, `fifdmax1`,
`fifdmin1`, `fifdsign`, `fifidint`, `fifmod`, `ftnstop`. Names alone do not prove
semantic equivalence. The remaining shared names include incomplete-beta/gamma
kernels, log-gamma helpers, normal inversion/initial approximations, machine
constants and direct/reverse-communication root finding.

F95 support also has an explicit public surface:

| Module | Public/support scope that remains to be reviewed |
|---|---|
| `biomath_mathlib_mod` | Default-public module with 35 declared numerical procedures, including `log_beta`, `log_gamma`, `log_bicoef`, gamma/beta ratios and approximations |
| `zero_finder` | Direct and reverse-communication interval/step solvers, setup, final-state reporting, bounds and solver-state type |
| `biomath_constants_mod` | Default-public kind and numeric constants |
| `cdf_aux_mod` | Default-public distribution metadata, validation and solver adapters, with explicit private exceptions |
| `biomath_interface_mod` | Numeric/string input generics, console output and message controls |
| `biomath_sort_mod` | Public `sort_list` generic and typed implementations |
| `biomath_strings_mod` | Public case conversion and lexical comparison |

Existing Python/SciPy functions and the package's numerical helpers may replace
many of these responsibilities. That mapping is **not yet established as a
complete public-API replacement**. For each remaining exported responsibility,
record either a validated Python equivalent or a justified implementation-detail
replacement; do not silently discard it because the distribution tests pass.
The declaration inventory records explicit PUBLIC names and the first default
access statement. It is not a compiler-derived export table, and declared
routines can include private procedures, generic implementations or nested scope.

## Alternate source and completion requirements

`#cdf_binomial_mod.f90#` differs from the primary `cdf_binomial_mod.f90`, including
formatting and IMPLICIT NONE declarations. It is not listed among the Makefile's
19 F95 source inputs. Both hashes and declaration inventories are retained;
semantic reconciliation remains part of the pending binomial work. It is not
assumed to be a byte-identical duplicate or silently counted as another product.

Before marking the catalog entry complete:

- Implement and validate the five pending distribution modules and all their
  parameter-inversion modes, including multiple-root and endpoint behavior.
- Compare the 24 legacy distribution entry-point contracts and validate any
  behavior not already established by the F95 references.
- Resolve the public numerical, root-finding and supporting interfaces described
  above, with evidence for each replacement or explicit scope decision.
- Reconcile the alternate binomial source and applicable documentation differences.
- Regenerate this inventory and update its coverage mapping as implementations
  land; keep catalog status partial until the remaining requirements are met.

Run `uv run python tools/audit_cdflib90.py` from the repository root to regenerate
the inventory. The original archive/extraction live under ignored `research/raw`;
the script deliberately fails if they are absent, changed or inconsistent.
