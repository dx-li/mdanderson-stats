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
| F95 distribution modules | 12 | All twelve implemented and validated |
| F95 support modules | 7 | Sorting and strings implemented; five other public/support modules remain open |
| Binomial editor backup | 1 | Distinct source variant; probability-assignment defect validated |
| F95 build files | 2 | Replaced by the package build and CI workflow |
| Legacy C implementations | 2 | All twelve distribution families independently validated |
| Legacy public C header | 1 | 73 external function prototypes inventoried |
| Legacy Fortran source | 64 | 66 declared entry points inventoried, including two ENTRY statements |
| Documentation, installation and notices | 17 | Reference material accounted for; legal terms retained |

## Distribution crosswalk

Each F95 distribution module explicitly exports four interfaces: its `cdf_*`,
`cum_*`, `ccum_*` and `inv_*` functions. The inventory checks all **48 names**.
The C/F77 libraries expose CDF solvers and paired-tail routines with older names.
All 24 legacy distribution names below are present in both source inventories.
The legacy routines do not introduce a thirteenth distribution. All twelve legacy
distribution contracts are implemented with separate C/F77 validation.

| Distribution | F95/Python suffix | Legacy CDF / tail names | Current distribution coverage |
|---|---|---|---|
| Beta | `beta` | `cdfbet`, `cumbet` | F95 and legacy C/F77 beta interfaces implemented |
| Binomial | `binomial` | `cdfbin`, `cumbin` | F95 and legacy C/F77 binomial interfaces implemented |
| Chi-square | `chisq` | `cdfchi`, `cumchi` | F95 and legacy C/F77 chi-square interfaces implemented |
| F | `f` | `cdff`, `cumf` | F95 and legacy C/F77 F interfaces implemented |
| Gamma | `gamma` | `cdfgam`, `cumgam` | F95 and legacy C/F77 gamma interfaces implemented |
| Noncentral chi-square | `nc_chisq` | `cdfchn`, `cumchn` | F95 and legacy C/F77 interfaces implemented |
| Noncentral F | `nc_f` | `cdffnc`, `cumfnc` | F95 and legacy C/F77 noncentral F interfaces implemented |
| Noncentral t | `nc_t` | `cdftnc`, `cumtnc` | F95 and signed legacy C/F77 interfaces implemented with df brackets |
| Negative binomial | `neg_binomial` | `cdfnbn`, `cumnbn` | F95 and legacy C/F77 interfaces implemented |
| Normal | `normal` | `cdfnor`, `cumnor` | F95 and legacy C/F77 normal interfaces implemented |
| Poisson | `poisson` | `cdfpoi`, `cumpoi` | F95 and legacy C/F77 Poisson interfaces implemented |
| Student's t | `t` | `cdft`, `cumt` | F95 and legacy C/F77 Student t interfaces implemented |

All 48 named F95 distribution interfaces now have implementations. The
noncentral t df solver accepts an explicit sign-changing bracket to select
between multiple roots; like the archived full-bound search, it does not
enumerate roots or find tangencies without a sign change.

The [method notes](cdflib90.md) document the implemented contracts, native F95
fixtures, independent identities and source repairs. Those F95 fixtures do not
independently validate legacy implementations.
The separate [legacy F](dcdflib-f.md) and [noncentral F](dcdflib-nc-f.md) ports
have their own unchanged C and F77 references. The [legacy normal port](dcdflib-normal.md)
also validates its unrestricted domains, all four modes and repaired native SD
results. The public support/helper contracts remain open. The [legacy Student t port](dcdflib-t.md)
validates its wider input/search domains and logarithmic tail repairs separately.
Future work must establish shared
behavior, document differences, and
preserve any additional substantive functionality before closing that scope.

The F95 F module accepts only which=1 (tails) and which=2 (F quantile), explicitly
excluding degrees-of-freedom inversion. The older C/F77 `cdff` additionally
accepts which=3 (dfn) and which=4 (dfd). The Python `cdff` and `cumf` interfaces
now implement those legacy modes and the wider input/search bounds, with
independent C/F77 validation and documented Python error/result semantics.
The inventory records that evidence separately from the F95 module.

The noncentral F F95 code additionally computes pnonc with which=3, despite
its header listing only two modes. Legacy C/F77 `cdffnc` uses which=5 for pnonc
and supports additional dfn/dfd modes at which=3/4. The Python `cdffnc`/`cumfnc`
port implements these modes, wide legacy domains and the ignored-q inversion
contract, with separate native and independent evidence.

The archived DCDFLIB readme explicitly warns that F and noncentral-F CDFs need
not be monotone in either degrees-of-freedom parameter and may have multiple
solutions. Their implemented inversions use explicit sign-changing brackets without
assuming the monotone shape-search contract used for beta/gamma. The discrete-family
ports must also preserve the archive's continuous extensions rather than substitute
integer quantiles.

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

| Module | Public/support scope and status |
|---|---|
| `biomath_mathlib_mod` | [Five elementary helpers](cdflib-elementary.md) and [four error/exponential helpers](cdflib-error-exponential.md) implemented; remaining default-public numerical procedures and imported constants still open |
| `zero_finder` | Direct and reverse-communication interval/step solvers, setup, final-state reporting, bounds and solver-state type |
| `biomath_constants_mod` | Default-public kind and numeric constants |
| `cdf_aux_mod` | Default-public distribution metadata, validation and solver adapters, with explicit private exceptions |
| `biomath_interface_mod` | Numeric/string input generics, console output and message controls |
| `biomath_sort_mod` | [Implemented](cdflib-sort.md): all four `sort_list` overloads and custom comparators |
| `biomath_strings_mod` | [Implemented](cdflib-strings.md): ASCII conversion and reentrant lexer `qlex` with documented repairs |

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
19 F95 source inputs. Both hashes and declaration inventories are retained.
The [binomial reference tool](../tools/reference_cdflib_binomial.py) compiles both
versions with unchanged source bytes and captures 102 cases each. All 25 backup
probability inversions in this fixture return success without replacing the
placeholder input probability. Source inspection confirms assignment/termination
ordering differences. Python follows the shared statistical definition and
repairs the inverse behavior; the Makefile-selected primary source remains the
reference version. This alternate source is reconciled, not counted as a separate
product or assumed byte-identical.

Before marking the catalog entry complete:

- Compare the remaining 2 legacy distribution entry-point contracts and validate any
  behavior not already established by the F95 references.
- Resolve the public numerical, root-finding and supporting interfaces described
  above, with evidence for each replacement or explicit scope decision.
- Resolve remaining applicable documentation and cross-version contract differences.
- Regenerate this inventory and update its coverage mapping as implementations
  land; keep catalog status partial until the remaining requirements are met.

Run `uv run python tools/audit_cdflib90.py` from the repository root to regenerate
the inventory. The original archive/extraction live under ignored `research/raw`;
the script deliberately fails if they are absent, changed or inconsistent.

The [legacy gamma port](dcdflib-gamma.md) independently validates all four
computed groups against unchanged C/F77 sources, including executable x/rate
bounds that differ from the source header and repairs for scaled underflow.

The [legacy chi-square port](dcdflib-chisq.md) validates all three computed
groups against unchanged C/F77 sources and independently checks wide inputs,
subnormal half-value rounding and bounded x/df inversions.

The [legacy Poisson port](dcdflib-poisson.md) validates paired tails and both
inversions against unchanged C/F77 sources, with explicit zero-mean semantics,
wide finite inputs and independent small-tail checks.

The [negative-binomial reference audit](dcdflib-neg-binomial-reference.md) records
unchanged C/F77 behavior, independent ordinary-domain validation, boundary
conflicts, false-success inversions and wide-input timeouts. The separate
[legacy Python implementation](dcdflib-neg-binomial.md) now covers all four
modes, wider counts, complementary chance coordinates and documented repairs.

The [legacy binomial reference audit](dcdflib-binomial-reference.md) records the
invalid-mode guard, C-only small-n process exits, wider search bounds and
independently established false-success inversions. The separate
[legacy Python interface](dcdflib-binomial.md) now implements all four modes with distinct count bounds and independent wide-domain validation;
the prior F95 backup-source reconciliation is unchanged.

The [legacy beta reference audit](dcdflib-beta-reference.md) records unchanged
C/F77 ordinary behavior, wider shape bounds, ambiguous endpoints, small-target
and large-shape false successes, and symmetric overflow timeouts. The separate
[legacy beta API](dcdflib-beta.md) now implements all four modes, including the
full two-small-shape domain, with independent implementation validation.

The [legacy noncentral chi-square reference audit](dcdflib-nc-chisq-reference.md)
records unchanged C/F77 behavior, the ignored-q inversion contract, early-series
small-tail failures, invalid wide central probabilities and a large-noncentrality
timeout. The [wider legacy API](dcdflib-nc-chisq.md) now implements all four
modes with independent small-tail checks and documented numerical limits.

The [legacy noncentral-t reference audit](dcdflib-nc-t-reference.md) records
signed noncentrality, ignored q, the executable df upper bound of 1e4,
misleading native status bounds and independently established tail/inverse
failures. The [wider legacy API](dcdflib-nc-t.md) now implements all four modes,
signed noncentrality and independently checked tail repairs.

The [sorting port](cdflib-sort.md) validates all four F95 overloads and custom
comparison callbacks against unchanged source. It repairs duplicate-induced
bounds failures and truncation of strings longer than 256 characters, while
preserving stable ordering, prefix semantics and full-value permutations.

The [string/lexer reference audit](cdflib-strings-reference.md) covers all five
public names, ASCII conversion rules, token classes and native buffer, quote,
malformed-number and overflow failures. The [Python port](cdflib-strings.md) now
implements all five public operations with documented token/numeric policies.

The [elementary support port](cdflib-elementary.md) implements `alnrel`, `rexp`,
`rlog`, `rlog1` and `evaluate_polynomial`, with unchanged F95 evidence and
800-digit checks, including a native subnormal-remainder repair. The mathematical
module remains partial.

The [error-function/exponential reference audit](cdflib-error-exponential-reference.md)
records 134 unchanged F95 calls to `erf`, `erfc1`, `esum` and `exparg`, with
independent defining-function checks. It identifies premature tail cutoff,
intermediate overflow, subnormal double-rounding and the actual normal-range
threshold contract. The [Python port](cdflib-error-exponential.md) now implements
all four with documented overflow handling and preserved subnormal tails.

The [gamma/digamma support audit](cdflib-gamma-support-reference.md) adds 237
native calls for `alngam`, `gamln`, `log_gamma`, `gamln1`, `gam1`, `gamma` and
`psi`, with independent recurrence/Stirling and exact-identity checks. It
separates the different negative-argument contracts and documents intermediate
overflow, lost subnormal tails, inaccurate logarithmic roots and a nonterminating
recurrence. These seven public Python interfaces remain pending.
