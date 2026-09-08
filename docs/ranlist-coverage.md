# RANLIST coverage audit

Catalog entry 29 is implemented for the documented randomization-list workflows
in the archived Fortran source and manual. This audit accounts for 46 program
units and seven ENTRY points. It does not claim identical terminal dialogs,
byte-for-byte report typography, or execution parity with the Windows binary.

## Archive evidence

The top-level archive contains the Fortran source, source readme/manual, and
`win32/ranlist_1.2_se.exe`. The latter is a self-extracting ZIP containing README,
ranlist.exe and ranlist.doc. The two manuals have identical text after newline
normalization. The Windows README describes the same restricted/unrestricted
and fixed/random-balance functionality; no additional documented workflow was
found. The source and manual identify version 1.1, July 1992. The Windows archive
filename identifies 1.2; its executable was inspected as a Windows artifact but
was not executed. Validation uses the archived source compiled with gfortran.

| Artifact | SHA-256 |
| --- | --- |
| source/ranlist.f | 346b9d3550bfec2a4b47ac5d3eb3b142b57417d38d450b36fc37b398a5284844 |
| source/ranlist.doc | 70131138346985b0c6372b4b7f6c2f1de20693b42177b5a23cf0430a9397a13b |
| source/readme | 1e54871d933ce7e4be529b4269dc803402576c6ea876cebfef01a86498aa907f |
| win32/ranlist_1.2_se.exe | 7afb8ef4aae0546f914bf319624a95dc4557b0986d62389a19b804ac9cce5cee |
| unpacked ranlist.exe | 2f01542fca25749c296c60a557f3e7c5f7adcc2d99808113933ceea01371521a |
| unpacked ranlist.doc | b0dd693b654f40a838ec2a62127481c41e28a4489409d9983347f302f9256593 |
| unpacked README | 62687d65b129affb078fcb670f1f0cce7e110d8b2d31e2caa3977a3e0079a7a3 |

Original archives, source, manuals and executables remain local research
material. The source readme and legal section are retained under `notices`;
their conditions are described in THIRD_PARTY_NOTICES.md, without a blanket
license replacing the original terms.

## Source-unit mapping

| Source units / entries | Python disposition and evidence |
| --- | --- |
| ranlst | Specification/session, persistence and report APIs replace the three top-level menus; all three workflows executed natively |
| mklst | `RanlistSpecification`, source parameter export and `ranlist_starting_seeds`; six native creations cover restricted/unrestricted and numeric/short/long phrase seeds |
| gtseed, phrtsd, lennob | `ranlist_starting_seeds` implements GTSEED's 31-character entry and zero repair; `ranlist_seeds` retains the raw PHRTSD hash and space trimming; nine hash fixtures and native setup cases |
| ignlgi, ranf, mltmod | Exact integer modular recurrence and vectorized jumps, modern/float32 scaling; 36 native stream/block cases and independent recurrence tests |
| setall, getcgn, setcgn, initgn | Explicit seeds and stream numbers; stream spacing 2^50 and block jumps 2^30 validated natively, plus distant-position arithmetic tests |
| inrgcm, qrgnin, qrgnsn, rgnqsd, setsd | Generator initialization, flags and seed replacement become explicit local state/arguments; no process-global generator state or initialization order; SETSD is not called by the RANLIST workflows |
| igtut | `ranlist_unrestricted`; 40 native cases including inclusive cumulative boundary and sequential float32 accumulation |
| igtrt, genprm, ignuin | `ranlist_restricted`; 48 native cases and independent balance/refill/rejection tests, including the source's skipped-draw behavior |
| wrklst | `RanlistSession.enroll/inquire`, counters, summary and persistence; four complete native enrollment/inquiry/file-update workflows |
| genlst | `ranlist_summary/ranlist_report`; eight complete native print workflows, 452 rows on 32 pages, enrolled/planned counts and named/unnamed strata |
| qgibfu, qnibfu, qgobfu, qnobfu, qgtofn, igtfun | Explicit paths/text replace interactive file selection and Fortran logical-unit allocation; validated original records and atomic JSON snapshot replacement, including real filesystem failure tests |
| gtcuio, stcuio, gtecun, stecun, gtstio | Terminal-unit bookkeeping is not required by the Python API; no hidden input/output unit state |
| chrgt, intgt, qgetin, qgetrl, qgtchr, qgtint, qgtstr, qgtyn, qyngt, qlex, strgt, trnchr, lens | Python arguments and validation replace keyboard lexing, selection, character translation and retry dialogs; metadata dimensions, numeric contracts and invalid requests tested |
| banner, clrscr, center, pause, prompt, wrhelp | Documentation, retained notices and readable reports replace screen clearing, pauses, text centering and interactive help; full labels retained rather than source display truncation |

## Deliberate corrections and limits

- Modern restricted allocation redraws random K at every refill as the manual
  specifies. Legacy reproduces IGTRT's single K per stratum and its indexed
  skip rule. Modern integer sampling removes IGNUIN's inclusive-endpoint bias.
- Modern unrestricted normalization avoids overflow and forces a complete CDF.
  Legacy preserves source rounding but raises if a draw exceeds the CDF, where
  the source could read outside the treatment array.
- GTSEED truncates phrase input to 31 characters, replaces a zero first seed by
  one and a zero second seed by twelve. Raw PHRTSD hashing remains separately
  available without these changes. Explicit invalid numeric seeds are rejected.
- Native MKLST execution confirmed that unrestricted setup can write `***` in
  unused balance fields. Import rejects that corruption by default;
  `repair_unrestricted_balance=True` explicitly ignores only those unused
  fields. Restricted balance fields always remain validated. New Python lists
  initialize unrestricted balance settings to `(1,1)`.
- Source I1 treatment counts, F6.3 weights and I6 counters have narrow ranges.
  Export rejects overflow and requires explicit permission through the API
  option for changes caused by decimal rounding. JSON preserves all precision,
  algorithm settings and counters.
- Source limits of 20 named strata/treatments and 500 assignments per balance
  block are retained. Indexed kernels expose all 32 RNG streams. Explicit row
  and block limits bound work; failures never silently truncate results.
- Patient numbers are per stratum. Session updates are immutable; inquiries
  cannot access unenrolled patients, while specification/report APIs may query
  future positions. File snapshots do not coordinate concurrent writers.
- Report assignments and page boundaries match the source. Python prints full
  labels/values and simpler headings, and omits irrelevant unrestricted balance
  settings. This is a functional replacement, not a terminal emulator.

## Validation and performance

Fixtures record source hashes and compiler flags. Native coverage comprises
36 stream/block cases, nine raw phrase hashes, 40 unrestricted and 48 restricted
allocation cases, two fixed-width record examples, four complete enrollment
workflows, eight complete print workflows and six complete creation workflows.
Independent tests cover balance invariants, probability boundaries, unbiased
refills, rejected draws, ordering/shapes, immutable state, limits and corrupt
files. The zero-seed repair follows the source's explicit branches; native
creation fixtures exercise ordinary and truncated phrases, not a zero hash.

[ranlist-benchmark.json](ranlist-benchmark.json) records three-run median
comparisons of batched and repeated scalar calls to the same Python APIs.
On the recorded machine, 10,000 unrestricted patients were about 1052x faster,
2,000 legacy restricted patients about 29.4x faster, and 500 modern restricted
patients about 226x faster when batched. Each run checks output equality.
These measure batch reuse and vectorized arithmetic, not speed relative to
Fortran, report formatting, filesystem I/O or arbitrary workloads. Modern
restricted queries still replay preceding blocks to preserve refill/rejection
state; legacy queries jump directly to requested blocks.
