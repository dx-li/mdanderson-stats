"""Audit STATTAB's pinned shared sources and compile their public imports.

Logical-line diffs aid review; they are not a Fortran semantic equivalence proof.
"""

import difflib
import hashlib
import json
import math
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256 as CDFLIB_SHA256
from audit_cdflib90 import F95_ROOT
from audit_stattab import ARCHIVE_SHA256, SOURCE_ROOT, read_archive, source_order

REVIEWS = {
    "biomath_constants_mod": "IMPLICIT NONE, comments and whitespace; constants unchanged.",
    "biomath_interface_mod": (
        "Public/data declarations relocated; CHARACTER(*) becomes CHARACTER(LEN=*); "
        "redundant INTRINSIC TRIM removed. Blank-input warning uses an equivalent "
        "quoted FORMAT with an explicit comma; both formats are executed below."
    ),
    "biomath_mathlib_mod": "Byte-identical source.",
    "biomath_sort_mod": "PUBLIC SORT_LIST relocated after generic interface; procedures unchanged.",
    "biomath_strings_mod": "PUBLIC names relocated after alphabet constants; procedures unchanged.",
    "cdf_aux_mod": (
        "Nine private constants use inline PRIVATE attributes; values unchanged. "
        "CRASH_LEFT/CRASH_HI local declaration order changes; call argument order unchanged."
    ),
    "cdf_beta_mod": "Comments, whitespace and redundant PUBLIC; statements unchanged.",
    "cdf_binomial_mod": (
        "Inverse chance branches exit before complement reconstruction. Final PR/CPR writes "
        "move inside the fourth branch only: the first three leave outputs unwritten. "
        "Python uses the validated beta inversion; native outputs are not correctness oracles."
    ),
    "cdf_chisq_mod": "Redundant INTRINSIC PRESENT removed; no overriding declaration.",
    "cdf_f_mod": "Private TINY declaration relocated; redundant INTRINSIC PRESENT removed.",
    "cdf_gamma_mod": "Byte-identical source.",
    "cdf_nc_chisq_mod": "PUBLIC declaration relocated; PRESENT removed from INTRINSIC list.",
    "cdf_nc_f_mod": "PRESENT removed from INTRINSIC list; no overriding declaration.",
    "cdf_nc_t_mod": "PRESENT removed from INTRINSIC list; no overriding declaration.",
    "cdf_neg_binomial_mod": "Local BETA_STATUS declaration relocated before INTRINSIC PRESENT.",
    "cdf_normal_mod": "Comments, whitespace and redundant PUBLIC; statements unchanged.",
    "cdf_poisson_mod": "Comments, whitespace and redundant PUBLIC; statements unchanged.",
    "cdf_t_mod": "Redundant INTRINSIC PRESENT removed; no overriding declaration.",
    "zero_finder": (
        "Public type/data names move from a standalone list to inline PUBLIC attributes; "
        "same twelve exports. State layout and executable statements unchanged."
    ),
}


def statements(text):
    """Extract free-form logical statements, retaining literals and declaration order."""
    records, buffer, quote, continuation = [], "", None, False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if continuation and line.startswith("&"):
            line = line[1:]
        i, continued = 0, False
        while i < len(line):
            char = line[i]
            # A trailing ampersand can continue a literal or an ordinary statement.
            suffix = line[i + 1 :].lstrip()
            if char == "&" and (not suffix or (quote is None and suffix.startswith("!"))):
                continued = True
                break
            if quote is not None:
                buffer += char
                if char == quote:
                    if i + 1 < len(line) and line[i + 1] == quote:
                        buffer += quote
                        i += 1
                    else:
                        quote = None
            elif char == "!":
                break
            elif char in "'\"":
                quote = char
                buffer += char
            elif char == ";":
                if buffer:
                    records.append(buffer)
                buffer = ""
            elif not char.isspace():
                buffer += char.upper()
            i += 1
        continuation = continued
        if not continuation:
            if quote is not None:
                raise ValueError("Unterminated Fortran literal")
            if buffer:
                records.append(buffer)
            buffer = ""
    if buffer or quote or continuation:
        raise ValueError("Unfinished Fortran statement")
    return [r for r in records if r not in ("IMPLICITNONE", "PUBLIC")]


def main():
    contents = read_archive()
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != CDFLIB_SHA256:
        raise RuntimeError("CDFLIB90 archive changed")
    with tarfile.open(archive) as tar:
        cdf = {
            Path(m.name).stem: tar.extractfile(m).read()
            for m in tar.getmembers()
            if m.isfile() and m.name.startswith(F95_ROOT + "source/") and m.name.endswith(".f90")
        }
    if set(cdf) != set(REVIEWS):
        raise RuntimeError("Shared module review coverage changed")
    records = []
    for module, review in sorted(REVIEWS.items()):
        before, after = cdf[module], contents[SOURCE_ROOT + module + ".f90"]
        a, b = statements(before.decode("latin1")), statements(after.decode("latin1"))
        records.append(
            dict(
                module=module,
                cdflib_sha256=hashlib.sha256(before).hexdigest(),
                stattab_sha256=hashlib.sha256(after).hexdigest(),
                byte_identical=before == after,
                logical_statements_equal=a == b,
                review=review,
                logical_diff=list(difflib.unified_diff(a, b, n=2, lineterm="")),
            )
        )
    prior = json.loads(Path("docs/cdflib90-completion.json").read_text())
    if prior["archive_sha256"] != CDFLIB_SHA256:
        raise RuntimeError("CDFLIB90 completion provenance differs")
    exports = prior["f95_modules"]
    if set(exports) != set(cdf):
        raise RuntimeError("CDFLIB90 export coverage changed")
    imports = []
    for module, names in exports.items():
        combined = set(names)
        for extra in prior["transitive_exports"].get(module, {}).values():
            combined.update(extra)
        imports.extend((module, name) for name in sorted(combined))
    shared_count = len(imports)
    additional = {
        "biomath_file_io_mod": ["open_file", "report_file_dialogue"],
        "stattab_aux_mod": [
            "ndist",
            "one_parameter",
            "the_distribution",
            "distributions",
            "display_banner",
        ],
    }
    imports.extend((m, n) for m, names in additional.items() for n in names)
    driver = (
        "program exports\n"
        + "\n".join(
            f"use {module}, only: audited_{i} => {name}" for i, (module, name) in enumerate(imports)
        )
        + "\nimplicit none\nend program exports\n"
    )
    # Cover all combinations of matching the smaller tail and varying PR/CPR.
    cases = [(0, 0.25), (0, 0.75), (3, 0.25), (3, 0.75)]
    probe = """program probe
use biomath_constants_mod, only: dpkind
use cdf_binomial_mod, only: cdf_binomial
implicit none
real(dpkind) :: s,n,p,q,c,cc
integer :: status
"""
    expected = []
    for s, p in cases:
        c = sum(math.comb(4, k) * p**k * (1 - p) ** (4 - k) for k in range(s + 1))
        expected.append(dict(s=s, n=4, target_pr=p, cum=c, ccum=1 - c))
        probe += f"s={s}; n=4; p=-123; q=-456; c={c}_dpkind; cc={1 - c}_dpkind\n"
        probe += "call cdf_binomial(4,c,cc,s,n,p,q,status=status)\nwrite(*,*) status,p,q\n"
    probe += "end program probe\n"
    formats = {
        "format_old": (
            "program formats\nwrite(*,'(/5X''A blank line is not allowed.'')')\nend program\n"
        ),
        "format_new": (
            "program formats\nwrite(*,'(/5X,\"A blank line is not allowed.\")')\nend program\n"
        ),
    }
    flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
    order = [n for n in source_order(contents) if n != "stattab_main.f90"]
    with tempfile.TemporaryDirectory(prefix="stattab-shared-") as tmp:
        work = Path(tmp)
        for name in order:
            (work / name).write_bytes(contents[SOURCE_ROOT + name])
        subprocess.run(
            [*flags, "-c", *order], cwd=work, check=True, capture_output=True, timeout=60
        )
        objects = [str(Path(n).with_suffix(".o")) for n in order]
        outputs = {}
        for name, source in [("exports", driver), ("probe", probe), *formats.items()]:
            (work / (name + ".f90")).write_text(source)
            subprocess.run(
                [*flags, name + ".f90", *objects, "-o", name],
                cwd=work,
                check=True,
                capture_output=True,
                timeout=60,
            )
            outputs[name] = subprocess.check_output([str(work / name)], text=True, timeout=3)
    if outputs["format_old"] != outputs["format_new"]:
        raise RuntimeError("Blank-input warning formats differ")
    rows = outputs["probe"].splitlines()
    if len(rows) != len(expected):
        raise RuntimeError("Unexpected native probe output")
    for row, record in zip(rows, expected, strict=True):
        status, pr, cpr = row.split()
        record.update(native_status=int(status), native_pr=float(pr), native_cpr=float(cpr))
    report = dict(
        archive_sha256=ARCHIVE_SHA256,
        cdflib_archive_sha256=CDFLIB_SHA256,
        status="shared_source_review_complete_application_manual_audit_pending",
        comparison_limits=(
            "Logical diffs preserve literals, executable statements and declaration order; "
            "omit comments, unquoted whitespace, case, standalone PUBLIC and IMPLICIT NONE. "
            "They aid source review, not semantic or numerical proof. Compiler imports prove "
            "accessibility of listed names, not exhaustive export discovery or correctness. "
            "Use existing independent numerical and behavioral tests for correctness."
        ),
        shared_modules=records,
        shared_import_count=shared_count,
        additional_exports=additional,
        compiler_import_count=len(imports),
        compiler_flags=flags,
        compiler_source_order=order,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        import_driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
        binomial_probe_source=probe,
        binomial_probe_results=expected,
        blank_warning_format_probe=formats,
        blank_warning_formats_equal=True,
        evidence=[
            "docs/cdflib90-completion.json",
            "tests/test_stattab_results.py",
            "tests/test_stattab_reference.py",
            "tests/test_stattab_console.py",
            "tests/test_stattab_files.py",
        ],
    )
    Path("docs/stattab-shared-source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                shared_modules=len(records),
                shared_imports=shared_count,
                all_imports=len(imports),
                binomial_probe_results=expected,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
