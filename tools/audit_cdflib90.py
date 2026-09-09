"""Inventory every pinned CDFLIB90 archive member and preserve unresolved API scope."""

import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

ARCHIVE_SHA256 = "2f5dd397b93546222a3b31e02073abeee1fc213cea75768c34b17e06c8264a3b"
F95_ROOT = "CDFLIB90/source/cdflib90_1.2/"
LEGACY_QUANTILE_NAMES = {"stvaln", "dinvnr", "dt1"}
LEGACY_MATH_NAMES = {
    "esum",
    "gsumln",
    "rcomp",
    "grat1",
    "alnrel",
    "fpser",
    "rlog",
    "bfrac",
    "rlog1",
    "erfc1",
    "brcmp1",
    "bup",
    "basym",
    "algdiv",
    "brcomp",
    "gamln",
    "bpser",
    "erf1",
    "gam1",
    "betaln",
    "apser",
    "bratio",
    "gamln1",
    "Xgamm",
    "bcorr",
    "exparg",
    "psi",
    "gratio",
    "bgrat",
    "rexp",
    "alngam",
}
LEGACY_MATH_F77_NAMES = (LEGACY_MATH_NAMES - {"Xgamm", "erf1"}) | {"gamma", "erf"}
DISTRIBUTIONS = {
    "beta": "bet",
    "binomial": "bin",
    "chisq": "chi",
    "f": "f",
    "gamma": "gam",
    "nc_chisq": "chn",
    "nc_f": "fnc",
    "nc_t": "tnc",
    "neg_binomial": "nbn",
    "normal": "nor",
    "poisson": "poi",
    "t": "t",
}
IMPLEMENTED = {
    "beta",
    "binomial",
    "chisq",
    "f",
    "gamma",
    "nc_chisq",
    "nc_f",
    "nc_t",
    "neg_binomial",
    "normal",
    "poisson",
    "t",
}
SUPPORT = {
    "biomath_constants_mod": "Kind declarations and numeric constants",
    "biomath_interface_mod": "Public console input/output and message controls",
    "biomath_mathlib_mod": "Default-public mathematical functions and numerical kernels",
    "biomath_sort_mod": "Public sort_list generic and character/real/integer overloads",
    "biomath_strings_mod": "Public case conversion and stateful command-language lexer",
    "cdf_aux_mod": "Default-public validation, distribution metadata and solver adapters",
    "zero_finder": "Public direct/reverse-communication root finding and solver state",
}


def fortran_inventory(content: str, *, fixed: bool) -> dict:
    lines = [line for line in content.splitlines() if not (fixed and line and line[0] in "cC*!")]
    source = "\n".join(line.split("!", 1)[0] for line in lines)
    source = re.sub(r"&\s*\n\s*&?", " ", source)
    routines = sorted(
        set(
            re.findall(
                r"^\s*(?:(?:DOUBLE\s+PRECISION|INTEGER|REAL|LOGICAL|CHARACTER)(?:\s*\([^\n]*?\))?\s+)?"
                r"(?:FUNCTION|SUBROUTINE|ENTRY)\s+(\w+)",
                source,
                re.M | re.I,
            )
        )
    )
    public = sorted(
        {
            name.strip().lower()
            for declaration in re.findall(r"^\s*PUBLIC\s*::\s*([^\n]+)", source, re.M | re.I)
            for name in declaration.split(",")
        }
    )
    default = re.search(r"^\s*(PUBLIC|PRIVATE)\s*$", source, re.M | re.I)
    return {
        "routine_declarations_including_entries": [name.lower() for name in routines],
        "explicit_public_names": public,
        "first_default_access_statement": default[1].lower() if default else None,
    }


def c_inventory(content: str) -> dict:
    declarations = re.findall(
        r"^(?:(static|extern)\s+)?(?:void|double|int|long)\s+(\w+)\s*\(", content, re.M
    )
    return {
        "external_function_names": sorted(
            {name for access, name in declarations if access != "static"}
        ),
        "static_function_names": sorted(
            {name for access, name in declarations if access == "static"}
        ),
    }


def classify(path: Path) -> tuple[str, str, str]:
    name = path.as_posix()
    if name.startswith(F95_ROOT + "source/"):
        stem = path.stem
        if path.name == "#cdf_binomial_mod.f90#":
            return (
                "source_backup",
                "alternate_source_reconciled",
                "Backup probability-assignment defects validated; primary source is authoritative",
            )
        if path.name in ("Makefile", "compile.cdflib90"):
            return "build", "build_replaced", "pyproject.toml, uv build and existing CI"
        if stem == "biomath_sort_mod" and path.suffix == ".f90":
            return (
                "f95_support",
                "implemented_public_support",
                "sort_list: four overloads, custom comparison, duplicate and long-string repairs",
            )
        if stem == "biomath_strings_mod" and path.suffix == ".f90":
            return (
                "f95_support",
                "implemented_public_support",
                "ASCII conversion and reentrant lexer with explicit token/numeric repairs",
            )
        if stem in SUPPORT and path.suffix == ".f90":
            return "f95_support", "public_support_review", SUPPORT[stem]
        if stem.startswith("cdf_") and stem.endswith("_mod"):
            distribution = stem[4:-4]
            if distribution not in DISTRIBUTIONS:
                raise RuntimeError(f"unknown distribution module: {path}")
            status = (
                "implemented_distribution"
                if distribution in IMPLEMENTED
                else "pending_distribution"
            )
            return "f95_distribution", status, distribution
    if name.startswith(F95_ROOT) and path.name in {
        "cdflib90_doc.pdf",
        "cdflib90_doc.ps",
        "cdflib90_doc.tex",
        "cdflib90_doc.txt",
        "DOC.TEX",
        "INSTALL",
        "LEGALITIES",
        "readme",
    }:
        return (
            "documentation",
            "reference_documentation",
            "F95 methods, installation and retained legal terms",
        )
    if name.startswith(("source/dcdflib.c/", "source/dcdflib.f/")):
        if "/doc/" in name or path.name in ("HOWTOGET", "readme"):
            return (
                "documentation",
                "reference_documentation",
                "DCDFLIB 1.1 legacy contracts and notices; compare with F95",
            )
        if path.suffix == ".f":
            if path.stem == "gaminv":
                return (
                    "f77_source",
                    "implemented_legacy_support",
                    "Validated inverse gamma with checked hints and numerical repairs",
                )
            if path.stem in LEGACY_QUANTILE_NAMES:
                return (
                    "f77_source",
                    "implemented_legacy_support",
                    "Validated normal inverse and normal/t starting approximations",
                )
            if path.stem in LEGACY_MATH_F77_NAMES:
                return (
                    "f77_source",
                    "implemented_legacy_support",
                    "Validated mathematical mapping; see dcdflib-math.md",
                )
            if path.stem in {"ipmpar", "spmpar", "devlpl"}:
                return (
                    "f77_source",
                    "implemented_legacy_support",
                    "Validated machine parameters and polynomial prefix mapping; "
                    "see legacy support ledger",
                )
            if path.stem in {"cdftnc", "cumtnc"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Signed noncentral t tails and inversions; independent C/F77 validation",
                )
            if path.stem in {"cdfchn", "cumchn"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Noncentral chi-square tails and inversions; independent C/F77 validation",
                )
            if path.stem in {"cdfbet", "cumbet"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Beta tails and inversions; independent C/F77 validation",
                )
            if path.stem in {"cdfbin", "cumbin"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Binomial tails and inversions; independent C/F77 validation",
                )
            if path.stem in {"cdfnbn", "cumnbn"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Negative-binomial tails and inversions; independent C/F77 validation",
                )
            if path.stem in {"cdfpoi", "cumpoi"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Poisson tails and inversions; independent C/F77 validation",
                )
            if path.stem in {"cdfchi", "cumchi"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Chi-square tails and inversions; independent C/F77 validation",
                )
            if path.stem in {"cdfgam", "cumgam"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Gamma tails and all parameter inversions; independent C/F77 validation",
                )
            if path.stem in {"cdft", "cumt"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Student t tails and inversions; independent C/F77 validation",
                )
            if path.stem in {"cdfnor", "cumnor"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "Normal tails and all parameter inversions; independent C/F77 validation",
                )
            if path.stem in {"cdff", "cumf", "cdffnc", "cumfnc"}:
                return (
                    "f77_source",
                    "implemented_legacy_distribution",
                    "F/noncentral F tails and inversions; independent C/F77 validation",
                )
            return (
                "f77_source",
                "legacy_contract_review",
                "Legacy entry point; not independently validated by F95 fixtures",
            )
        if path.suffix == ".c":
            if path.name == "ipmpar.c":
                return (
                    "c_source",
                    "implemented_legacy_support",
                    "Validated int32/IEEE machine-parameter mapping",
                )
            return (
                "c_source",
                "legacy_contract_review",
                "Legacy C implementation; retain header-to-definition inventory",
            )
        if path.suffix == ".h":
            return (
                "c_header",
                "legacy_contract_review",
                "Public C prototypes, including numerical helpers and solver setup",
            )
    raise RuntimeError(f"unclassified archive member: {path}")


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    extraction = Path("research/raw/CDFLIB90/source")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError("archive changed; review the inventory before updating its pinned hash")
    members, contents, directories = [], {}, []
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isdir():
                directories.append(member.name)
                continue
            if not member.isfile():
                raise RuntimeError(f"unexpected nonregular member: {member.name}")
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.name in contents:
                raise RuntimeError(f"unsafe or duplicate member: {member.name}")
            handle = tar.extractfile(member)
            if handle is None:
                raise RuntimeError(f"unreadable member: {member.name}")
            content = handle.read()
            if (extraction / path).read_bytes() != content:
                raise RuntimeError(f"extracted content differs: {member.name}")
            contents[member.name] = content
            role, status, scope = classify(path)
            row = {
                "path": member.name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "role": role,
                "coverage_status": status,
                "scope": scope,
            }
            if role in ("f95_distribution", "f95_support", "source_backup", "f77_source"):
                row["declarations"] = fortran_inventory(
                    content.decode("latin-1"), fixed=role == "f77_source"
                )
            elif role in ("c_source", "c_header"):
                row["declarations"] = c_inventory(content.decode("latin-1"))
            members.append(row)
    if len(members) != 106:
        raise RuntimeError("regular-file inventory changed")
    distributions = []
    for distribution, legacy in DISTRIBUTIONS.items():
        path = F95_ROOT + f"source/cdf_{distribution}_mod.f90"
        row = next(row for row in members if row["path"] == path)
        public = row["declarations"]["explicit_public_names"]
        if public != sorted(f"{prefix}_{distribution}" for prefix in ("cdf", "cum", "ccum", "inv")):
            raise RuntimeError(f"unexpected public distribution interface: {distribution}")
        evidence = (
            [
                f"src/mdanderson_stats/cdflib_{distribution}.py",
                f"tests/test_cdflib_{distribution}.py",
                f"tests/fixtures/cdflib_{distribution}.json",
            ]
            if distribution in IMPLEMENTED
            else []
        )
        if any(not Path(path).is_file() for path in evidence):
            raise RuntimeError(f"missing implementation evidence: {distribution}")
        distributions.append(
            {
                "name": distribution,
                "f95_public_interfaces": public,
                "python_status": row["coverage_status"],
                "evidence": evidence,
                "legacy_distribution_names": ["cdf" + legacy, "cum" + legacy],
                "legacy_contract_status": "not_yet_independently_validated",
            }
        )
    f_distribution = next(row for row in distributions if row["name"] == "f")
    f_distribution["f95_computed_groups"] = ["cum/ccum", "f"]
    f_distribution["legacy_additional_computed_groups_pending"] = []
    f_distribution["legacy_computed_groups"] = ["p/q", "f", "dfn", "dfd"]
    f_distribution["legacy_python_interfaces"] = ["cdff", "cumf"]
    f_distribution["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    f_distribution["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_f.py",
        "tests/test_dcdflib_f.py",
        "tests/fixtures/dcdflib_f.json",
        "docs/dcdflib-f.md",
    ]
    if any(not Path(path).is_file() for path in f_distribution["legacy_evidence"]):
        raise RuntimeError("missing legacy F implementation evidence")
    noncentral_f = next(row for row in distributions if row["name"] == "nc_f")
    noncentral_f["f95_computed_groups"] = ["cum/ccum", "f", "pnonc"]
    noncentral_f["legacy_additional_computed_groups_pending"] = []
    noncentral_f["legacy_computed_groups"] = ["p/q", "f", "dfn", "dfd", "pnonc"]
    noncentral_f["legacy_python_interfaces"] = ["cdffnc", "cumfnc"]
    noncentral_f["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    noncentral_f["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_nc_f.py",
        "tests/test_dcdflib_nc_f.py",
        "tests/fixtures/dcdflib_nc_f.json",
        "docs/dcdflib-nc-f.md",
    ]
    if any(not Path(path).is_file() for path in noncentral_f["legacy_evidence"]):
        raise RuntimeError("missing legacy noncentral F implementation evidence")
    noncentral_f["legacy_pnonc_which"] = 5
    normal = next(row for row in distributions if row["name"] == "normal")
    normal["legacy_computed_groups"] = ["p/q", "x", "mean", "sd"]
    normal["legacy_python_interfaces"] = ["cdfnor", "cumnor"]
    normal["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    normal["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_normal.py",
        "tests/test_dcdflib_normal.py",
        "tests/fixtures/dcdflib_normal.json",
        "docs/dcdflib-normal.md",
    ]
    if any(not Path(path).is_file() for path in normal["legacy_evidence"]):
        raise RuntimeError("missing legacy normal implementation evidence")
    student_t = next(row for row in distributions if row["name"] == "t")
    student_t["legacy_computed_groups"] = ["p/q", "t", "df"]
    student_t["legacy_python_interfaces"] = ["cdft", "cumt"]
    student_t["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    student_t["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_t.py",
        "tests/test_dcdflib_t.py",
        "tests/fixtures/dcdflib_t.json",
        "docs/dcdflib-t.md",
    ]
    if any(not Path(path).is_file() for path in student_t["legacy_evidence"]):
        raise RuntimeError("missing legacy Student t implementation evidence")
    gamma = next(row for row in distributions if row["name"] == "gamma")
    gamma["legacy_computed_groups"] = ["p/q", "x", "shape", "rate"]
    gamma["legacy_python_interfaces"] = ["cdfgam", "cumgam"]
    gamma["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    gamma["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_gamma.py",
        "tests/test_dcdflib_gamma.py",
        "tests/fixtures/dcdflib_gamma.json",
        "docs/dcdflib-gamma.md",
    ]
    if any(not Path(path).is_file() for path in gamma["legacy_evidence"]):
        raise RuntimeError("missing legacy gamma implementation evidence")
    chisq = next(row for row in distributions if row["name"] == "chisq")
    chisq["legacy_computed_groups"] = ["p/q", "x", "df"]
    chisq["legacy_python_interfaces"] = ["cdfchi", "cumchi"]
    chisq["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    chisq["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_chisq.py",
        "tests/test_dcdflib_chisq.py",
        "tests/fixtures/dcdflib_chisq.json",
        "docs/dcdflib-chisq.md",
    ]
    if any(not Path(path).is_file() for path in chisq["legacy_evidence"]):
        raise RuntimeError("missing legacy chi-square implementation evidence")
    poisson = next(row for row in distributions if row["name"] == "poisson")
    poisson["legacy_computed_groups"] = ["p/q", "s", "mean"]
    poisson["legacy_python_interfaces"] = ["cdfpoi", "cumpoi"]
    poisson["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    poisson["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_poisson.py",
        "tests/test_dcdflib_poisson.py",
        "tests/fixtures/dcdflib_poisson.json",
        "docs/dcdflib-poisson.md",
    ]
    if any(not Path(path).is_file() for path in poisson["legacy_evidence"]):
        raise RuntimeError("missing legacy Poisson implementation evidence")
    neg_binomial = next(row for row in distributions if row["name"] == "neg_binomial")
    neg_binomial["legacy_reference_evidence"] = [
        "tools/reference_dcdflib_neg_binomial.py",
        "tests/fixtures/dcdflib_neg_binomial.json",
        "tests/test_dcdflib_neg_binomial_reference.py",
        "docs/dcdflib-neg-binomial-reference.md",
    ]
    neg_binomial["legacy_computed_groups"] = ["p/q", "f", "s", "pr/cpr"]
    neg_binomial["legacy_python_interfaces"] = ["cdfnbn", "cumnbn"]
    neg_binomial["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    neg_binomial["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_neg_binomial.py",
        "tests/test_dcdflib_neg_binomial.py",
        "tests/fixtures/dcdflib_neg_binomial.json",
        "docs/dcdflib-neg-binomial.md",
    ]
    if any(not Path(path).is_file() for path in neg_binomial["legacy_evidence"]):
        raise RuntimeError("missing legacy negative-binomial implementation evidence")
    if any(not Path(path).is_file() for path in neg_binomial["legacy_reference_evidence"]):
        raise RuntimeError("missing legacy negative-binomial reference evidence")
    nc_t = next(row for row in distributions if row["name"] == "nc_t")
    nc_t["legacy_reference_evidence"] = [
        "tools/reference_dcdflib_nc_t.py",
        "tests/fixtures/dcdflib_nc_t.json",
        "tests/test_dcdflib_nc_t_reference.py",
        "docs/dcdflib-nc-t-reference.md",
    ]
    nc_t["legacy_review_notes"] = (
        "Unchanged C/F77 contracts audited: signed noncentrality, ignored q, executable "
        "df cap 1e4 versus header 1e10, misleading failure bounds and false-success repairs."
    )
    nc_t["legacy_computed_groups"] = ["p/q", "t", "df", "pnonc"]
    nc_t["legacy_python_interfaces"] = ["cdftnc", "cumtnc"]
    nc_t["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    nc_t["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_nc_t.py",
        "src/mdanderson_stats/_dcdflib_nc_t.py",
        "tests/test_dcdflib_nc_t.py",
        "tests/fixtures/dcdflib_nc_t.json",
        "docs/dcdflib-nc-t.md",
    ]
    if any(not Path(path).is_file() for path in nc_t["legacy_evidence"]):
        raise RuntimeError("missing legacy noncentral t implementation evidence")
    if any(not Path(path).is_file() for path in nc_t["legacy_reference_evidence"]):
        raise RuntimeError("missing legacy noncentral t reference evidence")
    nc_chisq = next(row for row in distributions if row["name"] == "nc_chisq")
    nc_chisq["legacy_reference_evidence"] = [
        "tools/reference_dcdflib_nc_chisq.py",
        "tests/fixtures/dcdflib_nc_chisq.json",
        "tests/test_dcdflib_nc_chisq_reference.py",
        "docs/dcdflib-nc-chisq-reference.md",
    ]
    nc_chisq["legacy_computed_groups"] = ["p/q", "x", "df", "pnonc"]
    nc_chisq["legacy_python_interfaces"] = ["cdfchn", "cumchn"]
    nc_chisq["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    nc_chisq["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_nc_chisq.py",
        "tests/test_dcdflib_nc_chisq.py",
        "tests/fixtures/dcdflib_nc_chisq.json",
        "docs/dcdflib-nc-chisq.md",
    ]
    if any(not Path(path).is_file() for path in nc_chisq["legacy_evidence"]):
        raise RuntimeError("missing legacy noncentral chi-square implementation evidence")
    if any(not Path(path).is_file() for path in nc_chisq["legacy_reference_evidence"]):
        raise RuntimeError("missing legacy noncentral chi-square reference evidence")
    beta = next(row for row in distributions if row["name"] == "beta")
    beta["legacy_reference_evidence"] = [
        "tools/reference_dcdflib_beta.py",
        "tests/fixtures/dcdflib_beta.json",
        "tests/test_dcdflib_beta_reference.py",
        "docs/dcdflib-beta-reference.md",
    ]
    beta["legacy_computed_groups"] = ["p/q", "x/cx", "a", "b"]
    beta["legacy_python_interfaces"] = ["cdfbet", "cumbet"]
    beta["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    beta["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_beta.py",
        "tests/test_dcdflib_beta.py",
        "tests/fixtures/dcdflib_beta.json",
        "docs/dcdflib-beta.md",
    ]
    if any(not Path(path).is_file() for path in beta["legacy_evidence"]):
        raise RuntimeError("missing legacy beta implementation evidence")
    if any(not Path(path).is_file() for path in beta["legacy_reference_evidence"]):
        raise RuntimeError("missing legacy beta reference evidence")
    binomial = next(row for row in distributions if row["name"] == "binomial")
    binomial["legacy_reference_evidence"] = [
        "tools/reference_dcdflib_binomial.py",
        "tests/fixtures/dcdflib_binomial.json",
        "tests/test_dcdflib_binomial_reference.py",
        "docs/dcdflib-binomial-reference.md",
    ]
    binomial["legacy_computed_groups"] = ["p/q", "s", "n", "pr/cpr"]
    binomial["legacy_python_interfaces"] = ["cdfbin", "cumbin"]
    binomial["legacy_contract_status"] = "implemented_with_documented_python_semantics"
    binomial["legacy_evidence"] = [
        "src/mdanderson_stats/dcdflib_binomial.py",
        "tests/test_dcdflib_binomial.py",
        "tests/fixtures/dcdflib_binomial.json",
        "docs/dcdflib-binomial.md",
    ]
    if any(not Path(path).is_file() for path in binomial["legacy_evidence"]):
        raise RuntimeError("missing legacy binomial implementation evidence")
    if any(not Path(path).is_file() for path in binomial["legacy_reference_evidence"]):
        raise RuntimeError("missing legacy binomial reference evidence")
    header = next(row for row in members if row["role"] == "c_header")["declarations"]
    definitions = {
        name
        for row in members
        if row["role"] == "c_source"
        for name in row["declarations"]["external_function_names"]
    }
    f77_names = {
        name
        for row in members
        if row["role"] == "f77_source"
        for name in row["declarations"]["routine_declarations_including_entries"]
    }
    for distribution in distributions:
        names = set(distribution["legacy_distribution_names"])
        if not names <= f77_names or not names <= set(header["external_function_names"]):
            raise RuntimeError(f"missing legacy distribution declarations: {distribution['name']}")
    support_interfaces = [
        {
            "module": "biomath_sort_mod",
            "public_names": ["sort_list"],
            "python_interfaces": ["sort_list"],
            "status": "implemented_with_documented_python_semantics",
            "evidence": [
                "src/mdanderson_stats/cdflib_sort.py",
                "tools/reference_cdflib_sort.py",
                "tests/fixtures/cdflib_sort.json",
                "tests/test_cdflib_sort.py",
                "docs/cdflib-sort.md",
                "docs/cdflib-sort-benchmark.json",
            ],
        }
    ]
    support_interfaces.append(
        {
            "module": "biomath_strings_mod",
            "public_names": [
                "lower_case_char",
                "lower_case_string",
                "qlex",
                "upper_case_char",
                "upper_case_string",
            ],
            "python_interfaces": [
                "lower_case_char",
                "lower_case_string",
                "qlex",
                "upper_case_char",
                "upper_case_string",
            ],
            "status": "implemented_with_documented_python_semantics",
            "evidence": [
                "tools/reference_cdflib_strings.py",
                "tests/fixtures/cdflib_strings.json",
                "tests/test_cdflib_strings_reference.py",
                "docs/cdflib-strings-reference.md",
                "src/mdanderson_stats/cdflib_strings.py",
                "tests/test_cdflib_strings.py",
                "docs/cdflib-strings.md",
                "docs/cdflib-strings-benchmark.json",
            ],
        }
    )
    constants = json.loads(Path("tests/fixtures/cdflib_constants.json").read_text())
    constant_evidence = [
        "src/mdanderson_stats/cdflib_constants.py",
        "tools/reference_cdflib_constants.py",
        "tests/fixtures/cdflib_constants.json",
        "tests/test_cdflib_constants.py",
        "docs/cdflib-constants.md",
    ]
    support_interfaces.append(
        {
            "module": "biomath_constants_mod",
            "public_names": constants["integer_names"] + constants["real_names"],
            "python_interfaces": ["cdflib_constants"],
            "status": "implemented_with_documented_python_semantics",
            "evidence": constant_evidence,
        }
    )
    support_interfaces.append(
        {
            "module": "biomath_mathlib_mod",
            "implemented_public_names": [
                "alnrel",
                "rexp",
                "rlog",
                "rlog1",
                "evaluate_polynomial",
                "erf",
                "erfc1",
                "esum",
                "exparg",
                "alngam",
                "gamln",
                "log_gamma",
                "gamln1",
                "gam1",
                "gamma",
                "psi",
                "algdiv",
                "bcorr",
                "gsumln",
                "betaln",
                "log_beta",
                "log_bicoef",
                "rcomp",
                "grat1",
                "gratio",
                "brcomp",
                "brcmp1",
                "bup",
                "fpser",
                "apser",
                "bpser",
                "bgrat",
                "basym",
                "bfrac",
                "bratio",
            ],
            "python_interfaces": [
                "alnrel",
                "rexp",
                "rlog",
                "rlog1",
                "evaluate_polynomial",
                "erf",
                "erfc1",
                "esum",
                "exparg",
                "alngam",
                "gamln",
                "log_gamma",
                "gamln1",
                "gam1",
                "gamma",
                "psi",
                "algdiv",
                "bcorr",
                "gsumln",
                "betaln",
                "log_beta",
                "log_bicoef",
                "rcomp",
                "grat1",
                "gratio",
                "brcomp",
                "brcmp1",
                "bup",
                "fpser",
                "apser",
                "bpser",
                "bgrat",
                "basym",
                "bfrac",
                "bratio",
            ],
            "status": "implemented_with_documented_python_semantics",
            "imported_constant_module": "biomath_constants_mod",
            "python_constant_namespace": "cdflib_constants",
            "reference_audited_public_names": [
                "erf",
                "erfc1",
                "esum",
                "exparg",
                "alngam",
                "gamln",
                "log_gamma",
                "gamln1",
                "gam1",
                "gamma",
                "psi",
                "algdiv",
                "bcorr",
                "betaln",
                "log_beta",
                "gsumln",
                "log_bicoef",
                "basym",
                "bfrac",
                "bgrat",
                "bratio",
            ],
            "evidence": [
                *constant_evidence,
                "src/mdanderson_stats/cdflib_elementary.py",
                "tools/reference_cdflib_elementary.py",
                "tests/fixtures/cdflib_elementary.json",
                "tests/test_cdflib_elementary.py",
                "docs/cdflib-elementary.md",
                "docs/cdflib-elementary-benchmark.json",
                "tools/reference_cdflib_error_exponential.py",
                "tests/fixtures/cdflib_error_exponential.json",
                "tests/test_cdflib_error_exponential_reference.py",
                "docs/cdflib-error-exponential-reference.md",
                "src/mdanderson_stats/cdflib_error_exponential.py",
                "tests/test_cdflib_error_exponential.py",
                "docs/cdflib-error-exponential.md",
                "docs/cdflib-error-exponential-benchmark.json",
                "tools/reference_cdflib_gamma_support.py",
                "tests/fixtures/cdflib_gamma_support.json",
                "tests/test_cdflib_gamma_support_reference.py",
                "docs/cdflib-gamma-support-reference.md",
                "src/mdanderson_stats/cdflib_gamma_support.py",
                "tests/test_cdflib_gamma_support.py",
                "docs/cdflib-gamma-support.md",
                "docs/cdflib-gamma-support-benchmark.json",
                "tools/reference_cdflib_beta_support.py",
                "tests/fixtures/cdflib_beta_support.json",
                "tests/test_cdflib_beta_support_reference.py",
                "docs/cdflib-beta-support-reference.md",
                "src/mdanderson_stats/cdflib_gamma_ratios.py",
                "tests/test_cdflib_gamma_ratios.py",
                "docs/cdflib-gamma-ratios.md",
                "docs/cdflib-gamma-ratios-benchmark.json",
                "src/mdanderson_stats/cdflib_beta_support.py",
                "tests/test_cdflib_beta_support.py",
                "docs/cdflib-beta-support.md",
                "docs/cdflib-beta-support-benchmark.json",
                "tools/reference_cdflib_incomplete_gamma.py",
                "tests/fixtures/cdflib_incomplete_gamma.json",
                "tests/test_cdflib_incomplete_gamma_reference.py",
                "docs/cdflib-incomplete-gamma-reference.md",
                "src/mdanderson_stats/cdflib_gamma_factor.py",
                "tests/test_cdflib_gamma_factor.py",
                "docs/cdflib-gamma-factor.md",
                "docs/cdflib-gamma-factor-benchmark.json",
                "src/mdanderson_stats/cdflib_incomplete_gamma.py",
                "tests/test_cdflib_incomplete_gamma.py",
                "docs/cdflib-incomplete-gamma.md",
                "docs/cdflib-incomplete-gamma-benchmark.json",
                "tools/reference_cdflib_beta_factors.py",
                "tests/fixtures/cdflib_beta_factors.json",
                "tests/test_cdflib_beta_factors_reference.py",
                "docs/cdflib-beta-factors-reference.md",
                "src/mdanderson_stats/cdflib_beta_factors.py",
                "tests/test_cdflib_beta_factors.py",
                "docs/cdflib-beta-factors.md",
                "docs/cdflib-beta-factors-benchmark.json",
                "src/mdanderson_stats/cdflib_beta_shift.py",
                "tests/test_cdflib_beta_shift.py",
                "docs/cdflib-beta-shift.md",
                "docs/cdflib-beta-shift-benchmark.json",
                "tools/reference_cdflib_beta_series.py",
                "tests/fixtures/cdflib_beta_series.json",
                "tests/test_cdflib_beta_series_reference.py",
                "docs/cdflib-beta-series-reference.md",
                "src/mdanderson_stats/cdflib_beta_series.py",
                "tests/test_cdflib_fpser.py",
                "docs/cdflib-fpser.md",
                "docs/cdflib-fpser-benchmark.json",
                "tests/test_cdflib_apser.py",
                "docs/cdflib-apser.md",
                "docs/cdflib-apser-benchmark.json",
                "tests/test_cdflib_bpser.py",
                "docs/cdflib-bpser.md",
                "docs/cdflib-bpser-benchmark.json",
                "tests/test_cdflib_bgrat.py",
                "docs/cdflib-bgrat.md",
                "docs/cdflib-bgrat-benchmark.json",
                "src/mdanderson_stats/cdflib_beta_asymptotic.py",
                "tests/test_cdflib_basym.py",
                "docs/cdflib-basym.md",
                "docs/cdflib-basym-benchmark.json",
                "src/mdanderson_stats/cdflib_beta_fraction.py",
                "tests/test_cdflib_bfrac.py",
                "docs/cdflib-bfrac.md",
                "docs/cdflib-bfrac-benchmark.json",
                "src/mdanderson_stats/cdflib_beta_ratio.py",
                "tests/test_cdflib_bratio.py",
                "docs/cdflib-bratio.md",
                "docs/cdflib-bratio-benchmark.json",
                "tools/reference_cdflib_beta_remaining.py",
                "tests/fixtures/cdflib_beta_remaining.json",
                "tests/test_cdflib_beta_remaining_reference.py",
                "docs/cdflib-beta-remaining-reference.md",
            ],
        }
    )
    support_interfaces.append(
        {
            "module": "zero_finder",
            "status": "implemented_with_documented_python_semantics",
            "implemented_public_names": [
                "set_zero_finder",
                "interval_zf",
                "rc_interval_zf",
                "step_zf",
                "rc_step_zf",
                "final_zf_state",
                "zf_locals",
                "zf_current_status",
                "zf_bound_low",
                "zf_bound_high",
                "zf_crash_left",
                "zf_crash_hi",
            ],
            "python_interfaces": [
                "set_zero_finder",
                "interval_zf",
                "rc_interval_zf",
                "step_zf",
                "rc_step_zf",
                "final_zf_state",
                "ZeroFinder",
                "ZeroFinderResult",
            ],
            "state_mapping": "Fortran shared globals become immutable per-search result fields",
            "evidence": [
                "src/mdanderson_stats/cdflib_root.py",
                "tests/test_cdflib_root.py",
                "docs/cdflib-root.md",
                "tools/reference_cdflib_root.py",
                "tests/fixtures/cdflib_root.json",
                "tests/test_cdflib_root_reference.py",
                "docs/cdflib-root-reference.md",
            ],
        }
    )
    support_interfaces.append(
        {
            "module": "cdf_aux_mod",
            "status": "implemented_with_documented_python_semantics",
            "implemented_public_names": [
                "add_to_one",
                "cdf_finalize_status",
                "cdf_set_zero_finder",
                "check_complements",
                "dbl_in_range",
                "int_in_range",
                "in_range",
                "validate_parameters",
                "which_miss",
                "one_parameter",
                "the_distribution",
                "the_beta",
                "the_binomial",
                "the_chi_square",
                "the_dummy_binomial",
                "the_f",
                "the_gamma",
                "the_negative_binomial",
                "the_non_central_chi_square",
                "the_non_central_f",
                "the_non_central_t",
                "the_normal",
                "the_poisson",
                "the_t",
            ],
            "python_namespace": "cdflib_aux",
            "type_mapping": {
                "one_parameter": "CDFParameter",
                "the_distribution": "CDFDistribution",
            },
            "evidence": [
                "src/mdanderson_stats/cdflib_aux.py",
                "tools/reference_cdflib_aux.py",
                "tests/fixtures/cdflib_aux.json",
                "tests/test_cdflib_aux.py",
                "docs/cdflib-aux.md",
                "tools/benchmark_cdflib_aux.py",
                "docs/cdflib-aux-benchmark.json",
            ],
        }
    )
    support_interfaces.append(
        {
            "module": "biomath_interface_mod",
            "status": "implemented_with_documented_python_semantics",
            "implemented_public_names": [
                "clear_screen",
                "get_character",
                "get_string",
                "get_yn",
                "hold",
                "prompt",
                "write_error",
                "write_message",
                "write_array",
                "get_numbers",
                "get_list_double",
                "report_unit",
                "print_message_format",
                "message_format",
                "num_subs",
                "always_print",
                "print_off",
            ],
            "python_interfaces": [
                "CDFConsole",
                "CDFConsoleError",
                "CDFNumberList",
                "format_cdflib_array",
            ],
            "evidence": [
                "src/mdanderson_stats/cdflib_console.py",
                "tools/reference_cdflib_console.py",
                "tests/fixtures/cdflib_console.json",
                "tests/test_cdflib_console.py",
                "docs/cdflib-console.md",
                "src/mdanderson_stats/cdflib_number_list.py",
                "tests/test_cdflib_number_list.py",
                "docs/cdflib-number-list.md",
                "src/mdanderson_stats/cdflib_array_format.py",
                "tools/reference_cdflib_array_format.py",
                "tests/fixtures/cdflib_array_format.json",
                "tests/test_cdflib_array_format.py",
                "docs/cdflib-array-format.md",
                "tests/test_cdflib_message_format.py",
                "docs/cdflib-message-format.md",
            ],
        }
    )
    for interface in support_interfaces:
        if any(not Path(path).is_file() for path in interface["evidence"]):
            raise RuntimeError("missing public support evidence")
    legacy_names = {
        name for name in header["external_function_names"] if not name.startswith(("cdf", "cum"))
    }
    legacy_implemented = {
        "ipmpar",
        "spmpar",
        "devlpl",
        "fifdint",
        "fifdmax1",
        "fifdmin1",
        "fifdsign",
        "fifidint",
        "fifmod",
        "ftnstop",
        "gaminv",
    }
    legacy_implemented.update(LEGACY_MATH_NAMES | LEGACY_QUANTILE_NAMES)
    legacy_evidence = [
        "src/mdanderson_stats/dcdflib_support.py",
        "tools/reference_dcdflib_support.py",
        "tests/fixtures/dcdflib_support.json",
        "tests/test_dcdflib_support.py",
        "docs/dcdflib-support.md",
        "tools/benchmark_dcdflib_support.py",
        "docs/dcdflib-support-benchmark.json",
        "tools/reference_dcdflib_math.py",
        "tests/fixtures/dcdflib_math.json",
        "tests/test_dcdflib_math.py",
        "docs/dcdflib-math.md",
        "src/mdanderson_stats/dcdflib_quantile_helpers.py",
        "tools/reference_dcdflib_quantile_helpers.py",
        "tests/fixtures/dcdflib_quantile_helpers.json",
        "tests/test_dcdflib_quantile_helpers.py",
        "docs/dcdflib-quantile-helpers.md",
        "tools/benchmark_dcdflib_quantile_helpers.py",
        "docs/dcdflib-quantile-helpers-benchmark.json",
        "src/mdanderson_stats/dcdflib_gamma_inverse.py",
        "tools/reference_dcdflib_gamma_inverse.py",
        "tests/fixtures/dcdflib_gamma_inverse.json",
        "tests/test_dcdflib_gamma_inverse.py",
        "docs/dcdflib-gamma-inverse.md",
        "tools/benchmark_dcdflib_gamma_inverse.py",
        "docs/dcdflib-gamma-inverse-benchmark.json",
    ]
    if not legacy_implemented <= legacy_names or any(
        not Path(path).is_file() for path in legacy_evidence
    ):
        raise RuntimeError("Missing legacy support declaration or evidence")
    legacy_support = {
        "status": "partially_implemented_with_documented_python_semantics",
        "c_public_count": len(legacy_names),
        "implemented_public_names": sorted(legacy_implemented),
        "remaining_public_names": sorted(legacy_names - legacy_implemented),
        "f77_name_aliases": {"erf1": "erf", "Xgamm": "gamma"},
        "c_only_names": [
            "fifdint",
            "fifdmax1",
            "fifdmin1",
            "fifdsign",
            "fifidint",
            "fifmod",
            "ftnstop",
        ],
        "python_namespace": "dcdflib_support",
        "mathematical_public_names": sorted(LEGACY_MATH_NAMES),
        "quantile_helper_names": sorted(LEGACY_QUANTILE_NAMES),
        "gamma_inverse_names": ["gaminv"],
        "cross_version_notes": {
            "exparg": "Legacy 0.99999 margin and rounded log(radix), distinct from F95",
            "bfrac": "Redundant displacement computed from a,b,x,y",
            "output_arguments": (
                "Returned arrays/accumulator; invalid status/sentinels become exceptions"
            ),
        },
        "evidence": legacy_evidence,
    }
    result = {
        "archive_sha256": digest,
        "regular_file_count": len(members),
        "directory_members": directories,
        "role_counts": dict(Counter(row["role"] for row in members)),
        "distribution_interfaces": distributions,
        "support_interfaces": support_interfaces,
        "legacy_support_interfaces": legacy_support,
        "f77_declared_entry_count": len(f77_names),
        "f77_names_not_in_c_header": sorted(f77_names - set(header["external_function_names"])),
        "c_names_not_in_f77": sorted(set(header["external_function_names"]) - f77_names),
        "c_header_external_count": len(header["external_function_names"]),
        "c_header_without_detected_definition": sorted(
            set(header["external_function_names"]) - definitions
        ),
        "c_definitions_without_header_prototype": sorted(
            definitions - set(header["external_function_names"])
        ),
        "backup_identical_to_primary": contents[F95_ROOT + "source/#cdf_binomial_mod.f90#"]
        == contents[F95_ROOT + "source/cdf_binomial_mod.f90"],
        "inventory_limits": (
            "Declaration extraction is an inventory, not a compiler/export or numerical "
            "equivalence proof. Public support and legacy contracts remain open."
        ),
        "members": members,
    }
    Path("docs/cdflib90-archive.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "regular_file_count",
                    "role_counts",
                    "c_header_external_count",
                    "c_header_without_detected_definition",
                    "c_definitions_without_header_prototype",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
