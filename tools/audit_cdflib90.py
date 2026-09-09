"""Inventory every pinned CDFLIB90 archive member and preserve unresolved API scope."""

import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

ARCHIVE_SHA256 = "2f5dd397b93546222a3b31e02073abeee1fc213cea75768c34b17e06c8264a3b"
F95_ROOT = "CDFLIB90/source/cdflib90_1.2/"
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
    "biomath_strings_mod": "Public character/string case conversion and lexical comparison",
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
    binomial = next(row for row in distributions if row["name"] == "binomial")
    binomial["legacy_reference_evidence"] = [
        "tools/reference_dcdflib_binomial.py",
        "tests/fixtures/dcdflib_binomial.json",
        "tests/test_dcdflib_binomial_reference.py",
        "docs/dcdflib-binomial-reference.md",
    ]
    binomial["legacy_review_notes"] = (
        "Unchanged C/F77 contracts audited: invalid-mode sentinel, C-only small-n "
        "process exits, wider search bounds and false-success inversions; legacy API pending."
    )
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
    result = {
        "archive_sha256": digest,
        "regular_file_count": len(members),
        "directory_members": directories,
        "role_counts": dict(Counter(row["role"] for row in members)),
        "distribution_interfaces": distributions,
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
