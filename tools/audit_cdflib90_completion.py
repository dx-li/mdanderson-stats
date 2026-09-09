"""Reconcile source exports, installed Python mappings and archived documentation."""

import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256, F95_ROOT

import mdanderson_stats as package


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    inventory = json.loads(Path("docs/cdflib90-archive.json").read_text())
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    require(hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA256, "Archive changed")
    with tarfile.open(archive) as tar:
        contents = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    require(len(contents) == 106, "Archive membership changed")
    require(
        {m["path"] for m in inventory["members"]} == set(contents),
        "Inventory/archive membership mismatch",
    )
    for member in inventory["members"]:
        require(
            hashlib.sha256(contents[member["path"]]).hexdigest() == member["sha256"],
            "Member changed",
        )
    exports = {}
    for row in inventory["distribution_interfaces"]:
        require(
            row["legacy_contract_status"] == "implemented_with_documented_python_semantics",
            "Incomplete legacy distribution",
        )
        exports["cdf_" + row["name"] + "_mod"] = row["f95_public_interfaces"]
        for name in row["f95_public_interfaces"] + row["legacy_distribution_names"]:
            require(callable(getattr(package, name, None)), f"Missing Python distribution: {name}")
    for row in inventory["support_interfaces"]:
        require(
            row["status"] == "implemented_with_documented_python_semantics", "Incomplete support"
        )
        module = row["module"]
        names = row.get("public_names", row.get("implemented_public_names"))
        source = contents[F95_ROOT + "source/" + module + ".f90"].decode("latin1")
        source = re.sub(
            r"&\s*\n\s*&?", " ", "\n".join(line.split("!", 1)[0] for line in source.splitlines())
        )
        record = next(
            m for m in inventory["members"] if m["path"] == F95_ROOT + "source/" + module + ".f90"
        )
        declarations = record["declarations"]
        if declarations["first_default_access_statement"] == "private":
            native = declarations["explicit_public_names"]
        elif module == "biomath_mathlib_mod":
            native = declarations["routine_declarations_including_entries"]
        elif module == "biomath_constants_mod":
            native = re.findall(r"(?:::|,)\s*(\w+)\s*=", source)
        elif module == "cdf_aux_mod":
            header = source.split("CONTAINS")[0]
            native = (
                declarations["routine_declarations_including_entries"]
                + re.findall(r"^\s*(?:TYPE\s*::|INTERFACE)\s+(\w+)", header, re.I | re.M)
                + re.findall(r"TYPE\s*\(the_distribution\),\s*PARAMETER\s*::\s*(\w+)", header, re.I)
            )
        else:
            raise RuntimeError(f"Unreviewed default-public module: {module}")
        require(
            set(native) == set(names),
            f"Public name mismatch in {module}: {set(native) ^ set(names)}",
        )
        exports[module] = sorted(names)
        namespace = getattr(package, row.get("python_namespace", ""), package)
        for name in row.get("python_interfaces", names):
            mapped = row.get("type_mapping", {}).get(name, name)
            require(hasattr(namespace, mapped), f"Missing Python support: {module}.{mapped}")
        if module == "biomath_constants_mod":
            require(set(package.cdflib_constants.__all__) == set(names), "Constant exports differ")
            for name in names:
                require(hasattr(package.cdflib_constants, name), f"Missing constant: {name}")

    # Fortran USE association re-exports these names from default-public modules.
    transitive = {
        "biomath_mathlib_mod": {"biomath_constants_mod": exports["biomath_constants_mod"]},
        "cdf_aux_mod": {
            m: exports[m] for m in ["biomath_constants_mod", "biomath_interface_mod", "zero_finder"]
        },
    }
    legacy = inventory["legacy_support_interfaces"]
    native_c = set(legacy["implemented_public_names"]) | {
        name
        for row in inventory["distribution_interfaces"]
        for name in row["legacy_distribution_names"]
    }
    header = next(m for m in inventory["members"] if m["path"] == "source/dcdflib.c/src/cdflib.h")
    require(native_c == set(header["declarations"]["external_function_names"]), "Unmapped C export")
    require(
        set(package.dcdflib_support.__all__) == set(legacy["implemented_public_names"]),
        "Python legacy exports differ",
    )
    for name in package.dcdflib_support.__all__:
        require(
            callable(getattr(package.dcdflib_support, name, None)),
            f"Missing legacy callable: {name}",
        )
    f77 = {
        n
        for m in inventory["members"]
        if m["role"] == "f77_source"
        for n in m["declarations"]["routine_declarations_including_entries"]
    }
    translated = {
        legacy["f77_name_aliases"].get(n, n) for n in native_c - set(legacy["c_only_names"])
    }
    require(f77 == translated, "Unmapped F77 export")
    require(
        contents[F95_ROOT + "LEGALITIES"]
        == Path("notices/mdanderson-cdflib90-LEGALITIES.txt").read_bytes(),
        "Notice differs",
    )

    imports = []
    for module, names in exports.items():
        imported = set(names)
        for extra in transitive.get(module, {}).values():
            imported.update(extra)
        imports.extend((module, name) for name in sorted(imported))
    driver = (
        "program public_exports\n"
        + "\n".join(
            f"use {module}, only: audited_{i} => {name}" for i, (module, name) in enumerate(imports)
        )
        + "\nimplicit none\nend program public_exports\n"
    )
    order = re.findall(r"\b(\w+\.f90)\b", contents[F95_ROOT + "source/compile.cdflib90"].decode())
    require(set(order) == {m + ".f90" for m in exports}, "Build/source module mismatch")
    flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
    with tempfile.TemporaryDirectory(prefix="cdflib90-exports-") as tmp:
        work = Path(tmp)
        for name in order:
            (work / name).write_bytes(contents[F95_ROOT + "source/" + name])
        (work / "exports.f90").write_text(driver)
        subprocess.run(
            [*flags, *order, "exports.f90", "-o", "exports"],
            cwd=work,
            check=True,
            capture_output=True,
            timeout=60,
        )
        subprocess.run([str(work / "exports")], check=True, capture_output=True, timeout=3)

    catalog = json.loads(Path("src/mdanderson_stats/catalog.json").read_text())
    stattab = next(e for e in catalog["entries"] if e["name"] == "STATTAB")
    require(b"STATTAB" in contents[F95_ROOT + "DOC.TEX"], "Foreign manual identity changed")
    documents = []
    for m in inventory["members"]:
        if m["role"] != "documentation":
            continue
        path = m["path"]
        if path == F95_ROOT + "DOC.TEX":
            disposition = (
                "STATTAB 2.0 manual; separate catalog entry, not a duplicate CDFLIB90 manual"
            )
        elif path.endswith("LEGALITIES"):
            disposition = "Retained byte-for-byte in the Python package notice"
        elif path.endswith(("INSTALL", "HOWTOGET", "readme")):
            disposition = (
                "Historical acquisition/build/platform instructions replaced by Python packaging; "
                "numerical guidance reconciled"
            )
        else:
            disposition = (
                "CDFLIB90/DCDFLIB API documentation reconciled against executable contracts "
                "and documented errata"
            )
        documents.append(dict(path=path, sha256=m["sha256"], disposition=disposition))
    require(len(documents) == 17, "Documentation inventory changed")
    report = dict(
        archive_sha256=ARCHIVE_SHA256,
        status="complete_with_documented_python_semantics",
        f95_modules=exports,
        transitive_exports=transitive,
        compiler_import_count=len(imports),
        compiler_flags=flags,
        compiler_source_order=order,
        compiler=subprocess.check_output([flags[0], "--version"], text=True).splitlines()[0],
        driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
        c_external_count=len(native_c),
        f77_entry_count=len(f77),
        legacy_support_count=len(legacy["implemented_public_names"]),
        documentation=documents,
        foreign_manual_catalog_entry=dict(
            id=stattab["id"],
            name=stattab["name"],
            scope="Separate STATTAB workflow remains tracked in the full catalog",
            evidence="docs/stattab-research.md",
        ),
        numerical_evidence=(
            "Per-interface native/independent fixtures and tests in cdflib90-archive.json; "
            "export compilation alone is not a numerical proof"
        ),
    )
    Path("docs/cdflib90-completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "compiler_import_count",
                    "c_external_count",
                    "f77_entry_count",
                    "legacy_support_count",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
