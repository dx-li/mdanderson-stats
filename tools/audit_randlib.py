"""Inventory every RANDLIB archive member against its implementation/evidence group."""

import hashlib
import json
import tarfile
from pathlib import Path

GROUPS = {
    "streams": ("randlib.py", "test_randlib.py"),
    "seeding": ("_randlib_seeding.py", "test_randlib_seeding.py"),
    "sampling": ("_randlib_sampling.py", "test_randlib_sampling.py"),
    "exponential": ("_randlib_distributions.py", "test_randlib_exponential.py"),
    "normal": ("_randlib_normal.py", "test_randlib_normal.py"),
    "gamma": ("_randlib_gamma.py", "test_randlib_gamma.py"),
    "chi_f": ("_randlib_chi_f.py", "test_randlib_chi_f.py"),
    "beta": ("_randlib_beta.py", "test_randlib_beta.py"),
    "binomial": ("_randlib_binomial.py", "test_randlib_binomial.py"),
    "poisson": ("_randlib_poisson.py", "test_randlib_poisson.py"),
    "negative_binomial": ("_randlib_negative_binomial.py", "test_randlib_negative_binomial.py"),
    "multinomial": ("_randlib_multinomial.py", "test_randlib_multinomial.py"),
    "multivariate_normal": ("randlib_multivariate.py", "test_randlib_multivariate_normal.py"),
}
F77 = {
    **dict.fromkeys(
        "advnst getcgn getsd ignlgi initgn inrgcm mltmod qrgnin setall setant setsd".split(),
        "streams",
    ),
    **dict.fromkeys("phrtsd lennob".split(), "seeding"),
    **dict.fromkeys("genprm genunf ignuin ranf".split(), "sampling"),
    **dict.fromkeys("genexp sexpo".split(), "exponential"),
    **dict.fromkeys("gennor snorm".split(), "normal"),
    **dict.fromkeys("gengam sgamma".split(), "gamma"),
    **dict.fromkeys("genchi gennch genf gennf".split(), "chi_f"),
    **dict.fromkeys("genmn setgmn spofa sdot".split(), "multivariate_normal"),
    "genbet": "beta",
    "ignbin": "binomial",
    "ignpoi": "poisson",
    "ignnbn": "negative_binomial",
    "genmul": "multinomial",
}
F95 = {
    "ecuyer_cote_mod": "streams",
    "user_set_generator": "seeding",
    **{
        f"random_{s}_mod": "sampling"
        for s in ["permutation", "uniform", "uniform_integer", "standard_uniform"]
    },
    **{f"random_{s}_mod": "chi_f" for s in ["chisq", "nc_chisq", "f", "nc_f"]},
    **{
        f"random_{s}_mod": s.removeprefix("standard_")
        for s in [
            "standard_exponential",
            "exponential",
            "standard_normal",
            "normal",
            "standard_gamma",
            "gamma",
            "beta",
            "binomial",
            "poisson",
            "negative_binomial",
            "multinomial",
            "multivariate_normal",
        ]
    },
}


def main():
    archive = Path("research/raw/RANDLIB/RANDLIB_V90.tar.gz")
    root = Path("research/raw/RANDLIB/source")
    records = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unexpected archive member: {member.name}")
            path = Path(member.name)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing data: {path}")
            data = stream.read()
            local = root / path
            if data != local.read_bytes():
                raise ValueError(f"Archive/extraction mismatch: {path}")
            groups = []
            if path.suffix == ".f" and path.parent.name == "src":
                groups = [F77[path.stem]]
                role = "library"
            elif path.suffix == ".f90":
                groups = [F95[path.stem]]
                role = "library"
            elif path.parent.name == "src" and path.suffix == ".c":
                groups = {
                    "com.c": ["streams"],
                    "linpack.c": ["multivariate_normal"],
                    "randlib.c": list(GROUPS),
                }[path.name]
                role = "library"
            elif path.parent.name == "test" and path.suffix in (".f", ".c"):
                groups = {
                    "tstbot": ["streams"],
                    "tstmid": list(GROUPS),
                    "tstgmn": ["multivariate_normal"],
                }[path.stem]
                role = "demonstration"
            elif path.suffix == ".h":
                role = "native declarations"
            elif path.name in ("Makefile", "compile.randlib90", "INSTALL"):
                role = "native build instructions"
            elif path.parent.name == "doc" or path.name in ("readme", "LEGALITIES", "HOWTOGET"):
                role = "documentation/terms"
            else:
                raise ValueError(f"Unclassified archive member: {path}")
            evidence = []
            for group in groups:
                module, test = GROUPS[group]
                for target in [Path("src/mdanderson_stats") / module, Path("tests") / test]:
                    if not target.is_file():
                        raise ValueError(f"Missing evidence: {target}")
                    evidence.append(str(target))
            records.append(
                dict(
                    path=str(path),
                    sha256=hashlib.sha256(data).hexdigest(),
                    role=role,
                    groups=groups,
                    evidence=sorted(set(evidence)),
                )
            )
    extracted = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if extracted != {r["path"] for r in records}:
        raise ValueError("Uninventoried extracted files")
    report = dict(
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        file_count=len(records),
        files=sorted(records, key=lambda r: r["path"]),
    )
    Path("docs/randlib-archive.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Inventoried and byte-verified {len(records)} archive files")


if __name__ == "__main__":
    main()
