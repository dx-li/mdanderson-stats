"""Account for every MULTINOMPOW archive member and its Python replacement."""

import hashlib
import json
import re
import tarfile
from pathlib import Path

ROLES = {
    "mp_main": "Validated multinomial_power arguments and independent repeated calls",
    "mp_setup_mod": "Statistics, whole-group critical regions and immutable result allocation",
    "mp_struct_mod": "MultinomialPower result and local function state",
    "power_mod": "Batched alternative probabilities and ordered cumulative powers",
    "calc_point_prob_mod": "Duplicate probability helper replaced by _mass and gammaln",
    "log_factorial_mod": "SciPy gammaln; native tests cross original table boundary at 1001",
    "gamma_mod": "SciPy gammaln replaces archived ACM gamma support; original code not bundled",
    "n_partitions_mod": "Exact integer math.comb sample-space count",
    "partial_sums_mod": "Support for original partition count replaced by math.comb",
    "update_partition_mod": "Stars-and-bars weak compositions, validated by exact/native powers",
    "summation_mod": "NumPy longdouble probability accumulation and checked normalization",
    "perm_sort_array_mod": "NumPy argsort; unused generic sort overloads are internal support",
    "get_numbers_mod": "Explicit Python numeric validation replaces interactive input loops",
    "open_file_mod": "Caller-controlled Path.write_text for returned report text",
    "print_it": "Python calls/exceptions replace console prompts, holds and format editing",
    "print_array_mod": "Explicit probability vectors in format_multinomial_power",
    "write_answers_mod": "format_multinomial_power problem, critical-region and power tables",
}


def main():
    archive = Path("research/raw/MULTINOMPOW/MULTINOMPOW _V1.tar.gz")
    extraction = Path("research/raw/MULTINOMPOW/source")
    entries = []
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"unexpected nonregular member: {member.name}")
            path = Path(member.name)
            handle = tar.extractfile(member)
            if handle is None:
                raise RuntimeError(f"unreadable archive member: {member.name}")
            content = handle.read()
            if (extraction / path).read_bytes() != content:
                raise RuntimeError(f"extracted content differs: {member.name}")
            routines = []
            if path.suffix == ".f90":
                role = "source"
                replacement = ROLES[path.stem]
                routines = sorted(
                    set(
                        re.findall(
                            r"^\s*(?:FUNCTION|SUBROUTINE)\s+(\w+)",
                            content.decode(),
                            re.MULTILINE | re.IGNORECASE,
                        )
                    )
                )
            elif path.suffix.lower() in (".exe", ".se"):
                role, replacement = (
                    "binary",
                    "Historical Windows build/installer replaced by Python package",
                )
            elif path.name in ("Makefile", "compile.multinom_pow"):
                role, replacement = (
                    "build",
                    "Repository pyproject.toml, uv build and three-version CI",
                )
            elif path.suffix in (".pdf", ".ps", ".tex") or path.name in (
                "HOWTOGET",
                "INSTALL",
                "LEGALITITES",
                "readme",
                "README~",
            ):
                role, replacement = (
                    "documentation",
                    "Method, notices, usage and compatibility documented",
                )
            else:
                raise RuntimeError(f"unclassified archive member: {member.name}")
            entries.append(
                {
                    "path": member.name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "role": role,
                    "replacement": replacement,
                    "routines": routines,
                }
            )
    if {Path(e["path"]).stem for e in entries if e["role"] == "source"} != set(ROLES):
        raise RuntimeError("source inventory differs from reviewed mapping")
    result = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "members": entries,
        "evidence": [
            "src/mdanderson_stats/multinomial_power.py",
            "tests/test_multinomial_power.py",
            "tests/test_multinomial_power_native.py",
            "docs/multinomial-power.md",
        ],
    }
    Path("docs/multinompow-archive.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited {len(entries)} regular archive members")


if __name__ == "__main__":
    main()
