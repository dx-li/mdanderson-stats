"""Verify every TDTASP archive member against extracted bytes and reviewed coverage."""

import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

ROLES = {
    "tdtasp": "tdtasp_study and TDTASPTemplate.run; independent repeated studies",
    "genetics_mod": "Validated direct/disequilibrium frequencies and all population inputs",
    "families_mod": "Immutable vectorized 256-parental-family enumeration and outputs",
    "offspring_mod": (
        "Mendelian transmission, recombination, affected offspring and corrected/legacy ASP"
    ),
    "ascertain_mod": (
        "Family/individual list selection, parent eligibility, contribution "
        "probabilities and moments"
    ),
    "exp_n_given_k": "Duplicate Poisson truncated-mean helper replaced by stable series ratios",
    "tail_poisson_mod": (
        "Positive-series log Poisson tails, checked convergence and rare-event normalization"
    ),
    "statistics_mod": "Power/sample-size study calls, fixed comparison and complete reports",
    "power_bin1_mod": (
        "Discrete fixed-observation critical values, size and power; half-up rounding"
    ),
    "power_bin1_vn_mod": (
        "Complete eligible-family binomial mixture replaces short-tail/normal approximations"
    ),
    "bin1_ss_pow_mod": (
        "First qualifying fixed integer sample-size search; generic internal inversions "
        "not public workflows"
    ),
    "bin1_vn_ss_pow_mod": (
        "Cached first-crossing family search with monotone upper bound; no assumed "
        "monotonic actual power"
    ),
    "normal_integrate": (
        "Hermite integration is unnecessary after complete discrete family-count averaging"
    ),
    "cdf_aux_mod": (
        "SciPy numerical support replaces archived CDF internals; original code not bundled"
    ),
    "cdf_beta_mod": "SciPy binomial distribution calculations replace archived beta-CDF support",
    "beta_gamma_mod": (
        "SciPy special functions replace archived beta/gamma support; original ACM code not bundled"
    ),
    "cdf_binomial_mod": "SciPy binomial pmf/cdf/sf and integer critical-region evaluation",
    "cdf_normal_mod": "Normal approximation removed from family mixtures; exact discrete averaging",
    "zero_finder": (
        "Discrete ordered/bounded searches replace continuous reverse-communication inversion"
    ),
    "param_defn_mod": (
        "Typed immutable template and calculation inputs with validated mathematical domains"
    ),
    "param_io": "Explicit component validation and canonical immutable specification values",
    "parser_mod": "Strict template numeric, list, assignment and choice parsing",
    "user_interface_mod": (
        "Template comments/lists/choices plus Python exceptions and caller-controlled I/O"
    ),
    "create_write_form_mod": "format_tdtasp_template blank and filled forms in source field order",
    "intro_mod": "Method/model/ascertainment/exact-test tutorial in docs/tdtasp.md",
    "user_dialogues_mod": "Python calls, template workflows and caller-managed report/form files",
    "get_numbers_mod": "Validated arguments replace console numeric prompts and retry loops",
    "open_file_mod": "Caller-controlled Path.read_text/write_text replaces file dialogs",
    "print_array": "Immutable full family arrays and readable study summaries",
    "print_it": "Returned reports and exceptions replace terminal formatting, paging and prompts",
}


def main():
    archive = Path("research/raw/TDTASP/TDTASP  _V1.tar.gz")
    extraction = Path("research/raw/TDTASP/source")
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
                role, replacement = "source", ROLES[path.stem]
                routines = sorted(
                    set(
                        re.findall(
                            r"^\s*(?:FUNCTION|SUBROUTINE)\s+(\w+)",
                            content.decode(),
                            re.MULTILINE | re.IGNORECASE,
                        )
                    )
                )
            elif path.suffix.lower() == ".exe":
                role, replacement = (
                    "binary",
                    "Historical Windows executable replaced by Python package",
                )
            elif path.name in ("Makefile", "compile.tdtasp"):
                role, replacement = "build", "pyproject.toml, uv build and Python 3.12/3.13/3.14 CI"
            elif path.suffix in (".pdf", ".ps", ".tex") or path.name in (
                "LEGALITIES.txt",
                "readme",
            ):
                role, replacement = (
                    "documentation",
                    "Method/usage/compatibility docs and retained legal terms",
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
        "counts": dict(Counter(e["role"] for e in entries)),
        "members": entries,
        "evidence": [
            "docs/tdtasp.md",
            "docs/tdtasp-coverage.md",
            "tests/test_tdtasp_genetics.py",
            "tests/test_tdtasp_ascertainment.py",
            "tests/test_tdtasp_power.py",
            "tests/test_tdtasp_sample_size.py",
            "tests/test_tdtasp_study.py",
            "tests/test_tdtasp_template.py",
        ],
    }
    Path("docs/tdtasp-archive.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited {len(entries)} regular archive members: {result['counts']}")


if __name__ == "__main__":
    main()
