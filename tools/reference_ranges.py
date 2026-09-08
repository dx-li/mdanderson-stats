"""Capture RANGE2/KWRANGE decisions and groups from the original executables."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from reference_numerics import RAW, compile_program, run


def collect(name: str) -> list[dict]:
    source = (
        "RANGE2/source/range2/range2.f"
        if name == "range2"
        else "KWRANGE/source/kwrange2/kwrange2.f"
    )
    executable = compile_program(name, [RAW / source])
    if name == "range2":
        scenarios = [
            ([1, 2, 4, 7], [10, 10, 10, 10], 2),
            ([9, 1, 4], [3, 20, 5], 2),
            ([0, 0.2, 0.8, 1], [1, 100, 100, 1], 1),
            ([2, 2, 1, 5], [5, 10, 20, 30], 3),
            ([2, 3, 4, 5, 6], [20, 10, 5, 20, 5], 4),
        ]
    else:
        # Valid pooled rank means; rank-sum input is exercised separately below.
        scenarios = [
            ([3, 8, 13], [5, 5, 5], None),
            ([17, 2, 8.5], [7, 3, 10], None),
            ([5, 5, 5], [3, 3, 3], None),
            ([2.5, 8, 13.5, 18], [4, 7, 4, 5], None),
        ]
    cases = []
    for means, sizes, mse in scenarios:
        for critical in (1.5, 3.5, 8):
            for sums in [False, True] if name == "kwrange" else [False]:
                values = [x * n for x, n in zip(means, sizes, strict=True)] if sums else means
                data = f"'reference'\n{len(values)}\n"
                if name == "kwrange":
                    data += "y\n" if sums else "n\n"
                data += "\n".join(f"{x} {n}" for x, n in zip(values, sizes, strict=True)) + "\n"
                if name == "range2":
                    data += f"{mse} 30\n"
                data += f"{critical} 0.05\nn\nn\n"
                output = run(executable, data)
                sorted_block = output.split("Sorted means in ascending order")[1].split(
                    "Enter the critical value"
                )[0]
                order = [
                    int(v) - 1
                    for v in re.findall(
                        r"^\s*(\d+)\s+[-\d.E+]+\s*$", sorted_block, flags=re.MULTILINE
                    )
                ]
                if len(order) != len(values):
                    raise RuntimeError(f"Cannot parse sorted order: {output}")
                groups = [
                    [int(v) - 1 for v in text.split()]
                    for text in re.findall(r"Similarity group\s+\d+\s*\n([^\n]+)", output)
                ]
                reject = [[False] * len(values) for _ in values]
                table = output.split("Similarity table")[1].split("Want a new p-value")[0]
                rows = re.findall(r"^\s*(\d+)\s*-([^\n]+)", table, flags=re.MULTILINE)
                if len(rows) != len(values) - 1:
                    raise RuntimeError(f"Cannot parse comparison table: {output}")
                for j, (label, symbols) in enumerate(rows, start=1):
                    decisions = re.findall(r"[=*]", symbols)[:j]
                    if int(label) - 1 != order[j] or len(decisions) != j:
                        raise RuntimeError(f"Unexpected comparison row: {label} {symbols}")
                    for i, symbol in enumerate(decisions):
                        reject[order[i]][order[j]] = reject[order[j]][order[i]] = symbol == "*"
                cases.append(
                    {
                        "program": name,
                        "values": values,
                        "sizes": sizes,
                        "error_mean_square": mse,
                        "critical_value": critical,
                        "rank_sums": sums,
                        "order": order,
                        "similarity_groups": groups,
                        "reject": reject,
                    }
                )
    return cases


def main() -> None:
    sources = {}
    for name in ("RANGE2", "KWRANGE"):
        archive = RAW / name / f"{name}_V1.tar.gz"
        sources[name] = {
            "url": "https://biostatistics.mdanderson.org/SoftwareDownload/"
            f"SoftwareFiles/{name}/{name}_V1.tar.gz",
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    cases = collect("range2") + collect("kwrange")
    result = {
        "sources": sources,
        "cases": cases,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "build_flags": ["-O2", "-std=legacy"],
    }
    Path("tests/fixtures/ranges.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Recorded {len(cases)} original range-test executions")


if __name__ == "__main__":
    main()
