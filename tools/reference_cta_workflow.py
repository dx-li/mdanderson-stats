"""Run the unchanged CTA program twice with retained analysis choices."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/CTA/source/cta0298/cta0298.f").resolve()
    work = Path("research/raw/reference/cta-workflow").resolve()
    work.mkdir(parents=True, exist_ok=True)
    executable = work / "cta"
    flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    subprocess.run(
        ["gfortran", *flags, str(source), "-o", str(executable)], check=True, capture_output=True
    )
    # Select all analyses, then ask for the same choices on the second table.
    script = (
        "2 2\n1 9\n5 15\ny\ny\ny\ny\ny\nc\n1\ny\nc\n1\n0.05\ny\nr\ny\ny\ny\ny\n"
        "2 2\n2 8\n4 16\ny\nc\n1\nc\n1\n0.05\nr\ny\ny\nn\n"
    )
    run = subprocess.run(
        [str(executable)],
        input=script,
        text=True,
        capture_output=True,
        check=True,
        cwd=work,
        timeout=30,
    )
    (work / "input.txt").write_text(script)
    (work / "stdout.txt").write_text(run.stdout)
    report = (work / "chisqt.lis").read_text()
    blocks = report.split(" ROW  COL")[1:]
    if len(blocks) != 2:
        raise RuntimeError("expected two complete CTA report blocks")
    cases = []
    for table, block in zip([[[1, 9], [5, 15]], [[2, 8], [4, 16]]], blocks, strict=True):

        def numbers(pattern):
            match = re.search(pattern, block, re.MULTILINE)
            if match is None:
                raise RuntimeError(f"missing report pattern {pattern}")
            return [float(x) for x in match.groups()]

        chi = re.findall(r"CHISQ=\s*(\S+)\s+P=\s*(\S+)", block)
        diagnostic = [
            numbers(r"^\s*" + label + r"\s+(\S+)\s+(\S+)")
            for label in (
                "Sensitivity",
                "Specificity",
                "Positive Predicitive Value",
                "Negative Predicitive Value",
            )
        ]
        cells = [
            list(map(float, line.split()))
            for line in block.splitlines()
            if re.match(r"^\s+[12]\s+[12]\s+\d+\.\d{2}", line)
        ]
        cases.append(
            {
                "observed": table,
                "cells": cells,
                "chi_square": [[float(x) for x in row] for row in chi],
                "fisher": numbers(r"SUM OF PR\(Fishers exact probability\) =\s*(\S+)")[0],
                "kappa": numbers(r"KAPPA=\s*(\S+)")[0],
                "variances": [numbers(r"VAR1=\s*(\S+)")[0], numbers(r"VAR2=\s*(\S+)")[0]],
                "mcnemar": numbers(r"Sum of Individual McNemars:\s*(\S+) Prob:\s*(\S+)"),
                "diagnostic": diagnostic,
                "odds": numbers(r"Relative risk:\s*(\S+)\s+(\S+)\s+(\S+)"),
                "binomial_tails": numbers(r"to get one-sided p-values\s*(\S+)\s+(\S+)"),
                "binomial_two_sided": numbers(r"and two-sided p-value\s*(\S+)")[0],
            }
        )
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "input": script,
        "workflow": "Original PROGRAM cta; two tables, all analyses, "
        "retained choices, details enabled",
        "cases": cases,
    }
    Path("tests/fixtures/cta_workflow.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print("Generated two full-program CTA workflow cases")


if __name__ == "__main__":
    main()
