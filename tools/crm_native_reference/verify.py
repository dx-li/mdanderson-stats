"""Regenerate bounded original-instruction fixtures for both desktop kernels.

Usage: python verify.py /path/CRMKernel.dll /path/BMA-CRMKernel.dll
Requires dnfile, dncil and unicorn only in the research environment.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "research/fixtures/crm-native"
for kernel in sys.argv[1:]:
    for script, fixture in (
        ("patient_reference.py", "patient-outcomes.jsonl"),
        ("cohort_reference.py", "cohort-control.jsonl"),
        ("managed_cohort_reference.py", "managed-cohort-control.jsonl"),
    ):
        result = subprocess.run(
            [sys.executable, str(HERE / script), kernel], check=True, capture_output=True
        )
        assert result.stdout == (FIXTURES / fixture).read_bytes(), (kernel, fixture)
        print(Path(kernel).name, fixture, len(result.stdout.splitlines()), "byte-identical")
if len(sys.argv) != 3:
    raise SystemExit("Supply both hash-pinned desktop kernels")
