"""Regenerate bounded original DSF2.2 native evidence, not a Windows execution.

Requires pefile and Unicorn in addition to NumPy. No original code is bundled.
Controller inference, vector/matrix access and CRT math are supplied services;
observation probes record original calls with an analytic survival/hazard service.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args()
    if (
        hashlib.sha256(args.kernel.read_bytes()).hexdigest()
        != "f203c2f774ecabb390e58213076fabe02df15aa74c62df9861a7e2e0ac675a1e"
    ):
        raise ValueError("original kernel hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("prior", "selector", "observation", "lognormal", "schedule", "final_clock"):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "dose_schedule_reference" / f"{name}.py"),
                str(args.kernel),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (args.output / f"dose-schedule-desktop-{name.replace(chr(95), chr(45))}.json").write_text(
            result.stdout
        )


if __name__ == "__main__":
    main()
