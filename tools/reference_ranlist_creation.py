"""Execute unchanged MKLST/GTSEED to create original parameter files."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/RANLIST/source/source/ranlist.f").resolve()
    work = Path("research/raw/reference/ranlist-creation").resolve()
    work.mkdir(parents=True, exist_ok=True)
    exe = work / "ranlist"
    flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    subprocess.run(["gfortran", *flags, str(source), "-o", str(exe)], check=True)
    cases = []
    for restricted in [False, True]:
        for phrase in [None, "abc", "a phrase longer than thirty one characters"]:
            folder = work / f"case-{len(cases)}"
            folder.mkdir(exist_ok=True)
            output = folder / "created.par"
            output.unlink(missing_ok=True)
            inputs = ["", "1", "1", "Native creation"]
            inputs.extend(["N", "1", "1"] if phrase is None else ["P", phrase])
            inputs.extend(
                [
                    "2",
                    "Y",
                    "North",
                    "South",
                    "2",
                    "Y",
                    "Control",
                    "Experimental",
                    "R" if restricted else "U",
                    "N",
                    "1 2",
                ]
            )
            if restricted:
                inputs.extend(["N", "1 3"])
            inputs.extend(["created.par", "P", "0"])
            run = subprocess.run(
                [str(exe)],
                cwd=folder,
                input="\n".join(inputs) + "\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            )
            if not output.exists():
                raise RuntimeError(f"native creation failed: {run.stdout[-3000:]}")
            cases.append(
                {"restricted": restricted, "phrase": phrase, "parameters": output.read_text()}
            )
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
    }
    Path("tests/fixtures/ranlist_creation.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native list-creation cases")


if __name__ == "__main__":
    main()
