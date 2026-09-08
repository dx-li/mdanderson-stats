"""Execute unchanged GENLST to capture printed rows and page boundaries."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from mdanderson_stats import RanlistSession, RanlistSpecification, ranlist_parameter_text


def main():
    source = Path("research/raw/RANLIST/source/source/ranlist.f").resolve()
    work = Path("research/raw/reference/ranlist-reports").resolve()
    work.mkdir(parents=True, exist_ok=True)
    exe = work / "ranlist"
    flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    subprocess.run(["gfortran", *flags, str(source), "-o", str(exe)], check=True)
    cases = []
    for restricted in [False, True]:
        for named in [False, True]:
            for requested in [None, 55]:
                directory = work / f"case-{len(cases)}"
                directory.mkdir(exist_ok=True)
                output = directory / "report.txt"
                output.unlink(missing_ok=True)
                labels = ("North", "South") if named else ("", "")
                title = tuple(f"Title line {i + 1}" for i in range(9 if named else 1))
                state = RanlistSession(
                    RanlistSpecification(
                        (1, 2),
                        restricted=restricted,
                        legacy=True,
                        strata=labels,
                        treatments=("Control", "Experimental") if named else (),
                        title=title,
                        balance=(1, 3) if restricted else (1, 1),
                    ),
                    (3, 0),
                )
                parameters = ranlist_parameter_text(state)
                (directory / "list.par").write_text(parameters)
                inputs = ["", "3", "list.par", "P", "Y", "N", "1" if requested is None else "2"]
                if requested is not None:
                    inputs.append(str(requested))
                inputs.extend(["report.txt", "P", "0", "0"])
                run = subprocess.run(
                    [str(exe)],
                    cwd=directory,
                    input="\n".join(inputs) + "\n",
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                if not output.exists():
                    raise RuntimeError(f"native report not created: {run.stdout[-2000:]}")
                pages = []
                for page in output.read_text().split("\f")[1:]:
                    header = re.search(r"STRATUM\s+(.+?)\s+PAGE\s+(\d+)", page)
                    if header is None:
                        raise RuntimeError("native page has no stratum/page header")
                    label = header[1].strip()
                    stream = labels.index(label) + 1 if named else int(label)
                    rows = re.findall(r"^\s*(\d+)\s+(\d+)\s+[^\n]*\.{29}", page, re.MULTILINE)
                    pages.append(
                        {
                            "stratum": stream,
                            "page": int(header[2]),
                            "patients": [int(row[0]) for row in rows],
                            "treatments": [int(row[1]) for row in rows],
                        }
                    )
                if sum(len(page["patients"]) for page in pages) != (
                    3 if requested is None else 110
                ):
                    raise RuntimeError("native report row count differs from requested count")
                cases.append({"parameters": parameters, "requested": requested, "pages": pages})
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
    }
    Path("tests/fixtures/ranlist_reports.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} complete-program report cases")


if __name__ == "__main__":
    main()
