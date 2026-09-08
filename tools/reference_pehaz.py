"""Run the unchanged archived S pehaz function in R and record numeric output."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/MUHAZ/source/S/all.s")
    directory = Path("research/raw/reference/pehaz").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    function = (
        "pehaz <- function"
        + source.read_text()
        .split("\npehaz <- function", 1)[1]
        .split("plot.muhaz <- function", 1)[0]
    )
    original = directory / "pehaz.R"
    original.write_text(function)
    driver = directory / "driver.R"
    driver.write_text("""source("pehaz.R")
a <- commandArgs(trailingOnly=TRUE)
d <- read.table(a[1])
invisible(capture.output(fit <- pehaz(d[,1],d[,2],width=as.numeric(a[2]),
                                      min.time=as.numeric(a[3]),max.time=as.numeric(a[4]))))
n <- length(fit$Hazard)
write.table(cbind(fit$Cuts[1:n],fit$Cuts[2:(n+1)],fit$Hazard,fit$Events,
                  fit$At.Risk,fit$F.U.Time),row.names=FALSE,col.names=FALSE,na="NA")
""")
    cases = []
    for times, delta in [
        ([0.2, 0.7, 1, 1.4, 2, 2.8, 3], [1, 0, 1, 1, 0, 1, 1]),
        ([0, 1, 1, 2, 3], [1, 1, 0, 1, 1]),
        ([1, 2, 3], [0, 0, 0]),
    ]:
        for width, bounds in [(1, (0, 3)), (0.8, (0, 2.5)), (0.7, (0.5, 4))]:
            cases.append(dict(times=times, delta=delta, width=width, bounds=bounds))
    cases.append(dict(times=[1, 2, 3, 4, 5], delta=[1, 1, 0, 1, 0], width=None, bounds=(0, 5)))
    for c in cases:
        data = directory / "data.txt"
        data.write_text(
            "\n".join(f"{t} {d}" for t, d in zip(c["times"], c["delta"], strict=True)) + "\n"
        )
        output = subprocess.run(
            [
                "Rscript",
                str(driver),
                str(data),
                str(c["width"]) if c["width"] else "NA",
                *map(str, c["bounds"]),
            ],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        def number(v):
            return None if v in ("NA", "NaN") else v if v in ("Inf", "-Inf") else float(v)

        c["rows"] = [[number(v) for v in row.split()] for row in output.stdout.splitlines()]
    result = dict(
        source="Unchanged archived pehaz S function, executed in R",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        runtime=subprocess.check_output(["Rscript", "--version"], text=True).strip(),
        cases=cases,
    )
    Path("tests/fixtures/pehaz.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} archived pehaz cases")


if __name__ == "__main__":
    main()
