"""Archived KPHaz calculations in R, with explicit S syntax adapters."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/MUHAZ/source/S/all.s")
    directory = Path("research/raw/reference/kphaz").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = (
        "kphaz.fit <- function"
        + source.read_text()
        .split("\nkphaz.fit <- function", 1)[1]
        .split("\nkphaz.plot <- function", 1)[0]
    )
    adapted = (
        original.replace("is.inf(", "is.infinite(")
        .replace(
            "return(time, haz, var, strata)",
            "return(list(time=time, haz=haz, var=var, strata=strata))",
        )
        .replace("return(time, haz, var)", "return(list(time=time, haz=haz, var=var))")
    )
    function = directory / "kphaz.R"
    function.write_text(adapted)
    driver = directory / "driver.R"
    driver.write_text("""source("kphaz.R")
a <- commandArgs(trailingOnly=TRUE)
d <- read.table(a[1])
f <- kphaz.fit(d[,1],d[,2],d[,3],q=as.integer(a[2]),method=a[3])
if(length(f$time)>0) write.table(cbind(f$time,f$haz,f$var,f$strata),
                                row.names=FALSE,col.names=FALSE,na="NA")
""")
    inputs = [
        ([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [1] * 5),
        ([0, 1, 1, 2, 3, 4], [1, 0, 1, 1, 1, 0], [1] * 6),
        ([1, 2, 3, 3], [1, 1, 1, 0], [1] * 4),
        (
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            [1] * 6 + [2] * 6,
        ),
    ]
    cases = []
    for times, status, strata in inputs:
        for method in ["nelson", "product-limit"]:
            for q in [1, 2, 4]:
                data = directory / "data.txt"
                data.write_text(
                    "\n".join(f"{t} {d} {s}" for t, d, s in zip(times, status, strata, strict=True))
                    + "\n"
                )
                out = subprocess.run(
                    ["Rscript", str(driver), str(data), str(q), method],
                    cwd=directory,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                def number(v):
                    return None if v in ("NA", "NaN") else v if v in ("Inf", "-Inf") else float(v)

                rows = [[number(v) for v in row.split()] for row in out.stdout.splitlines()]
                cases.append(
                    dict(time=times, status=status, strata=strata, method=method, q=q, rows=rows)
                )
    fixture = dict(
        source=(
            "Archived kphaz.fit numerical code; is.inf alias and S multi-value return adapted to R"
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        runtime=subprocess.check_output(["Rscript", "--version"], text=True).strip(),
        cases=cases,
    )
    Path("tests/fixtures/kphaz.json").write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} archived KPHaz cases")


if __name__ == "__main__":
    main()
