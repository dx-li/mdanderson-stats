"""Execute unchanged final-estimation statements from a user-supplied official R file.

Original source is not redistributed. Usage:
  python tools/reference_mtpi_legacy_selection.py path/to/mTPI-middle.R --rscript Rscript
The output is a synthetic reference, not native trial-controller validation.
"""

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/mtpi-legacy-selection.csv")
    )
    args = parser.parse_args()
    raw = args.source.read_bytes()
    # This pins the file in TPI_V2.1PID439.zip, not a subsequently edited script.
    expected = "2629c28a1bf607329b125d7fca6715d8d1bfd0615162ebc4d091b2c603f5f860"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original R source SHA-256 does not match the audited archive")
    source = raw.decode("utf-8")
    helpers = source[
        source.index("pava <- function") : source.index("################## simulation setup")
    ]
    start = source.index("tdose<-min(maxdose, toxdose-1)")
    end = source.index("\n", source.index("seldose<-c(mindose:tdose)", start))
    estimator = source[start:end]
    wrapper = r"""
cases <- list(
  list(n=c(3,9,6,0), x=c(1,1,3,0), target=.3, excluded=5),
  list(n=c(3,9,6,0), x=c(1,1,3,0), target=.1, excluded=5),
  list(n=c(0,0,6,3), x=c(0,0,2,2), target=.3, excluded=5),
  list(n=c(3,9,6,3), x=c(1,1,3,3), target=.3, excluded=3),
  list(n=c(3,3,3), x=c(0,0,0), target=.3, excluded=4),
  list(n=c(3,3,3), x=c(3,3,3), target=.3, excluded=4),
  list(n=c(6,6,6), x=c(3,2,1), target=.5, excluded=4),
  list(n=c(2,2), x=c(0,2), target=.5, excluded=3),
  list(n=c(1,199), x=c(1,0), target=.3, excluded=3),
  list(n=c(0,3,9,0), x=c(0,1,0,0), target=.3, excluded=5)
)
result <- list()
for (case_id in seq_along(cases)) {
  cfg <- cases[[case_id]]
  n <- cfg$n; x <- cfg$x; pT <- cfg$target; toxdose <- cfg$excluded
  mindose <- min(which(n > 0)); maxdose <- max(which(n > 0))
  ESTIMATOR
  for (dose in seq_along(n)) {
    result[[length(result)+1]] <- data.frame(
      case=case_id, dose=dose, patients=n[dose], toxicities=x[dose],
      target=pT, first_excluded_dose=toxdose, selected=seldose,
      score=if (dose <= tdose) pp[dose] else NA,
      eligible=as.integer(dose >= mindose && dose <= tdose))
  }
}
write.table(do.call(rbind, result), stdout(), sep=",", row.names=FALSE,
            col.names=TRUE, quote=FALSE, na="nan")
"""
    program = helpers + "\noptions(digits=17)\n" + wrapper.replace("ESTIMATOR", estimator)
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "reference.R"
        script.write_text(program)
        result = subprocess.run(
            [args.rscript, "--vanilla", str(script)], check=True, capture_output=True, text=True
        )
    args.output.write_text(result.stdout)
    print(f"Wrote original-R final-selection references to {args.output}")


if __name__ == "__main__":
    main()
