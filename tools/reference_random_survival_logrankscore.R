# Evaluate the pinned coin::logrank_trafo without installing or loading coin.
# The source file is cached locally under research/raw and is not distributed.
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1]] else "."
source(file.path(root, "research/raw/coin-logrankscore/Transformations.R"))
is.Surv <- function(x) inherits(x, "Surv")

cases <- list(
  list(name = "unique", time = c(1, 2, 3, 4, 5), event = c(1, 0, 1, 1, 0)),
  list(name = "ties", time = c(1, 1, 2, 2, 4, 4, 5), event = c(1, 0, 1, 1, 0, 1, 0)),
  list(name = "bootstrap_duplicates", time = c(1, 1, 2, 2, 2, 4), event = c(1, 1, 0, 1, 1, 0)),
  list(name = "censor_before_event", time = c(1, 2, 2, 3, 5), event = c(0, 1, 0, 1, 0))
)
output <- file(file.path(root, "tests/fixtures/random-survival-logrankscore-coin.csv"), "wt")
writeLines("case,row,time,event,coin_score,hothorn_lausen_score", output)
for (case in cases) {
  y <- structure(cbind(case$time, case$event), type = "right", class = c("Surv", "matrix", "array"))
  source_score <- as.numeric(logrank_trafo(y, ties.method = "Hothorn-Lausen"))
  for (i in seq_along(case$time)) {
    writeLines(sprintf("%s,%d,%.17g,%d,%.17g,%.17g", case$name, i,
                       case$time[i], case$event[i], source_score[i], -source_score[i]), output)
  }
}
close(output)
