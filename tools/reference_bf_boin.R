#!/usr/bin/env Rscript

# Bounded executable references for BF-BOIN.  This uses the public CRAN
# bfboin implementation as an independent reference; it is not a dump of the
# MD Anderson Shiny server.  Run with:
#   R_LIBS_USER=/tmp/bfboin-r-lib Rscript tools/reference_bf_boin.R

suppressPackageStartupMessages(library(bfboin))
suppressPackageStartupMessages(library(BOIN))
stopifnot(as.character(packageVersion("bfboin")) == "0.1.1")
stopifnot(as.character(packageVersion("BOIN")) == "2.7.2")

out_dir <- file.path(getwd(), "tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

trial_cases <- list(
  backfill_highest = list(seed = 1L, p.true = c(.05, .15, .30, .60),
    ncohort = 5L, cohortsize = 3L, n.cap = 5L, n.per.month = 6,
    p.response.true = rep(1, 4)),
  backfill_reaches_upper = list(seed = 3L, p.true = c(.05, .15, .30, .60),
    ncohort = 5L, cohortsize = 3L, n.cap = 5L, n.per.month = 6,
    p.response.true = rep(1, 4)),
  toxic_upper_conflict = list(seed = 5L, p.true = c(.10, .30, .55),
    ncohort = 5L, cohortsize = 3L, n.cap = 5L, n.per.month = 6,
    p.response.true = rep(1, 3))
)

write_trials <- function() {
  rows <- list()
  for (nm in names(trial_cases)) {
    z <- trial_cases[[nm]]
    set.seed(z$seed)
    x <- do.call(sim.one.trial, c(list(
      trial.id = z$seed, target = .25, n.earlystop = 100L,
      startdose = 1L, titration = FALSE, p.saf = .15, p.tox = .35,
      cutoff.eli = .95, extrasafe = FALSE, offset = .05,
      boundMTD = FALSE, end.backfill = TRUE, dlt.window = 1,
      three.plus.three = FALSE, accrual = "uniform",
      backfill.assign = "highest"), z[names(z) != "seed"]))
    stopifnot(all(x$n >= 0), all(x$y >= 0), all(x$y <= x$n),
              length(unique(x$dselect)) == 1L)
    for (i in seq_len(nrow(x))) rows[[length(rows) + 1L]] <- data.frame(
      case = nm, seed = z$seed, target = .25, ncohort = z$ncohort,
      cohortsize = z$cohortsize, n_cap = z$n.cap, n_per_month = z$n.per.month,
      dlt_window = 1, dose = x$d[i], patients = x$n[i], toxicities = x$y[i],
      selected_mtd = x$dselect[i], max_time = sprintf("%.17g", x$max.t[i]),
      stringsAsFactors = FALSE)
  }
  write.csv(do.call(rbind, rows), file.path(out_dir, "bf-boin-trials.csv"), row.names = FALSE)
}

write_boundaries <- function() {
  b <- get.boundary(.25, ncohort = 35, cohortsize = 1, n.earlystop = 100)
  t <- b$boundary_tab
  out <- data.frame(n = as.integer(t[1, ]),
    escalate_if_dlt_le = as.integer(t[2, ]),
    deescalate_if_dlt_ge = as.integer(t[3, ]),
    eliminate_if_dlt_ge = as.integer(t[4, ]),
    stringsAsFactors = FALSE)
  write.csv(out, file.path(out_dir, "bf-boin-boundaries.csv"), row.names = FALSE)
}

write_trials(); write_boundaries()
