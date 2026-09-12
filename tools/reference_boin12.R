#!/usr/bin/env Rscript

# Generate bounded native BOIN12 references. The runner uses the CRAN
# escalation implementation of the published BOIN12 algorithm and is kept
# outside the Python runtime.
#
#   R_LIBS_USER=/tmp/boin12-r-lib Rscript tools/reference_boin12.R

suppressPackageStartupMessages(library(escalation))
suppressPackageStartupMessages(library(magrittr))
stopifnot(as.character(packageVersion("escalation")) == "0.2.3")

out_dir <- file.path(getwd(), "tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

make_data <- function(dose, tox, eff) {
  data.frame(Dose = as.integer(dose), Toxicity = as.integer(tox),
             Efficacy = as.integer(eff))
}

counts_data <- function(n, y, e) {
  rows <- list()
  for (d in seq_along(n)) {
    if (n[d] > 0) rows[[length(rows) + 1L]] <- make_data(
      rep(d, n[d]), y[d], e[d]
    )
  }
  if (length(rows) == 0L) make_data(integer(), integer(), integer()) else do.call(rbind, rows)
}

utility_benchmark <- function(phi_t, phi_e, u1, u2, u3, u4 = 0) {
  u_bar <- u1 * (1 - phi_t) * phi_e + u2 * (1 - phi_t) * (1 - phi_e) +
    u3 * phi_t * phi_e + u4 * phi_t * (1 - phi_e)
  u_bar + (100 - u_bar) / 2
}

write_posterior <- function() {
  phi_t <- .35; phi_e <- .25; u1 <- 100; u2 <- 40; u3 <- 60; u4 <- 0
  ub <- utility_benchmark(phi_t, phi_e, u1, u2, u3, u4)
  cases <- list(
    published = list(n = c(3, 3, 3, 3, 0), y = c(0, 1, 2, 0, 0), e = c(0, 1, 1, 2, 0)),
    noninteger_quasi = list(n = c(4, 5, 6, 0, 0), y = c(1, 2, 2, 0, 0), e = c(2, 3, 4, 0, 0)),
    safety_efficiency_edges = list(n = c(3, 6, 9, 0, 0), y = c(0, 3, 6, 0, 0), e = c(0, 1, 5, 0, 0))
  )
  rows <- list()
  for (nm in names(cases)) {
    z <- cases[[nm]]; dat <- counts_data(z$n, z$y, z$e)
    p <- escalation:::boin12_pbenchmark(1:5, dat, u1, u2, u3, u4, ub)
    for (d in seq_len(5)) {
      n <- z$n[d]; y <- z$y[d]; e <- z$e[d]
      rows[[length(rows) + 1L]] <- data.frame(
        case = nm, dose = d, patients = n, toxicities = y, efficacies = e,
        utility_benchmark = ub,
        posterior_utility_gt_benchmark = p[d],
        posterior_tox_gt_limit = pbeta(phi_t, 1 + y, 1 + n - y, lower.tail = FALSE),
        posterior_eff_lt_limit = pbeta(phi_e, 1 + e, 1 + n - e, lower.tail = TRUE),
        admissible = p[d] == p[d] &&
          pbeta(phi_t, 1 + y, 1 + n - y, lower.tail = FALSE) < .95 &&
          pbeta(phi_e, 1 + e, 1 + n - e, lower.tail = TRUE) < .90
      )
    }
  }
  write.csv(do.call(rbind, rows), file.path(out_dir, "boin12-posterior.csv"), row.names = FALSE)
}

write_rds <- function() {
  tab <- boin12_rds(sample_sizes = c(0, 3, 6, 9), phi_t = .35, phi_e = .25,
                    u1 = 100, u2 = 40, u3 = 60, u4 = 0,
                    c_t = .95, c_e = .90, prior_alpha = 1, prior_beta = 1)
  tab$RDS_x <- as.numeric(tab$RDS_x)
  write.csv(tab, file.path(out_dir, "boin12-rds.csv"), row.names = FALSE, na = "NA")
}

run_decision <- function(case, dat, ...) {
  x <- escalation:::boin12_next_dose(data = dat, ndoses = 5, maxN = 36, ...)
  data.frame(
    case = case, current_dose = if (nrow(dat)) dat$Dose[nrow(dat)] else NA,
    patients_at_current = if (nrow(dat)) sum(dat$Dose == dat$Dose[nrow(dat)]) else 0,
    tox_at_current = if (nrow(dat)) sum(dat$Dose == dat$Dose[nrow(dat)] & dat$Toxicity == 1) else 0,
    efficacy_at_current = if (nrow(dat)) sum(dat$Dose == dat$Dose[nrow(dat)] & dat$Efficacy == 1) else 0,
    next_dose = x$next_dose,
    admissible = paste(x$admissible, collapse = ";"),
    utility = paste(sprintf("%.17g", x$utility), collapse = ";"),
    stringsAsFactors = FALSE
  )
}

write_decisions <- function() {
  rows <- list()
  rows[[1]] <- run_decision("empty_start", make_data(integer(), integer(), integer()),
                            start = 1, phi_t = .35, phi_e = .25, u2 = 40, u3 = 60, Nstar = 6)
  rows[[2]] <- run_decision("published_example", make_data(
    c(rep(1, 3), rep(2, 3), rep(3, 3), rep(2, 3)),
    c(0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0),
    c(0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0)),
    phi_t = .35, phi_e = .25, u2 = 40, u3 = 60, Nstar = 6)
  rows[[3]] <- run_decision("stay_interval_rds", make_data(
    c(rep(1, 3), rep(2, 6), rep(3, 3)),
    c(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1),
    c(0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1)),
    phi_t = .35, phi_e = .25, u2 = 40, u3 = 60, Nstar = 6)
  rows[[4]] <- run_decision("extra_exploration_at_nine", make_data(rep(1, 9), rep(0, 9), rep(0, 9)),
                            phi_t = .35, phi_e = .25, u2 = 40, u3 = 60, Nstar = 6)
  rows[[5]] <- run_decision("deescalate_toxic_current", make_data(rep(3, 3), rep(1, 3), rep(0, 3)),
                            phi_t = .35, phi_e = .25, u2 = 40, u3 = 60, Nstar = 6)
  rows[[6]] <- run_decision("custom_prior_and_exploration", make_data(rep(2, 6), c(0, 0, 1, 0, 0, 1), c(1, 0, 1, 1, 0, 0)),
                            phi_t = .35, phi_e = .25, u2 = 40, u3 = 60, Nstar = 6,
                            alpha = 0.5, beta = 2, maxN_dose = 12)
  write.csv(do.call(rbind, rows), file.path(out_dir, "boin12-decisions.csv"), row.names = FALSE, na = "NA")
}

write_obd <- function() {
  cases <- list(
    published_example = "1NNN 2ENT 3ETT 2EEN",
    plateau_efficacy = "1NNN 2NNN 3NNN 4EEE 4EEE",
    toxicity_isotonic_pool = "1NNN 2NNT 2NNT 3ETT 3NNN 4EEE 4EEE"
  )
  model <- get_boin12(num_doses = 5, phi_t = .35, phi_e = .25,
                      u1 = 100, u2 = 40, u3 = 60, u4 = 0, n_star = 6,
                      c_t = .95, c_e = .90, prior_alpha = 1, prior_beta = 1) %>%
    select_boin12_obd(when = "always")
  rows <- list()
  for (nm in names(cases)) {
    fit <- model %>% fit(cases[[nm]])
    etr <- empiric_tox_rate(fit); names(etr) <- dose_indices(fit)
    given <- n_at_dose(fit) > 0; pava <- escalation:::pava(etr[given])
    mtd <- tail(which(abs(pava - .35) == min(abs(pava - .35))), 1)
    rows[[length(rows) + 1L]] <- data.frame(
      case = nm, outcomes = cases[[nm]], obd = recommended_dose(fit), mtd = mtd,
      isotonic_toxicity = paste(sprintf("%.17g", pava), collapse = ";"),
      utility = paste(sprintf("%.17g", utility(fit)), collapse = ";"),
      admissible = paste(as.integer(dose_admissible(fit)), collapse = ";"),
      stringsAsFactors = FALSE
    )
  }
  write.csv(do.call(rbind, rows), file.path(out_dir, "boin12-obd.csv"), row.names = FALSE)
}

write_posterior(); write_rds(); write_decisions(); write_obd()
