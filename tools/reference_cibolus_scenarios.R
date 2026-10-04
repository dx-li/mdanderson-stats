# Independent base-R reference for CiBolus scenario interpolation.
# These synthetic endpoint grids check the published formulas; they do not
# reproduce scenarios 1--6 from the paper's separate Web Tables.
options(warn = 2, digits = 17)

args <- commandArgs(trailingOnly = TRUE)
repo_root <- if (length(args) >= 1L) normalizePath(args[[1]], mustWork = TRUE) else normalizePath(".", mustWork = TRUE)
out_dir <- if (length(args) >= 2L) args[[2]] else file.path(repo_root, "tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

concentrations <- c(0.15, 0.55)
bolus_fractions <- c(0, 0.35, 1)
endpoints <- c(0.13, 0.38, 0.50, 0.79, 1.00)
response_zero <- rbind(c(0, 0, 0), c(0.03, 0.10, 0.18))
response_one <- rbind(c(0.12, 0.12, 0.12), c(0.35, 0.56, 0.72))
toxicity_zero <- rbind(c(0.015, 0.020, 0.025), c(0.030, 0.050, 0.080))
toxicity_one <- rbind(c(0.050, 0.060, 0.070), c(0.170, 0.250, 0.380))
toxicity_failure <- rbind(c(0.11, 0.12, 0.13), c(0.27, 0.36, 0.49))
curve_names <- c("linear", "below_linear", "above_linear", "s_shaped")

shape <- function(s, curve) {
  switch(curve,
    linear = s,
    below_linear = s^2,
    above_linear = sqrt(s),
    s_shaped = {
      result <- numeric(length(s))
      lower <- s <= 0.5
      result[lower] <- 0.5 * (2 * s[lower])^2
      result[!lower] <- 0.5 + 0.5 * sqrt(2 * s[!lower] - 1)
      result
    },
    stop("unknown interpolation curve")
  )
}

utility <- cbind(
  no_toxicity = c(100, 94, 86, 74, 58, 39, 12),
  toxicity = c(7, 6, 5, 3, 1, 0, -8)
)
utility_rows <- data.frame(category_index = 0:6,
  category = c("bolus", "interval_1", "interval_2", "interval_3", "interval_4", "interval_5", "failure"),
  no_toxicity_utility = utility[, 1], toxicity_utility = utility[, 2])

settings_rows <- cell_rows <- marginal_rows <- utility_rows_out <- list()
for (response_curve in curve_names) for (toxicity_curve in curve_names) {
  case <- paste0("response_", response_curve, "__toxicity_", toxicity_curve)
  for (i in seq_along(concentrations)) for (j in seq_along(bolus_fractions)) {
    r0 <- response_zero[i, j]
    r1 <- response_one[i, j]
    t0 <- toxicity_zero[i, j]
    t1 <- toxicity_one[i, j]
    tf <- toxicity_failure[i, j]
    cdf <- r0 + (r1 - r0) * shape(endpoints, response_curve)
    tox_endpoint <- t0 + (t1 - t0) * shape(endpoints, toxicity_curve)
    response_mass <- c(r0, diff(c(r0, cdf)), 1 - r1)
    conditional_tox <- c(t0, tox_endpoint, tf)
    stopifnot(length(response_mass) == length(endpoints) + 2L,
      all(response_mass >= 0), all(response_mass <= 1),
      all(conditional_tox >= 0), all(conditional_tox <= 1),
      abs(sum(response_mass) - 1) < 1e-14)
    joint <- cbind(response_mass * (1 - conditional_tox), response_mass * conditional_tox)
    stopifnot(abs(sum(joint) - 1) < 1e-14)
    expected_utility <- sum(joint * utility)
    settings_rows[[length(settings_rows) + 1L]] <- data.frame(
      case, response_curve, toxicity_curve,
      concentration_index = i - 1L, bolus_index = j - 1L,
      concentration = concentrations[i], bolus_fraction = bolus_fractions[j],
      response_zero = r0, response_one = r1, toxicity_zero = t0,
      toxicity_one = t1, toxicity_failure = tf,
      expected_utility = expected_utility
    )
    for (k in seq_along(response_mass)) {
      category <- if (k == 1L) "bolus" else if (k == length(response_mass)) "failure" else paste0("interval_", k - 1L)
      start <- if (k <= 2L) 0 else endpoints[k - 2L]
      stop <- if (k == 1L) 0 else if (k == length(response_mass)) Inf else endpoints[k - 1L]
      cell_rows[[length(cell_rows) + 1L]] <- data.frame(
        case, concentration_index = i - 1L, bolus_index = j - 1L,
        category_index = k - 1L, category, interval_start = start, endpoint = stop,
        response_probability = response_mass[k], conditional_toxicity = conditional_tox[k],
        no_toxicity = joint[k, 1], toxicity = joint[k, 2]
      )
      marginal_rows[[length(marginal_rows) + 1L]] <- data.frame(
        case, concentration_index = i - 1L, bolus_index = j - 1L,
        category_index = k - 1L, response_probability = sum(joint[k, ])
      )
    }
    utility_rows_out[[length(utility_rows_out) + 1L]] <- data.frame(
      case, concentration_index = i - 1L, bolus_index = j - 1L,
      expected_utility
    )
  }
}

write.csv(do.call(rbind, settings_rows), file.path(out_dir, "cibolus-scenario-settings.csv"), row.names = FALSE)
write.csv(do.call(rbind, cell_rows), file.path(out_dir, "cibolus-scenario-cells.csv"), row.names = FALSE)
write.csv(do.call(rbind, marginal_rows), file.path(out_dir, "cibolus-scenario-marginals.csv"), row.names = FALSE)
write.csv(utility_rows, file.path(out_dir, "cibolus-scenario-utility.csv"), row.names = FALSE)
write.csv(do.call(rbind, utility_rows_out), file.path(out_dir, "cibolus-scenario-expected-utility.csv"), row.names = FALSE)

cat("cases:", length(curve_names)^2, "regimens per case:", length(concentrations) * length(bolus_fractions),
  "outcome cells:", nrow(do.call(rbind, cell_rows)), "\n")
