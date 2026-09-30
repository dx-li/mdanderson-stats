# Independent base-R reference for IPDfromKM 0.1.10 report diagnostics.
# Source: CRAN IPDfromKM commit 16ea3e163b8ad409e51e035154c52803dcb1c28b,
# R/getIPD.R lines 287-306. R 4.4.1 stats::ks.test defaults are retained.
# Observed values are not rounded; fitted values and their differences are.

cases <- list(
  signed_error = list(
    observed = c(0.9, 0.8, 0.7, 0.6, 0.5),
    reconstructed = c(0.8, 0.7, 0.6, 0.5, 0.4)
  ),
  tied_exact = list(
    observed = c(1, 0.8, 0.8, 0.5, 0.3, 0.1),
    reconstructed = c(1, 0.9, 0.8, 0.6, 0.3, 0.1)
  ),
  tied_tail_exact = list(
    observed = c(rep(0.1, 3), rep(0.5, 3)),
    reconstructed = c(rep(0.5, 3), rep(0.9, 3))
  ),
  distinct_exact = list(
    observed = c(0.95, 0.82, 0.71, 0.58, 0.42, 0.31, 0.19, 0.08),
    reconstructed = c(0.96, 0.79, 0.74, 0.55, 0.45, 0.29, 0.17, 0.1)
  ),
  exact_separation = list(
    observed = c(0.4, 0.3, 0.2, 0.1, 0),
    reconstructed = c(0.9, 0.8, 0.7, 0.6, 0.5)
  ),
  exact_separation_99 = list(
    observed = seq(0, 0.098, by = 0.001),
    reconstructed = seq(0.901, 0.999, by = 0.001)
  ),
  decimal_rounding = list(
    observed = c(0.5005, 0.5015, 0.4995, 0.35, 0.25),
    reconstructed = c(0.5005, 0.5015, 0.4995, 0.351, 0.251)
  ),
  asymptotic_tied = list(
    observed = seq(0.01, 0.99, length.out = 100),
    reconstructed = rep(c(0.18, 0.33, 0.49, 0.66, 0.83), 20)
  ),
  identical = list(
    observed = c(1, 0.9, 0.8, 0.65, 0.5, 0.4, 0.2),
    reconstructed = c(1, 0.9, 0.8, 0.65, 0.5, 0.4, 0.2)
  ),
  paired_missing = list(
    observed = c(0.95, 0.8, NA, 0.55, 0.3, 0.1),
    reconstructed = c(0.9, 0.82, 0.5, 0.5, 0.35, 0.12)
  )
)

inputs <- list()
summary <- list()
for (name in names(cases)) {
  observed <- cases[[name]]$observed
  reconstructed <- cases[[name]]$reconstructed
  stopifnot(length(observed) == length(reconstructed))
  retained <- complete.cases(observed, reconstructed)
  observed_retained <- observed[retained]
  reconstructed_retained <- reconstructed[retained]
  fitted <- round(reconstructed_retained, 3)
  difference <- round(fitted - observed_retained, 3)
  ks <- suppressWarnings(stats::ks.test(fitted, observed_retained))
  fitted_full <- difference_full <- rep(NA_real_, length(observed))
  fitted_full[retained] <- fitted
  difference_full[retained] <- difference
  inputs[[length(inputs) + 1L]] <- data.frame(
    case = name,
    index = seq_along(observed),
    observed = observed,
    reconstructed = reconstructed,
    fitted_rounded = fitted_full,
    difference_rounded = difference_full,
    retained = retained
  )
  summary[[length(summary) + 1L]] <- data.frame(
    case = name,
    n = length(observed_retained),
    omitted = sum(!retained),
    rmse = round(sqrt(sum(difference^2) / (length(observed_retained) - 1)), 3),
    mean_absolute_error = round(sum(abs(difference)) / length(observed_retained), 3),
    legacy_signed_max_error = round(max(difference), 3),
    max_absolute_error = round(max(abs(difference)), 3),
    ks_statistic = unname(ks$statistic),
    ks_pvalue = unname(ks$p.value),
    ks_exact = isTRUE(ks$exact)
  )
}

directory <- file.path("tests", "fixtures")
write.csv(do.call(rbind, inputs), file.path(directory, "ipdfromkm-diagnostics-input.csv"),
          row.names = FALSE, na = "")
write.csv(do.call(rbind, summary), file.path(directory, "ipdfromkm-diagnostics-summary.csv"),
          row.names = FALSE, na = "")
