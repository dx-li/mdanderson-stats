# Independent supplied-pair transforms and propagated covariance, using base R.
# This is mathematical reference data, not output from the native WFMM binary.
options(digits = 17, warn = 2)
cases <- list(
  shear = matrix(c(1, 2, 0, 1), 2, byrow = TRUE),
  triangular = matrix(c(1, 1, 0, 0, 2, 1, 0, 0, 4), 3, byrow = TRUE),
  nonsymmetric = matrix(c(2, 0, 0, 1, 0, 1, 1, 0,
                         0, 0, 3, 0, 0, 0, 0, 4), 4, byrow = TRUE),
  large_analysis = 1e100 * matrix(c(1, 2, 0, 1), 2, byrow = TRUE),
  small_analysis = 1e-100 * matrix(c(1, 2, 0, 1), 2, byrow = TRUE)
)
scales <- c(shear = 1, triangular = 1, nonsymmetric = 1,
            large_analysis = 1e100, small_analysis = 1e-100)
matrix_rows <- list()
curve_rows <- list()
covariance_rows <- list()
summary_rows <- list()
for (name in names(cases)) {
  analysis <- cases[[name]]
  dimension <- nrow(analysis)
  synthesis <- solve(analysis)
  stopifnot(max(abs(analysis %*% synthesis - diag(dimension))) < 1e-14,
            max(abs(synthesis %*% analysis - diag(dimension))) < 1e-14)
  indices <- expand.grid(row = seq_len(dimension), column = seq_len(dimension))
  matrix_rows[[name]] <- data.frame(
    case = name, row = indices$row - 1, column = indices$column - 1,
    analysis = analysis[cbind(indices$row, indices$column)],
    synthesis = synthesis[cbind(indices$row, indices$column)]
  )
  # Sixteen rows: two chains, four retained draws, two functional effects.
  reconstructed <- array(NA_real_, c(8, 2, dimension))
  records <- list()
  for (sample in seq_len(8)) {
    for (effect in seq_len(2)) {
      observed <- cos(seq_len(dimension) * .2) +
        .15 * sin(sample + seq_len(dimension)) + effect * sample / 20
      coefficient <- as.numeric(observed %*% analysis)
      restored <- as.numeric(coefficient %*% synthesis)
      stopifnot(max(abs(restored - observed)) < 2e-14)
      reconstructed[sample, effect, ] <- restored
      records[[length(records) + 1]] <- data.frame(
        case = name, chain = (sample - 1) %/% 4, draw = (sample - 1) %% 4,
        effect = effect - 1, coordinate = seq_len(dimension) - 1,
        observed = observed, coefficient = coefficient, reconstructed = restored
      )
    }
  }
  curve_rows[[name]] <- do.call(rbind, records)
  omega <- seq_len(dimension) * .2 * scales[[name]]^2
  # Work with standard-deviation-scaled rows to avoid premature squaring.
  weighted_synthesis <- sqrt(omega) * synthesis
  covariance <- crossprod(weighted_synthesis)
  covariance_rows[[name]] <- data.frame(
    case = name, row = indices$row - 1, column = indices$column - 1,
    variance = omega[indices$row],
    covariance = covariance[cbind(indices$row, indices$column)]
  )
  records <- list()
  for (effect in seq_len(2)) {
    draws <- reconstructed[, effect, ]
    means <- colMeans(draws)
    sds <- apply(draws, 2, sd)
    quantiles <- apply(draws, 2, quantile, probs = c(.1, .5, .9), type = 7)
    standardized <- sweep(sweep(draws, 2, means), 2, sds, "/")
    critical <- as.numeric(quantile(apply(abs(standardized), 1, max), .9, type = 7))
    records[[effect]] <- data.frame(
      case = name, effect = effect - 1, coordinate = seq_len(dimension) - 1,
      mean = means, sd = sds, lower_quantile = quantiles[1, ],
      median = quantiles[2, ], upper_quantile = quantiles[3, ],
      simultaneous_lower = means - critical * sds,
      simultaneous_upper = means + critical * sds, critical = critical
    )
  }
  summary_rows[[name]] <- do.call(rbind, records)
}
for (item in list(matrices = matrix_rows, curves = curve_rows,
                  covariance = covariance_rows, summaries = summary_rows) |> names()) {
  values <- switch(item, matrices = matrix_rows, curves = curve_rows,
                   covariance = covariance_rows, summaries = summary_rows)
  write.csv(do.call(rbind, values),
            paste0("tests/fixtures/wfmm-custom-", item, ".csv"), row.names = FALSE)
}
cat("Wrote five supplied-pair WFMM transform and covariance reference cases.\n")
