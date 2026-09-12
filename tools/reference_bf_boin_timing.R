#!/usr/bin/env Rscript
# Small independent reference for bfboin 0.1.1's Weibull timing calibration.
# Only base R is required. No simulations or package installation are run.
p <- c(1e-8, 0.05, 0.25, 0.6, 0.95, 1 - 1e-8)
window <- 1
a <- -log1p(-p)
b <- -log1p(-p / 2)
shape <- (log(a) - log(b)) / log(2)
scale <- exp(log(window) - log(a) / shape)
out <- data.frame(
  probability = p, window = window, shape = shape, scale = scale,
  cdf_half_window = pweibull(window / 2, shape, scale),
  cdf_window = pweibull(window, shape, scale)
)
stopifnot(max(abs(out$cdf_half_window - p / 2)) < 1e-13,
          max(abs(out$cdf_window - p)) < 1e-13)
options(digits = 17)
write.csv(out, "tests/fixtures/bf-boin-timing.csv", row.names = FALSE)
