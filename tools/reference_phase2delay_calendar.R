# Independent base-R reference for Cai, Liu and Yuan (2014), simulation P28.
# No native application or Monte Carlo trial simulation is executed here.
# Run from the repository root: Rscript tools/reference_phase2delay_calendar.R
options(digits = 17, warn = 2)
window <- 3
cases <- expand.grid(event_probability = c(.1, .3, .6, .9), late_fraction = c(.7, .9))
hazard_end <- -log1p(-cases$event_probability)
hazard_half <- -log1p(-(1 - cases$late_fraction) * cases$event_probability)
cases$shape <- log(hazard_end / hazard_half) / log(2)
cases$scale <- window / hazard_end^(1 / cases$shape)
cases$window <- window
cases$probability_at_window <- pweibull(window, cases$shape, cases$scale)
cases$recovered_late_fraction <- (
  cases$probability_at_window - pweibull(window / 2, cases$shape, cases$scale)
) / cases$probability_at_window
stopifnot(max(abs(cases$probability_at_window - cases$event_probability)) < 1e-14)
stopifnot(max(abs(cases$recovered_late_fraction - cases$late_fraction)) < 1e-14)
stopifnot(max(abs(pweibull(7 * window, cases$shape, 7 * cases$scale) -
                 cases$probability_at_window)) < 1e-14)

probabilities <- c(.01, .1, .3, .5, .9, .99)
quantiles <- do.call(rbind, lapply(seq_len(nrow(cases)), function(index) {
  spec <- cases[index, ]
  times <- qweibull(probabilities, spec$shape, spec$scale)
  stopifnot(max(abs(pweibull(times, spec$shape, spec$scale) - probabilities)) < 1e-14)
  data.frame(case = index, probability = probabilities, time = times,
             observed_by_window = probabilities <= spec$event_probability)
}))
write.csv(cases, "tests/fixtures/phase2delay-weibull-calibration.csv", row.names = FALSE)
write.csv(quantiles, "tests/fixtures/phase2delay-weibull-quantiles.csv", row.names = FALSE)
cat(nrow(cases), "calibrations and", nrow(quantiles), "inverse-CDF values verified\n")
