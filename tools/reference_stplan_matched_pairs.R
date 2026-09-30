#!/usr/bin/env Rscript
# Independent base-R mathematics for the STPLAN 4.5 matched-pairs cases.
# Native routines are separately called by reference_stplan_matched_pairs.f90.
options(digits = 17)

cases <- data.frame(
  case_id = c("known_asymmetric_one_sided", "known_asymmetric_two_sided",
              "pilot_counts_scaled_same_proportions", "inverse_difference", "inverse_alpha",
              "inverse_n", "low_target_n_native", "no_pilot_theta_unequal",
              "no_pilot_default", "forward_power_nonzero_delta",
              "low_target_no_pilot_lost_sign", "pilot_all_concordant_positive_delta"),
  iwhich = c(4, 4, 4, 1, 3, 2, 2, 5, 6, 4, 5, 4),
  sides = c(1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1),
  delta = c(.35, .35, .35, .5, .35, .35, .35, .3, .3, .05, .05, .1),
  n = c(rep(80, 9), 30, 80, 50),
  alpha = c(rep(.05, 9), .01, .05, .05),
  target = c(.8, .8, .8, .8, .8, .8, .1, .8, .8, .8, .01, .8),
  z11 = c(28, 28, 280, 28, 28, 28, 28, 0, 0, 28, 0, 20),
  z10 = c(17, 17, 170, 17, 17, 17, 17, .2, 0, 9, .2, 0),
  z01 = c(9, 9, 90, 9, 9, 9, 9, .65, 0, 17, .65, 0),
  z00 = c(26, 26, 260, 26, 26, 26, 26, 0, 0, 26, 0, 80)
)

psi_known <- function(delta, z11, z10, z01, z00) {
  a <- z10 + z01
  b <- z10 - z01
  total <- z11 + z10 + z01 + z00
  r <- (a + delta * b) / (2 * total)
  s <- r^2 - delta * (b - delta * (z11 + z00)) / total
  r + sqrt(s)
}

psi_uninformed <- function(theta1, theta2) {
  theta1 + theta2 - 2 * theta1 * theta2
}

power <- function(delta, n, alpha, sides, psi) {
  one_sided <- if (sides == 2) alpha / 2 else alpha
  za <- qnorm(1 - one_sided)
  denominator <- sqrt(psi^2 - .25 * delta^2 * (3 + psi))
  pnorm((-za * psi + abs(delta) * sqrt(n * psi)) / denominator)
}

get_psi <- function(row, delta) {
  if (row$iwhich == 5) return(psi_uninformed(row$z10, row$z01))
  if (row$iwhich == 6) return(psi_uninformed(.1, .9))
  psi_known(delta, row$z11, row$z10, row$z01, row$z00)
}

rows <- lapply(seq_len(nrow(cases)), function(i) {
  row <- cases[i, ]
  iw <- row$iwhich
  delta <- row$delta
  n <- row$n
  alpha <- row$alpha
  target <- row$target
  psi <- get_psi(row, delta)
  if (iw == 1) {
    delta <- uniroot(function(d) power(d, n, alpha, row$sides,
                                       get_psi(row, d)) - target,
                     c(1e-8, .99999999), tol = 1e-12)$root
    psi <- get_psi(row, delta)
  } else if (iw %in% c(2, 5, 6)) {
    if (iw == 5) psi <- psi_uninformed(row$z10, row$z01)
    if (iw == 6) psi <- psi_uninformed(.1, .9)
    if (iw == 5 && target < .5) {
      # Reproduce the native closed-form branch to expose its lost-sign case.
      one_sided <- if (row$sides == 2) alpha / 2 else alpha
      za <- qnorm(1 - one_sided)
      zp <- qnorm(target)
      y <- sqrt(psi^2 - .25 * delta^2 * (3 + psi))
      n <- (za * psi + zp * y)^2 / (psi * delta^2)
    } else {
      n <- uniroot(function(nn) power(delta, nn, alpha, row$sides, psi) - target,
                   c(1, 1e10), tol = 1e-10)$root
    }
  } else if (iw == 3) {
    alpha <- uniroot(function(a) power(delta, n, a, row$sides, psi) - target,
                     c(1e-8, .99999999), tol = 1e-12)$root
  }
  if (iw == 4) target <- power(delta, n, alpha, row$sides, psi)
  achieved <- power(delta, n, alpha, row$sides, psi)
  recommendation <- if (iw == 5) n / 4 else if (iw == 6) n / 6 else NA_real_
  data.frame(case_id = row$case_id, psi = psi, result = switch(as.character(iw),
    "1" = delta, "2" = n, "3" = alpha, "4" = target, "5" = n, "6" = n),
    achieved_power = achieved, recommendation = recommendation)
})
out <- do.call(rbind, rows)
write.csv(out, row.names = FALSE, na = "")
