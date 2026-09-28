# Independent geometry references for Thall and Cook (2004), pp. 686--687.
# This evaluates the published formula; it does not execute native EffTox.
options(digits = 17)

contours <- list(
  pentostatin = list(e = c(.15, .25, 1), t = c(0, .30, .60)),
  inverse_linear = list(e = c(.1, .5, 1), t = c(0, .64, .72)),
  inverse_square = list(e = c(.2, .6, 1), t = c(0, 4 / 9, .48)),
  flat_lower_endpoint = list(e = c(.2, .5, 1), t = c(0, .28125, .5))
)
query <- rbind(
  c(0, 0), c(0, 1), c(1, 1), c(1, .25), c(.5, 0),
  c(.15, .3), c(.4, .15), c(.8, .1), c(.99, .001)
)
rows <- list()
for (name in names(contours)) {
  contour <- contours[[name]]
  coefficients <- solve(cbind(1, 1 / contour$e, 1 / contour$e^2), contour$t)
  f <- function(e) sum(coefficients * c(1, 1 / e, 1 / e^2))
  points <- rbind(cbind(contour$e, contour$t), query)
  # For these q values, the exact published radial ratio is 2, so utility=1.
  points <- rbind(points, cbind(1 - (1 - contour$e) / 2, contour$t / 2))
  for (i in seq_len(nrow(points))) {
    e <- points[i, 1]
    t <- points[i, 2]
    if (e == 1) {
      p_e <- 1
      p_t <- f(1)
    } else if (t == 0) {
      p_e <- contour$e[1]
      p_t <- 0
    } else {
      # Cross-multiplied ray equation avoids forming t/(1-e).
      equation <- function(u) (1 - e) * f(u) - t * (1 - u)
      p_e <- uniroot(equation, c(contour$e[1], 1), tol = 1e-14)$root
      p_t <- f(p_e)
    }
    rho_p <- sqrt((1 - p_e)^2 + p_t^2)
    rho_q <- sqrt((1 - e)^2 + t^2)
    rows[[length(rows) + 1]] <- data.frame(
      contour = name, efficacy_intercept = contour$e[1],
      middle_efficacy = contour$e[2], middle_toxicity = contour$t[2],
      toxicity_intercept = contour$t[3], a = coefficients[1],
      b = coefficients[2], c = coefficients[3], efficacy = e, toxicity = t,
      intersection_efficacy = p_e, intersection_toxicity = p_t,
      utility = rho_p / rho_q - 1
    )
  }
}
output <- do.call(rbind, rows)
write.csv(output, "tests/fixtures/efftox-legacy-contour.csv", row.names = FALSE)
cat("Wrote", nrow(output), "independent contour reference rows.\n")
