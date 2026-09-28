# Independent augmented-system thin-plate spline calculation.
# Baseline: common-intercept raw-dose regression on marginal observations.
# Penalized criterion: sum((residual - surface)^2) + lambda * nu' K nu.
options(digits = 17)
knots <- expand.grid(dose1 = c(0, .5, 1), dose2 = c(0, .75, 1.5))
index <- c(1:9, 1, 3, 5, 9)
doses <- knots[index, ]
y <- 1.1 - .2 * doses$dose1 - .15 * doses$dose2 -
  .3 * doses$dose1 * doses$dose2 + .025 * sin(seq_along(index))
design <- cbind(1, as.matrix(doses))
marginal <- doses$dose1 == 0 | doses$dose2 == 0
beta <- as.numeric(qr.solve(design[marginal, ], y[marginal]))
baseline <- as.numeric(design %*% beta)
residual <- (y - baseline) * !marginal
counts <- tabulate(index, nrow(knots))
mean_residual <- as.numeric(rowsum(residual, index)) / counts
kernel <- function(left, right) {
  distance_squared <- outer(left[, 1], right[, 1], "-")^2 +
    outer(left[, 2], right[, 2], "-")^2
  result <- matrix(0, nrow(left), nrow(right))
  positive <- distance_squared > 0
  result[positive] <- distance_squared[positive] * log(distance_squared[positive]) /
    (16 * pi)
  result
}
knots <- as.matrix(knots)
k <- kernel(knots, knots)
t_design <- cbind(1, knots)
points <- rbind(knots, c(.2, .6), c(.7, 1.2), c(0, 1.1))
predictions <- list()
coefficients <- list()
for (lambda in c(.02, .2)) {
  lhs <- rbind(cbind(k + lambda * diag(1 / counts), t_design),
               cbind(t(t_design), matrix(0, 3, 3)))
  solution <- solve(lhs, c(mean_residual, rep(0, 3)))
  nu <- solution[seq_len(nrow(knots))]
  affine <- tail(solution, 3)
  fitted <- as.numeric(k %*% nu + t_design %*% affine)
  stopifnot(max(abs(crossprod(t_design, nu))) < 1e-12,
            max(abs(fitted + lambda * nu / counts - mean_residual)) < 1e-12)
  surface <- as.numeric(kernel(points, knots) %*% nu + cbind(1, points) %*% affine)
  predictions[[length(predictions) + 1]] <- data.frame(
    lambda, dose1 = points[, 1], dose2 = points[, 2],
    baseline = as.numeric(cbind(1, points) %*% beta), surface = surface,
    response = as.numeric(cbind(1, points) %*% beta) + surface
  )
  coefficients[[length(coefficients) + 1]] <- data.frame(
    lambda, baseline_intercept = beta[1], baseline_dose1 = beta[2],
    baseline_dose2 = beta[3], affine_intercept = affine[1],
    affine_dose1 = affine[2], affine_dose2 = affine[3],
    residual_sum_squares = sum((residual - fitted[index])^2),
    roughness = as.numeric(crossprod(nu, k %*% nu))
  )
}
write.csv(data.frame(doses, response = y),
          "tests/fixtures/synergy-surface-inputs.csv", row.names = FALSE)
write.csv(do.call(rbind, predictions),
          "tests/fixtures/synergy-surface-predictions.csv", row.names = FALSE)
write.csv(do.call(rbind, coefficients),
          "tests/fixtures/synergy-surface-coefficients.csv", row.names = FALSE)

# Direct full-covariance REML, independent of the Python error-contrast spectrum.
null <- qr.Q(qr(t_design), complete = TRUE)[, 4:nrow(knots)]
penalty <- crossprod(null, k %*% null)
random_design <- k[index, ] %*% null
random_covariance <- random_design %*% solve(penalty, t(random_design))
solve_chol <- function(factor, rhs) {
  backsolve(factor, forwardsolve(t(factor), rhs))
}
reml <- function(log_lambda) {
  covariance <- diag(length(y)) + random_covariance / exp(log_lambda)
  factor <- chol(covariance)
  solved_design <- solve_chol(factor, design)
  info_factor <- chol(crossprod(design, solved_design))
  affine <- solve_chol(info_factor, crossprod(design, solve_chol(factor, residual)))
  error <- residual - as.numeric(design %*% affine)
  df <- length(y) - ncol(design)
  variance <- sum(error * solve_chol(factor, error)) / df
  logdet <- 2 * sum(log(diag(factor))) + 2 * sum(log(diag(info_factor))) -
    as.numeric(determinant(crossprod(design), logarithm = TRUE)$modulus)
  objective <- .5 * (df * (log(2 * pi) + 1 + log(variance)) + logdet)
  c(objective = objective, variance = variance)
}
optimum <- optimize(function(value) reml(value)[1], c(-20, 10), tol = 1e-10)
stopifnot(optimum$minimum > -19, optimum$minimum < 9)
write.csv(data.frame(lambda = exp(optimum$minimum),
                     objective = optimum$objective,
                     residual_variance = reml(optimum$minimum)[2]),
          "tests/fixtures/synergy-surface-reml.csv", row.names = FALSE)
cat("Wrote augmented-system TPS fits and independently profiled REML reference.\n")
