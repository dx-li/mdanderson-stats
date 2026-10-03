# Independent fixed-tape wild-bootstrap reference for the raw-dose SYNERGY fit.
# Uses a direct augmented TPS solve for fixed lambda and full-covariance GLS
# REML via determinant/solve, rather than Python's error-contrast eigensystem.
# Base R only; writes compact CSV fixtures to the optional output directory.
options(digits = 17)
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else "."
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

knots <- expand.grid(dose1 = c(0, .5, 1), dose2 = c(0, .75, 1.5))
index <- c(1:9, 1, 3, 5, 9)
doses <- knots[index, , drop = FALSE]
Y <- 1.1 - .2 * doses$dose1 - .15 * doses$dose2 -
  .3 * doses$dose1 * doses$dose2 + .025 * sin(seq_along(index))
d1 <- doses$dose1
d2 <- doses$dose2
n <- length(Y)
K <- as.matrix(knots)
combination <- d1 > 0 & d2 > 0
marginal <- !combination
counts <- tabulate(index, nbins = nrow(K))
stopifnot(all(counts > 0), nrow(K) == 9L, n == 13L)

kernel <- function(left, right) {
  a <- as.matrix(left)
  b <- as.matrix(right)
  d1sq <- outer(a[, 1], b[, 1], "-")^2
  d2sq <- outer(a[, 2], b[, 2], "-")^2
  r2 <- d1sq + d2sq
  ans <- matrix(0, nrow(a), nrow(b))
  positive <- r2 > 0
  ans[positive] <- r2[positive] * log(r2[positive]) / (16 * pi)
  ans
}
Omega <- kernel(K, K)
T <- cbind(1, as.matrix(K))
X <- cbind(1, d1, d2)

fit_baseline <- function(y) {
  beta <- as.numeric(qr.solve(X[marginal, , drop = FALSE], y[marginal]))
  list(beta = beta, fitted = as.numeric(X %*% beta))
}
fit_fixed <- function(y, lambda) {
  base <- fit_baseline(y)
  residual <- numeric(n)
  residual[combination] <- y[combination] - base$fitted[combination]
  average <- as.numeric(rowsum(residual, index, reorder = FALSE)) / counts
  lhs <- rbind(
    cbind(Omega + lambda * diag(1 / counts), T),
    cbind(t(T), matrix(0, 3, 3))
  )
  solution <- solve(lhs, c(average, rep(0, 3)))
  nu <- solution[seq_len(nrow(K))]
  affine <- tail(solution, 3)
  departure <- as.numeric(kernel(doses, K) %*% nu + X %*% affine)
  list(beta = base$beta, baseline = base$fitted, departure = departure,
       response_fit = base$fitted + departure, nu = nu, affine = affine,
       residual = residual)
}

# Full covariance REML profile from the independent reference script, using
# direct solves and determinants. This explicitly returns variance on input-Y
# units (Python's stored field is response-scale-normalized).
null <- qr.Q(qr(T), complete = TRUE)[, 4:nrow(K), drop = FALSE]
Z <- Omega[index, , drop = FALSE] %*% null
penalty <- crossprod(null, Omega %*% null)
G <- Z %*% solve(penalty, t(Z))
logdet <- function(a) {
  value <- determinant(a, logarithm = TRUE)
  if (value$sign <= 0) stop("nonpositive determinant in independent REML")
  as.numeric(value$modulus)
}
reml_profile <- function(y, log_lambda) {
  lambda <- exp(log_lambda)
  base <- fit_baseline(y)
  residual <- numeric(n)
  residual[combination] <- y[combination] - base$fitted[combination]
  V <- diag(n) + G / lambda
  ViX <- solve(V, X)
  Vir <- solve(V, residual)
  information <- crossprod(X, ViX)
  beta <- solve(information, crossprod(X, Vir))
  error <- residual - as.numeric(X %*% beta)
  Vie <- solve(V, error)
  df <- n - ncol(X)
  variance <- as.numeric(crossprod(error, Vie)) / df
  if (!is.finite(variance) || variance <= 0) return(c(objective = Inf, variance = 0))
  objective <- .5 * (df * (log(2 * pi) + 1 + log(variance)) +
    logdet(V) + logdet(information) - logdet(crossprod(X)))
  c(objective = objective, variance = variance)
}
fit_reml <- function(y) {
  optimum <- optimize(function(ll) reml_profile(y, ll)[["objective"]],
                      interval = c(-30, 30), tol = 1e-10)
  lambda <- exp(optimum$minimum)
  profile <- reml_profile(y, optimum$minimum)
  if (!is.finite(profile[["objective"]]) || profile[["variance"]] <= 0)
    stop("independent REML optimization failed")
  c(lambda = lambda, objective = profile[["objective"]],
    residual_variance = profile[["variance"]], log_lambda = optimum$minimum)
}

sqrt5 <- sqrt(5)
wminus <- (1 - sqrt5) / 2
wplus <- (1 + sqrt5) / 2
pminus <- (sqrt5 + 1) / (2 * sqrt5)
pplus <- (sqrt5 - 1) / (2 * sqrt5)
# Deterministic tapes; rows are bootstrap replicates, columns are observations.
# The four replicated-dose pairs receive different weights within replicate.
tape_index <- rbind(
  c(1,2,1,2,1,2,1,2,1,2,1,2,1),
  c(2,1,2,1,2,1,2,1,2,1,2,1,2),
  c(1,1,2,2,1,2,2,1,1,2,1,1,2),
  c(2,2,1,1,2,1,1,2,2,1,2,2,1),
  c(1,2,2,1,2,1,1,2,1,1,2,2,1)
)
weights <- matrix(c(wminus, wplus)[as.vector(tape_index)], nrow = nrow(tape_index), byrow = TRUE)
# Verify the fixed tape exercises rowwise residual weights at repeated knots.
for (duplicate_id in c(1L, 3L, 5L, 9L)) {
  rows <- which(index == duplicate_id)
  stopifnot(length(rows) == 2L,
            any(weights[, rows[1]] != weights[, rows[2]]))
}

original_lambda <- fit_reml(Y)
original_fixed <- fit_fixed(Y, original_lambda[["lambda"]])
lower_lambda <- original_lambda[["lambda"]] / 2
upper_lambda <- original_lambda[["lambda"]] * 2
lower <- fit_fixed(Y, lower_lambda)
upper <- fit_fixed(Y, upper_lambda)
residual <- Y - lower$baseline - lower$departure
center <- original_fixed$baseline + upper$departure

input_fixture <- data.frame(
  observation = seq_len(n), index = index, dose1 = d1, dose2 = d2,
  response = Y, combination = combination,
  baseline_original = original_fixed$baseline,
  original_departure = original_fixed$departure,
  lower_lambda = lower_lambda,
  lower_baseline = lower$baseline,
  lower_departure = lower$departure,
  upper_lambda = upper_lambda,
  upper_baseline = upper$baseline,
  upper_departure = upper$departure,
  wild_residual = residual
)
write.csv(input_fixture, file.path(out_dir, "synergy-bootstrap-anchors.csv"), row.names = FALSE)
write.csv(data.frame(
  original_lambda = original_lambda[["lambda"]],
  original_reml_objective = original_lambda[["objective"]],
  original_residual_variance = original_lambda[["residual_variance"]],
  lower_lambda = lower_lambda, upper_lambda = upper_lambda,
  mammen_minus = wminus, mammen_plus = wplus,
  probability_minus = pminus, probability_plus = pplus
), file.path(out_dir, "synergy-bootstrap-reml.csv"), row.names = FALSE)

draw_rows <- vector("list", nrow(weights))
for (b in seq_len(nrow(weights))) {
  synthetic <- center + residual * weights[b, ]
  refit <- fit_reml(synthetic)
  fitted <- fit_fixed(synthetic, refit[["lambda"]])
  draw_rows[[b]] <- data.frame(
    replicate = b, observation = seq_len(n), index = index,
    dose1 = d1, dose2 = d2, original_response = Y,
    baseline_original = original_fixed$baseline,
    original_departure = original_fixed$departure,
    lower_lambda = lower_lambda, lower_departure = lower$departure,
    upper_lambda = upper_lambda, upper_departure = upper$departure,
    wild_residual = residual, multiplier = weights[b, ],
    synthetic_response = synthetic,
    refit_lambda = refit[["lambda"]],
    refit_reml_objective = refit[["objective"]],
    refit_residual_variance = refit[["residual_variance"]],
    refit_baseline = fitted$baseline,
    departure_draw = fitted$departure,
    fitted_response = fitted$response_fit
  )
}
draws <- do.call(rbind, draw_rows)
write.csv(draws, file.path(out_dir, "synergy-bootstrap-draws.csv"), row.names = FALSE)
wide <- matrix(draws$departure_draw, nrow = nrow(weights), byrow = TRUE)
summary_fixture <- data.frame(
  observation = seq_len(n), dose1 = d1, dose2 = d2, index = index,
  original_departure = original_fixed$departure,
  departure_mean = colMeans(wide),
  departure_sd_ddof1 = apply(wide, 2, sd),
  replicates = nrow(wide)
)
write.csv(summary_fixture, file.path(out_dir, "synergy-bootstrap-summary.csv"), row.names = FALSE)
cat(sprintf("Wrote %d fixed-tape replicates over %d observations; lambda=%.17g, lambda/2=%.17g, 2*lambda=%.17g.\n",
            nrow(weights), n, original_lambda[["lambda"]], lower_lambda, upper_lambda))
