# Independent Gaussian-mixture integration for a reduced WFMM model.
# Variance components are fixed here solely to obtain exact posterior references.
options(digits = 17)
x <- cbind(intercept = 1, covariate = c(-1, -.5, 0, .5, 1))
z <- model.matrix(~ factor(c(1, 1, 2, 2, 3)) - 1)
d <- cbind(c(.2, .6, -.1, .5, 1.1), c(-.4, .1, .7, .3, -.2))
colnames(z) <- paste0("z", 1:ncol(z))
colnames(d) <- paste0("d", 1:ncol(d))
q <- c(.3, .1)
s <- c(.2, .4)
pi <- c(.7, .4)
tau <- c(.8, .5)  # variances, not standard deviations
states <- as.matrix(expand.grid(gamma_1 = 0:1, gamma_2 = 0:1))
solve_chol <- function(chol_upper, rhs) {
  backsolve(chol_upper, forwardsolve(t(chol_upper), rhs))
}
results <- list()
for (k in 1:ncol(d)) {
  sigma <- q[k] * tcrossprod(z) + s[k] * diag(nrow(x))
  log_weight <- numeric(nrow(states))
  means <- matrix(0, nrow(states), ncol(x))
  covariances <- array(0, c(ncol(x), ncol(x), nrow(states)))
  for (h in 1:nrow(states)) {
    gamma <- states[h, ]
    prior_cov <- diag(tau * gamma)
    marginal_cov <- sigma + x %*% prior_cov %*% t(x)
    factor <- chol(marginal_cov)
    solved_d <- solve_chol(factor, d[, k])
    log_weight[h] <- sum(ifelse(gamma == 1, log(pi), log1p(-pi))) -
      .5 * nrow(x) * log(2 * base::pi) -
      sum(log(diag(factor))) - .5 * sum(d[, k] * solved_d)
    means[h, ] <- prior_cov %*% crossprod(x, solved_d)
    covariances[, , h] <- prior_cov -
      prior_cov %*% crossprod(x, solve_chol(factor, x)) %*% prior_cov
  }
  weight <- exp(log_weight - max(log_weight))
  weight <- weight / sum(weight)
  mean_beta <- colSums(means * weight)
  second_beta <- Reduce(`+`, lapply(1:nrow(states), function(h) {
    weight[h] * (covariances[, , h] + tcrossprod(means[h, ]))
  }))
  covariance_beta <- second_beta - tcrossprod(mean_beta)
  inclusion <- colSums(states * weight)
  stopifnot(abs(sum(weight) - 1) < 1e-14,
            min(eigen(covariance_beta, symmetric = TRUE)$values) > 0)
  results[[k]] <- data.frame(
    coefficient = k, random_variance = q[k], residual_variance = s[k],
    prior_pi_1 = pi[1], prior_pi_2 = pi[2], prior_tau_1 = tau[1], prior_tau_2 = tau[2],
    posterior_mean_1 = mean_beta[1], posterior_mean_2 = mean_beta[2],
    posterior_variance_1 = covariance_beta[1, 1],
    posterior_variance_2 = covariance_beta[2, 2],
    posterior_covariance = covariance_beta[1, 2],
    inclusion_1 = inclusion[1], inclusion_2 = inclusion[2],
    probability_00 = weight[1], probability_10 = weight[2],
    probability_01 = weight[3], probability_11 = weight[4]
  )
}
write.csv(data.frame(x, z, d), "tests/fixtures/wfmm-coefficient-inputs.csv", row.names = FALSE)
write.csv(do.call(rbind, results), "tests/fixtures/wfmm-coefficient-posterior.csv", row.names = FALSE)
cat("Wrote exact two-coefficient spike-and-slab mixture references.\n")
