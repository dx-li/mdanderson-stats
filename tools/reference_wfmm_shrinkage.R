# Independent GLS and direct likelihood optimization for WFMM shrinkage.
# This uses no EM recurrence and conditions on explicit variance estimates.
options(digits = 17)
x <- cbind(intercept = 1, covariate = c(0, .4, .8, 1.2, 1.6, 2))
z <- model.matrix(~ factor(c(1, 1, 2, 2, 3, 3)) - 1)
colnames(z) <- paste0("z", 1:ncol(z))
random_group <- c(0, 0, 1)
residual_group <- c(0, 0, 0, 1, 1, 1)
partition <- rep(0:1, each = 6)
q <- rbind(.08 + .01 * 1:12, .12 + .005 * 1:12)
s <- rbind(.2 + .02 * 1:12, .3 + .015 * 1:12)
target_z <- rbind(
  c(.1, -.2, .3, 0, 3.5, -4.5, -.1, .3, -.4, .2, 2.7, -3.1),
  c(.1, .2, -.2, .4, 4.5, -5.5, -.3, .1, .2, 0, 3, -4)
)
d <- matrix(NA_real_, nrow(x), length(partition))
reference <- list()
solve_chol <- function(factor, rhs) {
  backsolve(factor, forwardsolve(t(factor), rhs))
}
for (k in seq_along(partition)) {
  sigma <- z %*% diag(q[random_group + 1, k]) %*% t(z) +
    diag(s[residual_group + 1, k])
  factor <- chol(sigma)
  info <- crossprod(x, solve_chol(factor, x))
  conditional_variance <- 1 / diag(info)
  beta <- target_z[, k] * sqrt(conditional_variance)
  d[, k] <- x %*% beta
  beta_gls <- solve_chol(chol(info), crossprod(x, solve_chol(factor, d[, k])))
  joint_variance <- diag(chol2inv(chol(info)))
  stopifnot(max(abs(beta_gls - beta)) < 1e-13,
            min(joint_variance / conditional_variance) > 2)
  reference[[k]] <- data.frame(
    coefficient = k - 1, effect = 0:1, partition = partition[k],
    beta_gls = as.numeric(beta_gls), conditional_variance = conditional_variance,
    joint_variance = joint_variance, z = as.numeric(beta_gls) / sqrt(conditional_variance)
  )
}
log_sum <- function(a, b) {
  m <- pmax(a, b)
  m + log(exp(a - m) + exp(b - m))
}
log_likelihood <- function(parameters, observed_z) {
  pi <- plogis(parameters[1])
  u <- exp(parameters[2])
  sum(log_sum(log1p(-pi) + dnorm(observed_z, log = TRUE),
              log(pi) + dnorm(observed_z, sd = sqrt(1 + u), log = TRUE)))
}
group_reference <- list()
for (effect in 0:1) {
  for (group in 0:1) {
    observed_z <- target_z[effect + 1, partition == group]
    starts <- expand.grid(pi = c(.2, .5, .8), u = c(1, 10, 50))
    solutions <- lapply(seq_len(nrow(starts)), function(i) {
      optim(c(qlogis(starts$pi[i]), log(starts$u[i])),
            function(par) -log_likelihood(par, observed_z),
            method = "BFGS", hessian = TRUE,
            control = list(reltol = 1e-13, maxit = 1000))
    })
    best <- solutions[[which.min(vapply(solutions, function(fit) fit$value, 0.0))]]
    pi <- plogis(best$par[1])
    u <- exp(best$par[2])
    null_ll <- sum(dnorm(observed_z, log = TRUE))
    full_u <- max(0, mean(observed_z^2) - 1)
    full_ll <- sum(dnorm(observed_z, sd = sqrt(1 + full_u), log = TRUE))
    stopifnot(best$convergence == 0, pi > .01, pi < .99, u > 0,
              min(eigen(best$hessian, symmetric = TRUE)$values) > 0,
              -best$value > max(null_ll, full_ll) + .1)
    group_reference[[length(group_reference) + 1]] <- data.frame(
      effect = effect, partition = group, inclusion_probability = pi,
      slab_to_sampling_variance = u, log_marginal_likelihood = -best$value,
      null_log_marginal_likelihood = null_ll, full_log_marginal_likelihood = full_ll
    )
  }
}
colnames(d) <- paste0("d", seq_len(ncol(d)))
write.csv(data.frame(x, z, residual_group, d),
          "tests/fixtures/wfmm-shrinkage-inputs.csv", row.names = FALSE)
write.csv(data.frame(coefficient = 0:11, partition, q1 = q[1, ], q2 = q[2, ],
                     s1 = s[1, ], s2 = s[2, ]),
          "tests/fixtures/wfmm-shrinkage-variances.csv", row.names = FALSE)
write.csv(do.call(rbind, reference),
          "tests/fixtures/wfmm-shrinkage-coefficients.csv", row.names = FALSE)
write.csv(do.call(rbind, group_reference),
          "tests/fixtures/wfmm-shrinkage-groups.csv", row.names = FALSE)
cat("Wrote GLS and four independently optimized interior mixture references.\n")
