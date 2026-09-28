# Independent one-coordinate GAO posterior integration, without Python outputs.
# 2017 Appendix-A binary margins, conditional-normal Gaussian rectangles and
# caller-defined Normal priors on a raw intercept or Fisher-z correlation.
options(warn = 2, digits = 17)

doses1 <- c(1, 3)
doses2 <- c(2, 5)
counts <- array(0L, c(2, 2, 2, 2))
partial <- array(0L, c(2, 2, 2))
for (i in 1:2) for (j in 1:2) {
  for (e in 1:2) for (t in 1:2) counts[i, j, e, t] <- (i + 2*j + e + t) %% 3
  partial[i, j, ] <- c(i %% 2, j %% 2)
}

margin <- function(d1, d2, intercept, slope) {
  eta <- intercept + slope * c(d1, d2)
  s <- exp(eta[1]) + exp(eta[2]) + .4 * exp(sum(eta))
  # Both link shapes are one in these reference cases.
  c(1 / (1 + s), s / (1 + s))
}

rectangles <- function(ep, tp, rho) {
  eb <- qnorm(c(0, ep[1], 1))
  tb <- qnorm(c(0, tp[1], 1))
  scale <- sqrt(1 - rho*rho)
  joint <- matrix(0, 2, 2)
  for (e in 1:2) for (t in 1:2) {
    integrand <- function(x) dnorm(x) * (
      pnorm((tb[t+1] - rho*x) / scale) - pnorm((tb[t] - rho*x) / scale))
    joint[e, t] <- integrate(integrand, eb[e], eb[e+1],
      abs.tol=1e-12, rel.tol=1e-10, subdivisions=1000L)$value
  }
  stopifnot(all(joint >= 0), max(abs(rowSums(joint)-ep)) < 2e-10,
            max(abs(colSums(joint)-tp)) < 2e-10)
  joint
}

evaluate <- function(theta, free) {
  intercept <- if (free == 'efficacy_intercept') c(theta, .3) else c(-1, .3)
  rho <- if (free == 'fisher_z') tanh(theta) else .55
  loglikelihood <- 0
  cells <- numeric(16)
  k <- 0L
  for (i in 1:2) for (j in 1:2) {
    ep <- margin(doses1[i], doses2[j], intercept, c(.2, -.1))
    tp <- margin(doses1[i], doses2[j], c(-.4, -1.2), c(-.05, .15))
    jp <- rectangles(ep, tp, rho)
    for (e in 1:2) for (t in 1:2) {
      k <- k + 1L
      cells[k] <- jp[e, t]
      if (counts[i, j, e, t] > 0) {
        loglikelihood <- loglikelihood + counts[i, j, e, t] * log(jp[e, t])
      }
    }
    loglikelihood <- loglikelihood + sum(partial[i, j, ] * log(tp))
  }
  list(loglikelihood=loglikelihood, values=c(theta, theta^2, rho, cells))
}

integrate_posterior <- function(order, free, mean, sd) {
  # Golub-Welsch quadrature for the standard Normal probability measure.
  jacobi <- matrix(0, order, order)
  for (k in 1:(order-1L)) jacobi[k, k+1L] <- jacobi[k+1L, k] <- sqrt(k)
  quadrature <- eigen(jacobi, symmetric=TRUE)
  weight <- quadrature$vectors[1, ]^2
  theta <- mean + sd * quadrature$values
  evaluated <- lapply(theta, evaluate, free=free)
  ll <- vapply(evaluated, function(x) x$loglikelihood, numeric(1))
  values <- vapply(evaluated, function(x) x$values, numeric(19))
  centered <- weight * exp(ll - max(ll))
  posterior <- drop(values %*% (centered / sum(centered)))
  c(mean=posterior[1], variance=posterior[2]-posterior[1]^2,
    mean_association=posterior[3], posterior[-(1:3)],
    log_normalizer=max(ll) + log(sum(centered)))
}

cell_names <- character(16)
k <- 0L
for (i in 0:1) for (j in 0:1) for (e in 0:1) for (t in 0:1) {
  k <- k + 1L
  cell_names[k] <- sprintf('joint.%d.%d.%d.%d', i, j, e, t)
}
rows <- list()
for (free in c('efficacy_intercept', 'fisher_z')) {
  mean <- if (free == 'efficacy_intercept') -1 else atanh(.3)
  sd <- if (free == 'efficacy_intercept') .5 else .2
  coarse <- integrate_posterior(64L, free, mean, sd)
  fine <- integrate_posterior(128L, free, mean, sd)
  difference <- max(abs(coarse - fine))
  stopifnot(all(is.finite(fine)), fine[2] > 0, difference < 2e-9)
  rows[[length(rows)+1L]] <- data.frame(case=free,
    metric=c('mean', 'variance', 'mean_association', cell_names, 'log_normalizer'),
    value=unname(fine), refinement_error=difference)
}
write.csv(do.call(rbind, rows), 'tests/fixtures/u2oet-gao-posterior.csv', row.names=FALSE)
cat('Generated two one-coordinate posteriors with 64/128-node refinement.\n')
