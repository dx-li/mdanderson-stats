# Independent restricted-Gaussian and uniform-correlation posterior quadrature.
# No Python outputs enter these references. The normal base prior is multiplied
# by ONE joint validity indicator; it is not renormalized at each alpha value.
options(warn = 2, digits = 17)

quadrature <- function(n, normal = FALSE) {
  j <- matrix(0, n, n)
  k <- seq_len(n - 1L)
  off <- if (normal) sqrt(k) else k / sqrt(4*k*k - 1)
  j[cbind(k, k+1L)] <- j[cbind(k+1L, k)] <- off
  eig <- eigen(j, symmetric = TRUE)
  list(x = eig$values, w = eig$vectors[1, ]^2 * if (normal) 1 else 2)
}

restricted <- function(order, posterior) {
  alpha_rule <- quadrature(order, TRUE)
  gamma_rule <- quadrature(order)
  alpha <- .2 + .6 * alpha_rule$x
  values <- weights <- list()
  k <- 0L
  for (a in seq_along(alpha)) {
    # Dose grids {0,2}: centered predictors alpha + .3*{-1,1} and .4+.2*{-1,1}.
    lower <- -(exp(-alpha[a]-.3) + exp(-.6))
    lo <- max(-9, (lower + 1) / .65)
    hi <- 9
    if (lo >= hi) next
    z <- (hi + lo)/2 + (hi-lo)/2 * gamma_rule$x
    gamma <- -1 + .65*z
    weight <- alpha_rule$w[a] * gamma_rule$w * (hi-lo)/2 * dnorm(z)
    probability <- matrix(0, length(gamma), 4)
    column <- 0L
    for (d1 in c(-1,1)) for (d2 in c(-1,1)) {
      column <- column + 1L
      e1 <- exp(alpha[a]+.3*d1)
      e2 <- exp(.4+.2*d2)
      bracket <- e1 + e2 + gamma*e1*e2
      stopifnot(all(bracket > 0))
      probability[,column] <- bracket/(1+bracket)
    }
    # Efficacy totals at dose pairs in row-major order; toxicity is independent.
    if (posterior) {
      successes <- c(1,3,2,4)
      failures <- c(3,1,2,1)
      loglik <- drop(log(probability) %*% successes + log1p(-probability) %*% failures)
      weight <- weight * exp(loglik)
    }
    k <- k + 1L
    values[[k]] <- cbind(alpha=alpha[a], gamma=gamma,
                         alpha2=alpha[a]^2, gamma2=gamma^2,
                         alpha_gamma=alpha[a]*gamma, probability)
    weights[[k]] <- weight
  }
  v <- do.call(rbind, values)
  w <- unlist(weights)
  normalizer <- sum(w)
  moments <- drop(crossprod(w/normalizer, v))
  c(alpha_mean=moments[1], gamma_mean=moments[2],
    alpha_variance=moments[3]-moments[1]^2,
    gamma_variance=moments[4]-moments[2]^2,
    alpha_gamma_covariance=moments[5]-moments[1]*moments[2],
    setNames(moments[6:9],paste0('efficacy.',0:3)), normalizer=normalizer)
}

correlation <- function(order) {
  rule <- quadrature(order)
  angle <- pi/2 * rule$x
  rho <- sin(angle)
  same <- .25 + angle/(2*pi)
  different <- .25 - angle/(2*pi)
  # Each of four dose pairs has complete counts [[2,1],[1,3]].
  weight <- rule$w * pi/2 * cos(angle)/2 * same^20 * different^8
  normalizer <- sum(weight)
  weight <- weight/normalizer
  mean <- sum(weight*rho)
  c(association_mean=mean, association_variance=sum(weight*rho^2)-mean^2,
    joint_same=sum(weight*same), joint_different=sum(weight*different),
    normalizer=normalizer)
}

rows <- list()
for (case in c('restricted_prior','restricted_posterior','uniform_rho_posterior')) {
  calculate <- if (case=='uniform_rho_posterior') correlation else
    function(n) restricted(n,case=='restricted_posterior')
  coarse <- calculate(64L)
  fine <- calculate(128L)
  error <- max(abs(fine-coarse))
  stopifnot(all(is.finite(fine)), error < 2e-9)
  # Strip R's nested names inherited from vector indexing.
  metric <- if (case=='uniform_rho_posterior')
    c('association_mean','association_variance','joint_same','joint_different','normalizer') else
    c('alpha_mean','gamma_mean','alpha_variance','gamma_variance',
      'alpha_gamma_covariance',paste0('efficacy.',0:3),'normalizer')
  rows[[case]] <- data.frame(case=case,metric=metric,value=unname(fine),refinement_error=error)
}
write.csv(do.call(rbind,rows),'tests/fixtures/u2oet-gao2010-posterior.csv',row.names=FALSE)
cat('Generated restricted joint-prior/posterior and uniform-rho references.\n')
