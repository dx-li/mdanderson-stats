#!/usr/bin/env Rscript
# Independent base-R references for the opt-in WFMM variance initializer.
# The mixed case uses balanced one-way random-effects REML closed forms.

y_resid <- c(1, 2, 3, 4)
beta_resid <- mean(y_resid)
df_resid <- length(y_resid) - 1L
rss <- sum((y_resid - beta_resid)^2)
variance_resid <- rss / df_resid
log_reml_resid <- -0.5 * (
  df_resid * log(2 * pi) + df_resid * log(variance_resid) + log(4) + df_resid
)
residual_only <- data.frame(
  kind="residual_only", component=1L, variance=variance_resid,
  beta=beta_resid, log_reml=log_reml_resid
)

y <- c(-1, -0.8, 0.2, 0.4, 1.1, 0.9, 2.4, 2.2)
groups <- factor(rep(1:4, each=2))
means <- as.numeric(tapply(y, groups, mean))
grand <- mean(y)
within_ss <- sum((y - ave(y, groups, FUN=mean))^2)
between_ss <- 2 * sum((means - grand)^2)
within_df <- length(y) - nlevels(groups)
between_df <- nlevels(groups) - 1L
ms_within <- within_ss / within_df
ms_between <- between_ss / between_df
random_variance <- (ms_between - ms_within) / 2
residual_variance <- ms_within

Z <- model.matrix(~ groups - 1)
X <- matrix(1, nrow=length(y), ncol=1)
V <- random_variance * Z %*% t(Z) + diag(residual_variance, length(y))
logdet_v <- as.numeric(determinant(V, logarithm=TRUE)$modulus)
inv_v <- solve(V)
information <- drop(t(X) %*% inv_v %*% X)
beta <- drop(solve(information, t(X) %*% inv_v %*% y))
residual <- y - drop(X %*% beta)
quadratic <- drop(t(residual) %*% inv_v %*% residual)
df <- length(y) - ncol(X)
log_reml <- -0.5 * (
  df * log(2 * pi) + logdet_v + log(information) + quadratic
)
mixed <- data.frame(
  kind="mixed_reml", component=1:2,
  variance=c(random_variance, residual_variance), beta=rep(beta, 2),
  log_reml=rep(log_reml, 2)
)
write.csv(rbind(residual_only, mixed), "tests/fixtures/wfmm-variance-init.csv", row.names=FALSE)
