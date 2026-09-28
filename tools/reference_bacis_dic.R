# Independent base-R quadrature for the JAGS/Plummer classification-model DIC.
# Run from the repository root. No JAGS session or package installation is needed.
options(digits = 17)
softplus <- function(x) pmax(x, 0) + log1p(exp(-abs(x)))

component <- function(y, n, location, precision) {
  score <- function(eta) y - n * plogis(eta) - precision * (eta - location)
  mode <- uniroot(score, c(location + (y-n)/precision, location + y/precision),
                  tol=1e-12)$root
  kernel <- function(eta) {
    -y*softplus(-eta) - (n-y)*softplus(eta) - precision*(eta-location)^2/2
  }
  peak <- kernel(mode)
  scale <- 1/sqrt(precision + n*plogis(mode)*plogis(-mode))
  weight <- function(z) exp(kernel(mode + scale*z) - peak)
  integral <- function(f) integrate(function(z) f(z)*weight(z), -Inf, Inf,
                                    rel.tol=2e-11, abs.tol=1e-12,
                                    subdivisions=400)$value
  norm <- integral(function(z) rep(1, length(z)))
  expected <- function(f) integral(f)/norm
  centered_eta <- expected(function(z) scale*z)
  mean_p <- expected(function(z) plogis(mode + scale*z))
  covariance <- expected(function(z) (scale*z-centered_eta)*
                           (plogis(mode+scale*z)-mean_p))
  list(log_evidence=peak+log(scale)+log(precision/(2*pi))/2+log(norm),
       eta=mode+centered_eta, p=mean_p, covariance=covariance,
       log_p=expected(function(z) -softplus(-(mode+scale*z))),
       log_failure=expected(function(z) -softplus(mode+scale*z)))
}

inputs <- read.csv('tests/fixtures/bacis-classification.csv')
inputs <- inputs[, c('case','group','y','n','phi_low','phi_high','precision')]
inputs <- rbind(inputs, data.frame(case='separated_mixture', group=1, y=5, n=10,
                                  phi_low=.01, phi_high=.99, precision=100))
rows <- lapply(seq_len(nrow(inputs)), function(i) {
  x <- inputs[i,]
  lo <- component(x$y, x$n, qlogis(x$phi_low), x$precision)
  hi <- component(x$y, x$n, qlogis(x$phi_high), x$precision)
  high_weight <- plogis(hi$log_evidence-lo$log_evidence)
  low_weight <- plogis(lo$log_evidence-hi$log_evidence)
  average <- function(name) low_weight*lo[[name]] + high_weight*hi[[name]]
  covariance <- average('covariance') + low_weight*high_weight*
    (hi$p-lo$p)*(hi$eta-lo$eta)
  mean_deviance <- -2*(lchoose(x$n,x$y) + x$y*average('log_p') +
                         (x$n-x$y)*average('log_failure'))
  penalty <- x$n*covariance
  stopifnot(is.finite(mean_deviance), is.finite(penalty), penalty >= 0)
  cbind(x, posterior_mean=average('p'), posterior_logit_mean=average('eta'),
        mean_deviance=mean_deviance, penalty=penalty, dic=mean_deviance+penalty)
})
write.csv(do.call(rbind, rows), 'tests/fixtures/bacis-dic-reference.csv', row.names=FALSE)

# Verify the covariance reduction against explicit directed binomial KL sums
# on an independent, finite posterior distribution (not a quadrature result).
p <- c(.02, .3, .85)
w <- c(.2, .5, .3)
n <- 25
direct <- 0
for(i in seq_along(p)) for(j in seq_along(p)) {
  direct <- direct + w[i]*w[j]*n*(p[i]*log(p[i]/p[j]) +
                                  (1-p[i])*log((1-p[i])/(1-p[j])))
}
reduced <- n*(sum(w*p*qlogis(p))-sum(w*p)*sum(w*qlogis(p)))
stopifnot(abs(direct-reduced) < 1e-13)
cat('Verified', length(rows), 'quadrature rows and the pairwise KL identity.\n')
