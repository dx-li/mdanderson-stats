# Independent base-R tensor Gauss-Legendre reference for explicit bivariate
# Normal priors on (log shape, log scale). Distribution equations follow the
# cached BCSTTE User's Guide, §§4.2-4.4; priors and right-censor likelihood are
# explicit independent reference conventions, not recovered native defaults.
options(warn=2, digits=17)

legendre <- function(n, lower=-8, upper=8) {
  i <- seq_len(n-1L)
  off <- i / sqrt(4*i*i-1)
  jacobi <- matrix(0, n, n)
  jacobi[cbind(seq_len(n-1L), 2:n)] <- off
  jacobi[cbind(2:n, seq_len(n-1L))] <- off
  eig <- eigen(jacobi, symmetric=TRUE)
  ord <- order(eig$values)
  list(x=(lower+upper)/2+(upper-lower)/2*eig$values[ord],
       w=(upper-lower)*eig$vectors[1L,ord]^2)
}
softplus <- function(x) pmax(x,0)+log1p(exp(-abs(x)))

log_density <- function(family, t, log_shape, log_scale) {
  shape <- exp(log_shape); scale <- exp(log_scale); lt <- log(t)
  if (family == 'gamma') {
    return((shape-1)*lt - t/scale - lgamma(shape) - shape*log_scale)
  }
  if (family == 'inverse_gamma') {
    return(shape*log_scale - lgamma(shape) - (shape+1)*lt - scale/t)
  }
  if (family == 'loglogistic') {
    z <- shape*(lt-log_scale)
    return(log_shape-log_scale+(shape-1)*(lt-log_scale)-2*softplus(z))
  }
  stop('unknown family')
}
log_survival <- function(family, t, log_shape, log_scale) {
  shape <- exp(log_shape); scale <- exp(log_scale); lt <- log(t)
  if (family == 'gamma') return(pgamma(t, shape=shape, scale=scale, lower.tail=FALSE, log.p=TRUE))
  if (family == 'inverse_gamma') return(pgamma(scale/t, shape=shape, lower.tail=TRUE, log.p=TRUE))
  if (family == 'loglogistic') return(-softplus(shape*(lt-log_scale)))
  stop('unknown family')
}
cdf_value <- function(family, t, log_shape, log_scale) {
  shape <- exp(log_shape); scale <- exp(log_scale); lt <- log(t)
  if (family == 'gamma') return(pgamma(t, shape=shape, scale=scale))
  if (family == 'inverse_gamma') return(pgamma(scale/t, shape=shape, lower.tail=FALSE))
  if (family == 'loglogistic') return(plogis(shape*(lt-log_scale)))
  stop('unknown family')
}

summarize_case <- function(family, times, event, mean, covariance, nquad, half_width=8) {
  q <- legendre(nquad, -half_width, half_width)
  mesh <- expand.grid(z1=q$x, z2=q$x)
  prior_nodes <- cbind(mesh$z1, mesh$z2)
  chol <- t(chol(covariance))
  pars <- sweep(prior_nodes %*% t(chol), 2L, mean, '+')
  lp <- dnorm(mesh$z1, log=TRUE)+dnorm(mesh$z2, log=TRUE)+
    log(q$w[match(mesh$z1,q$x)])+log(q$w[match(mesh$z2,q$x)])
  ll <- vapply(seq_len(nrow(pars)), function(j) {
    ld <- log_density(family, times, pars[j,1], pars[j,2])
    ls <- log_survival(family, times, pars[j,1], pars[j,2])
    sum(ifelse(event==1, ld, ls))
  }, numeric(1))
  logw <- lp+ll
  w <- exp(logw-max(logw)); w <- w/sum(w)
  cdfmat <- vapply(times, function(t) vapply(seq_len(nrow(pars)), function(j)
    cdf_value(family,t,pars[j,1],pars[j,2]), numeric(1)), numeric(nrow(pars)))
  c(mean_log_shape=sum(w*pars[,1]), mean_log_scale=sum(w*pars[,2]),
    mean_shape=sum(w*exp(pars[,1])), mean_scale=sum(w*exp(pars[,2])),
    var_log_shape=sum(w*(pars[,1]-sum(w*pars[,1]))^2),
    var_log_scale=sum(w*(pars[,2]-sum(w*pars[,2]))^2),
    covariance_log_shape_scale=sum(w*(pars[,1]-sum(w*pars[,1]))*(pars[,2]-sum(w*pars[,2]))),
    setNames(colSums(cdfmat*w), paste0('posterior_mean_cdf_',seq_along(times))))
}

# Complete and right-censored examples for every family, same proper prior
# correlation but family-specific moderate centers/scales.
times <- c(.55, .9, 1.4, 2.2, 3.1)
cases <- list(
 gamma_complete=list(family='gamma', event=rep(1,5), mean=log(c(2.2,1.1))),
 gamma_censored=list(family='gamma', event=c(1,0,1,1,0), mean=log(c(2.2,1.1))),
 inverse_gamma_complete=list(family='inverse_gamma', event=rep(1,5), mean=log(c(3.2,2.1))),
 inverse_gamma_censored=list(family='inverse_gamma', event=c(1,0,1,1,0), mean=log(c(3.2,2.1))),
 loglogistic_complete=list(family='loglogistic', event=rep(1,5), mean=log(c(2.0,1.3))),
 loglogistic_censored=list(family='loglogistic', event=c(1,0,1,1,0), mean=log(c(2.0,1.3)))
)
covariance <- matrix(c(.16,.055,.055,.25),2,2)
out <- list()
for (name in names(cases)) for (setting in list(c(49L,8),c(65L,8),c(81L,8),c(91L,9))) {
  x <- cases[[name]]
  nquad <- as.integer(setting[1]); half_width <- setting[2]
  result <- summarize_case(x$family,times,x$event,x$mean,covariance,nquad,half_width)
  out[[length(out)+1L]] <- data.frame(case=name,nquad=nquad,half_width=half_width,
    metric=names(result),value=unname(result))
}
reference <- do.call(rbind,out)
reference$value <- sprintf('%.17g', reference$value)
args <- commandArgs(trailingOnly=TRUE)
outdir <- if (length(args)) args[[1]] else getwd()
if (!dir.exists(outdir)) stop('output directory does not exist')
write.csv(reference, file.path(outdir,'reference.csv'), row.names=FALSE)
# Fixed-parameter extreme right-censor tails. For shape 2, Gamma survival is
# exp(-x)*(1+x); inverse-Gamma survival at beta/t=x is P(Gamma(2)<=x), which
# is asymptotic to x^2/2 and must remain available in log form at x=1e-200.
gamma_log_tail <- log_survival('gamma',1000,log(2),0)
small_ratio <- 1e-200
ig_log_tail <- log_survival('inverse_gamma',1,log(2),log(small_ratio))
tails <- data.frame(case=c('gamma_shape2_x1000','inverse_gamma_shape2_beta_over_t_1e-200'),
  log_survival=c(gamma_log_tail,ig_log_tail),
  analytic=c(-1000+log(1001),2*log(small_ratio)-log(2)))
tails$absolute_delta <- abs(tails$log_survival-tails$analytic)
for (column in names(tails)[-1]) tails[[column]] <- sprintf('%.17g',tails[[column]])
write.csv(tails,file.path(outdir,'tail-reference.csv'),row.names=FALSE)
cat('Generated posterior reference grid and fixed-parameter tail checks.\n')
