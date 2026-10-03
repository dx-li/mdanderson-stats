# Independent base-R direct observed-likelihood quadrature for a censored
# log-normal posterior under the explicit Python Normal-Inverse-Gamma prior.
# This reference does not use latent-variable augmentation or claim a native
# BCSTTE prior/fitter.
options(warn=2, digits=17)

legendre <- function(n, lower, upper) {
  index <- seq_len(n-1L)
  off <- index / sqrt(4*index*index-1)
  jacobi <- matrix(0,n,n)
  jacobi[cbind(seq_len(n-1L),2:n)] <- off
  jacobi[cbind(2:n,seq_len(n-1L))] <- off
  eig <- eigen(jacobi,symmetric=TRUE)
  ord <- order(eig$values)
  nodes <- (lower+upper)/2+(upper-lower)/2*eig$values[ord]
  weights <- (upper-lower)*eig$vectors[1L,ord]^2
  stopifnot(abs(sum(weights)-(upper-lower)) < 1e-12*max(1,abs(upper-lower)))
  list(x=nodes,w=weights)
}

prior <- list(m=0.30,k=2.0,a=4.5,b=1.1)
cases <- list(
  mixed=list(t=c(.4,1.2,1.2,2.6,5.0,0,3.5),
             event=c(TRUE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE),
             reference=.4),
  all_censored=list(t=c(.7,1.5,2.0,4.0,4.0),
                    event=rep(FALSE,5),reference=.7)
)
resolutions <- list(
  grid81=list(n=81L,q=c(-7,7),z=c(-11,7)),
  grid121=list(n=121L,q=c(-7,7),z=c(-11,7)),
  grid151=list(n=151L,q=c(-7,7),z=c(-11,7)),
  wide151=list(n=151L,q=c(-8,8),z=c(-13,9))
)

posterior_reference <- function(case, resolution) {
  q_rule <- legendre(resolution$n,resolution$q[1],resolution$q[2])
  z_rule <- legendre(resolution$n,resolution$z[1],resolution$z[2])
  grid <- expand.grid(q=q_rule$x,z=z_rule$x)
  q <- grid$q
  z <- grid$z
  variance <- exp(z)
  offset <- log(case$reference)
  centered_prior_location <- prior$m-offset
  # q is standard Normal under mu|V; this variable transform absorbs the
  # conditional-location Jacobian. z=log(V) includes the inverse-gamma Jacobian.
  mu <- centered_prior_location+q*sqrt(variance/prior$k)
  log_prior <- dnorm(q,log=TRUE) + prior$a*log(prior$b)-lgamma(prior$a) -
    prior$a*z-prior$b*exp(-z)
  log_likelihood <- numeric(length(q))
  for (i in seq_along(case$t)) {
    time <- case$t[i]
    if (case$event[i]) {
      y <- log(time)-offset
      log_likelihood <- log_likelihood + dnorm(y,mean=mu,sd=sqrt(variance),log=TRUE)-log(time)
    } else if (time > 0) {
      censor_y <- log(time)-offset
      log_likelihood <- log_likelihood +
        pnorm(mu-censor_y,mean=0,sd=sqrt(variance),log.p=TRUE)
    }
    # A zero-time right censor has survival one and adds exactly zero.
  }
  log_weights <- log_prior + log(q_rule$w[match(q,q_rule$x)]) +
    log(z_rule$w[match(z,z_rule$x)]) + log_likelihood
  weights <- exp(log_weights-max(log_weights))
  weights <- weights/sum(weights)
  mean_mu <- sum(weights*mu)
  mean_z <- sum(weights*z)
  centered_mu <- mu-mean_mu
  centered_z <- z-mean_z
  result <- c(mean_location_centered=mean_mu,
              mean_location_absolute=mean_mu+offset,
              mean_log_variance=mean_z,
              var_location_centered=sum(weights*centered_mu^2),
              var_log_variance=sum(weights*centered_z^2),
              cov_location_log_variance=sum(weights*centered_mu*centered_z))
  positive <- which(case$t > 0)
  for (i in positive) {
    y <- log(case$t[i])-offset
    result[paste0('posterior_mean_cdf_',i)] <-
      sum(weights*pnorm(y,mean=mu,sd=sqrt(variance)))
  }
  result
}

rows <- list()
for (case_name in names(cases)) for (resolution_name in names(resolutions)) {
  value <- posterior_reference(cases[[case_name]],resolutions[[resolution_name]])
  rows[[length(rows)+1L]] <- data.frame(
    case=case_name,resolution=resolution_name,metric=names(value),value=unname(value)
  )
}
reference <- do.call(rbind,rows)
reference$value <- sprintf('%.17g',reference$value)
args <- commandArgs(trailingOnly=TRUE)
outdir <- if (length(args)) args[[1]] else 'tests/fixtures'
if (!dir.exists(outdir)) stop('output directory does not exist')
write.csv(reference,file.path(outdir,'lognormal-censored-reference.csv'),row.names=FALSE)
cat('Generated direct log-normal censored-posterior quadrature references.\n')
