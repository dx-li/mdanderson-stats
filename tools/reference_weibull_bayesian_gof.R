# Independent BCSTTE fixed-shape Weibull/Gamma-posterior references.
# Johnson posterior chi-square expectations are integrated exactly over the
# finite intervals where all observed-data bin memberships are constant.
options(warn=2, digits=17)

log_sum_exp <- function(x) {
  center <- max(x)
  center + log(sum(exp(x-center)))
}
cases <- list(
  exponential=list(times=c(1,2,4,8), beta=1, a=2, b=3, bins=4),
  decreasing_hazard=list(times=c(.03,.2,1,2,5,13), beta=.7, a=1.5, b=.4, bins=3),
  increasing_hazard=list(times=c(.2,.5,1,1.5,2.8), beta=3, a=2, b=.1, bins=4),
  base_units=list(times=c(.5,1,2,4,8), beta=2.3, a=0, b=0, bins=3),
  small_units=list(times=c(.5,1,2,4,8)*1e-200, beta=2.3, a=0, b=0, bins=3),
  large_units=list(times=c(.5,1,2,4,8)*1e200, beta=2.3, a=0, b=0, bins=3),
  huge_shape=list(times=rep(1e100,4), beta=1e305, a=0, b=0, bins=3)
)
inputs <- values <- list()
for (name in names(cases)) {
  x <- cases[[name]]
  n <- length(x$times)
  shape <- x$a+n
  # Dimensionless powers preserve the exact identical-time case even when
  # beta*log(time) dwarfs log(n). No absolute power or rate is formed.
  log_scale <- x$beta*log(max(x$times))
  relative_power <- (x$times/max(x$times))^x$beta
  log_factor <- log_sum_exp(c(log(relative_power), if (x$b>0) log(x$b)-log_scale else -Inf))
  log_rate <- log_scale+log_factor
  scaled_power <- exp(log(relative_power)-log_factor)
  edges <- (1:(x$bins-1))/x$bins
  # Z = lambda * posterior_rate is Gamma(shape, rate=1).
  breakpoints <- sort(unique(c(0, outer(-log1p(-edges), 1/scaled_power), Inf)))
  mass <- statistic <- numeric(length(breakpoints)-1L)
  for (i in seq_along(mass)) {
    lower <- breakpoints[i]
    upper <- breakpoints[i+1L]
    middle <- if (is.finite(upper)) lower+(upper-lower)/2 else lower+max(1,lower)
    cdf <- -expm1(-middle*scaled_power)
    bin <- vapply(cdf, function(p) 1L+sum(p>edges), integer(1))
    count <- tabulate(bin, nbins=x$bins)
    statistic[i] <- sum((count-n/x$bins)^2/(n/x$bins))
    mass[i] <- if (lower>shape) {
      pgamma(lower,shape,lower.tail=FALSE)-pgamma(upper,shape,lower.tail=FALSE)
    } else {
      pgamma(upper,shape)-pgamma(lower,shape)
    }
  }
  stopifnot(all(mass>=0), abs(sum(mass)-1)<2e-14)
  expected_cdf <- -expm1(-shape*log1p(scaled_power))
  summary <- c(posterior_shape=shape, log_posterior_rate=log_rate,
    log_rate_scale=log_scale, log_rate_factor=log_factor,
    mean_centered_log_rate=digamma(shape)-log_factor,
    mean_log_rate=digamma(shape)-log_rate, var_log_rate=trigamma(shape),
    mean_statistic=sum(mass*statistic),
    area_against_reference=sum(mass*pchisq(statistic,x$bins-1)),
    mean_reference_tail=sum(mass*pchisq(statistic,x$bins-1,lower.tail=FALSE)),
    critical_exceedance_fraction=sum(mass*(statistic>qchisq(.95,x$bins-1))),
    setNames(expected_cdf,paste0('mean_cdf.',seq_len(n)-1L)))
  values[[name]] <- data.frame(case=name,metric=names(summary),value=unname(summary))
  inputs[[name]] <- data.frame(case=name,index=seq_len(n)-1L,time=x$times,
    weibull_shape=x$beta,prior_shape=x$a,prior_rate=x$b,bins=x$bins)
}
write_exact <- function(rows,path) {
  data <- do.call(rbind,rows)
  for (name in names(data)) if (is.numeric(data[[name]])) data[[name]] <- sprintf('%.17g',data[[name]])
  write.csv(data,path,row.names=FALSE)
}
write_exact(inputs,'tests/fixtures/weibull-bayesian-inputs.csv')
write_exact(values,'tests/fixtures/weibull-bayesian-reference.csv')
cat('Generated seven exact-posterior and integrated diagnostic cases.\n')
