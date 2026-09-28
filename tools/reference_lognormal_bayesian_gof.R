# Independent N-InvGamma posterior, Student-t predictive CDF, and integrated
# Johnson diagnostics. Integrate out location analytically at each precision,
# then integrate the remaining scalar Gamma variable with base R quadrature.
options(warn=2, digits=17)
cases <- list(
  moderate=list(t=c(.5,1,2,4,8),m=.2,k=1.5,a=2,b=.8,bins=3),
  concentrated_prior=list(t=c(.7,1,1.5,2),m=1.2,k=12,a=5,b=.5,bins=4),
  broad_times=list(t=c(.01,.2,1,10,100),m=-.5,k=.4,a=1.5,b=2,bins=3),
  equal_times=list(t=rep(3,4),m=log(3),k=2,a=2,b=.3,bins=3),
  small_units=list(t=c(.5,1,2,4,8)*1e-200,m=.2+log(1e-200),k=1.5,a=2,b=.8,bins=3),
  large_units=list(t=c(.5,1,2,4,8)*1e200,m=.2+log(1e200),k=1.5,a=2,b=.8,bins=3)
)
inputs <- output <- list()
for (name in names(cases)) {
  x <- cases[[name]]
  y <- log(x$t)
  n <- length(y)
  k <- x$k+n
  m <- (x$k*x$m+sum(y))/k
  a <- x$a+n/2
  b <- x$b+sum((y-mean(y))^2)/2+x$k*n/(2*k)*(mean(y)-x$m)^2
  edge_z <- qnorm((1:(x$bins-1))/x$bins)
  centers <- y-m
  # Conditional on Z=b/sigma^2, standardized location is N(0,1).
  # Bin memberships change only at these finitely many location cuts.
  conditional <- function(z) {
    cuts <- sort(unique(as.vector(sqrt(k)*outer(centers*sqrt(z/b),edge_z,'-'))))
    lower <- c(-Inf,cuts)
    upper <- c(cuts,Inf)
    middle <- (lower+upper)/2
    middle[1] <- upper[1]-max(1,abs(upper[1]))
    middle[length(middle)] <- tail(lower,1)+max(1,abs(tail(lower,1)))
    mass <- ifelse(lower>0,pnorm(lower,lower.tail=FALSE)-pnorm(upper,lower.tail=FALSE),
                   pnorm(upper)-pnorm(lower))
    statistic <- vapply(middle,function(mu_z) {
      p <- pnorm(centers*sqrt(z/b)-mu_z/sqrt(k))
      bin <- vapply(p,function(q) 1L+sum(q>(1:(x$bins-1))/x$bins),integer(1))
      count <- tabulate(bin,nbins=x$bins)
      sum((count-n/x$bins)^2/(n/x$bins))
    },numeric(1))
    c(mean_statistic=sum(mass*statistic),
      mean_reference_tail=sum(mass*pchisq(statistic,x$bins-1,lower.tail=FALSE)),
      critical_exceedance_fraction=sum(mass*(statistic>qchisq(.95,x$bins-1))))
  }
  # Split at every positive precision where two location cuts exchange order.
  ratios <- as.vector(outer(as.vector(outer(edge_z,edge_z,'-')),
                            as.vector(outer(centers,centers,'-')),'/'))
  split <- sort(unique(c(0,b*ratios[is.finite(ratios)&ratios>0]^2,Inf)))
  integrated <- vapply(seq_len(3),function(metric) {
    sum(vapply(seq_len(length(split)-1L),function(i) {
      integrate(function(z) vapply(z,function(u) conditional(u)[metric]*dgamma(u,a),numeric(1)),
        split[i],split[i+1L],abs.tol=2e-10,rel.tol=2e-9,subdivisions=200L)$value
    },numeric(1)))
  },numeric(1))
  names(integrated) <- names(conditional(a))
  predictive_scale <- sqrt(b*(k+1)/(a*k))
  summary <- c(posterior_location=m,posterior_location_precision=k,
    posterior_variance_shape=a,posterior_log_variance_scale=log(b),
    mean_centered_location=m-log(x$t[1]),var_location=b/((a-1)*k),
    mean_log_variance=log(b)-digamma(a),var_log_variance=trigamma(a),
    integrated,setNames(pt((y-m)/predictive_scale,df=2*a),paste0('mean_cdf.',seq_len(n)-1L)))
  inputs[[name]] <- data.frame(case=name,index=seq_len(n)-1L,time=x$t,
    prior_location=x$m,prior_location_precision=x$k,prior_variance_shape=x$a,
    prior_variance_scale=x$b,bins=x$bins)
  output[[name]] <- data.frame(case=name,metric=names(summary),value=unname(summary))
}
write_exact <- function(rows,path) {
  data <- do.call(rbind,rows)
  for (name in names(data)) if (is.numeric(data[[name]])) data[[name]] <- sprintf('%.17g',data[[name]])
  write.csv(data,path,row.names=FALSE)
}
write_exact(inputs,'tests/fixtures/lognormal-bayesian-inputs.csv')
write_exact(output,'tests/fixtures/lognormal-bayesian-reference.csv')
cat('Generated six conjugate-posterior and integrated Johnson diagnostic cases.\n')
