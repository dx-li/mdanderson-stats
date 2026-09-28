# Independent BARD paper BF-BLRM references (published RAW dose-ratio model).
# Base R integration over independent normal log-alpha/log-beta coordinates.
# These are explicit example priors, not defaults of the BARD application.
options(warn=2, digits=17)

probability <- function(dose,reference,log_alpha,log_beta) {
  plogis(log_alpha+exp(log_beta)*(dose/reference))
}

curves <- list()
for(a in c(-4,-1.1)) for(b in c(-1,0)) for(d in c(10,20,50,100,200)) {
  curves[[length(curves)+1L]] <- data.frame(dose=d,reference=50,
    log_alpha=a,log_beta=b,probability=probability(d,50,a,b))
}
write.csv(do.call(rbind,curves),'tests/fixtures/bard-blrm-probabilities.csv',row.names=FALSE)

cases <- list(
  fixed_slope=list(d=c(1,2,4),reference=2,n=c(3,6,6),y=c(0,2,3),
    mu=c(-1.3,log(.4)),sd=c(.7,0),target=c(.16,.33)),
  two_parameter=list(d=c(.5,1,2),reference=1,n=c(4,6,4),y=c(0,2,3),
    mu=c(-1.5,-1.2),sd=c(.8,.5),target=c(.16,.33)))

# Normal coordinates outside +/-10 have prior probability below 3.1e-23.
# Likelihood is an unnormalized product of Bernoulli probabilities, <=1;
# dividing this tail bound by the evidence bounds omitted posterior mass.
limit <- 10
rows <- list(); inputs <- list()
for(name in names(cases)) {
  x <- cases[[name]]
  likelihood <- function(a,b) {
    vapply(a,function(ai) {
      p <- probability(x$d,x$reference,ai,b)
      exp(sum(dbinom(x$y,x$n,p,log=TRUE)-lchoose(x$n,x$y)))
    },numeric(1))
  }
  inner <- function(zb,metric,j=1L) {
    b <- x$mu[2]+x$sd[2]*zb
    lo <- -limit; hi <- limit
    if(metric %in% c('target','overdose')) {
      lower_a <- qlogis(x$target[1])-exp(b)*(x$d[j]/x$reference)
      upper_a <- qlogis(x$target[2])-exp(b)*(x$d[j]/x$reference)
      if(metric=='target') {
        lo <- max(lo,(lower_a-x$mu[1])/x$sd[1])
        hi <- min(hi,(upper_a-x$mu[1])/x$sd[1])
      } else lo <- max(lo,(upper_a-x$mu[1])/x$sd[1])
    }
    if(lo>=hi) return(0)
    integrate(function(za) {
      a <- x$mu[1]+x$sd[1]*za
      value <- switch(metric,evidence=1,log_alpha=a,log_beta=b,
        probability=probability(x$d[j],x$reference,a,b),target=1,overdose=1)
      dnorm(za)*likelihood(a,b)*value
    },lo,hi,abs.tol=1e-13,rel.tol=2e-10,subdivisions=300L)$value
  }
  integrate_metric <- function(metric,j=1L) {
    if(x$sd[2]==0) return(inner(0,metric,j))
    integrate(function(zb) vapply(zb,function(z) inner(z,metric,j),numeric(1))*dnorm(zb),
      -limit,limit,abs.tol=1e-13,rel.tol=2e-9,subdivisions=300L)$value
  }
  evidence <- integrate_metric('evidence')
  omitted <- 2*sum(x$sd>0)*pnorm(limit,lower.tail=FALSE)/evidence
  for(metric in c('log_alpha','log_beta')) rows[[length(rows)+1L]] <- data.frame(
    scenario=name,quantity=metric,dose_index=-1L,value=integrate_metric(metric)/evidence,
    evidence=evidence,omitted_posterior_mass_bound=omitted)
  for(j in seq_along(x$d)) {
    inputs[[length(inputs)+1L]] <- data.frame(scenario=name,dose_index=j-1L,
      dose=x$d[j],reference=x$reference,patients=x$n[j],toxicities=x$y[j],
      mean_log_alpha=x$mu[1],mean_log_beta=x$mu[2],
      sd_log_alpha=x$sd[1],sd_log_beta=x$sd[2],lower=x$target[1],upper=x$target[2])
    for(metric in c('probability','target','overdose')) rows[[length(rows)+1L]] <- data.frame(
      scenario=name,quantity=metric,dose_index=j-1L,value=integrate_metric(metric,j)/evidence,
      evidence=evidence,omitted_posterior_mass_bound=omitted)
  }
}
write.csv(do.call(rbind,inputs),'tests/fixtures/bard-blrm-inputs.csv',row.names=FALSE)
write.csv(do.call(rbind,rows),'tests/fixtures/bard-blrm-posterior.csv',row.names=FALSE)
cat('Generated',length(curves),'raw-ratio probabilities and',length(rows),
    'posterior summaries from direct normal-prior integration.\n')
