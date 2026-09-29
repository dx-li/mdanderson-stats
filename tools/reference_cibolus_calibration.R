# Independent reduced CiBolus posterior / prior-predictive quadrature.
# Only log_beta0 varies. Response likelihood cancels in this validation slice.
options(warn = 2, digits = 17)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop('usage: Rscript reference_cibolus_calibration.R <fixture-dir>')
dir <- args[[1L]]
counts <- read.csv(file.path(dir,'joint_counts.csv'), check.names=FALSE)
regimens <- read.csv(file.path(dir,'regimens.csv'), check.names=FALSE)
endpoints <- read.csv(file.path(dir,'endpoints.csv'), check.names=FALSE)$endpoint
prior <- read.csv(file.path(dir,'prior.csv'), check.names=FALSE)
metrics <- read.csv(file.path(dir,'prior_metrics.csv'), check.names=FALSE)
coordinates <- c(paste0('log_alpha',0:5),paste0('log_beta',0:4))
if (!identical(as.character(prior$coordinate),coordinates)) stop('prior coordinates/order mismatch')
if (!all(c('replicate','regimen','response_category','toxicity','count') %in% names(counts))) stop('joint_counts schema mismatch')
if (any(counts$count < 0) || any(counts$count != floor(counts$count)) || any(counts$toxicity %in% c(0,1)==FALSE)) stop('invalid grouped pseudo counts')
obs <- merge(counts,regimens,by='regimen',sort=FALSE)
if (nrow(obs)!=nrow(counts)) stop('unknown regimen in grouped counts')
category <- obs$response_category
if (any(category < 0 | category > length(endpoints)+1 | category != floor(category))) stop('response category out of range')
obs$failure <- as.integer(category == length(endpoints)+1)
obs$time <- ifelse(category==0,0,
  ifelse(category==length(endpoints)+1,1,endpoints[pmax(1,category)]))

risk <- function(z, concentration, bolus, time, failure, logb1, logb2, logb3, logb4) {
  b0 <- exp(z)
  cpower <- exp(exp(logb1) * log(concentration))
  expo <- b0 + exp(logb2) * cpower * bolus +
    exp(logb3) * cpower * (1 - bolus) * min(time, 1) +
    if (failure == 1) exp(logb4) else 0
  -expm1(-expo)
}
coordinate <- function(name,field) {
  i <- match(name,prior$coordinate)
  prior[[field]][[i]]
}
integrate_normal <- function(f, mean, sd, center=mean) {
  lo <- min(mean-14*sd,center-14*sd); hi <- max(mean+14*sd,center+14*sd)
  cuts <- sort(unique(c(lo,mean+c(-10,-6,-3,0,3,6,10)*sd,
                        center+c(-10,-6,-3,0,3,6,10)*sd,hi)))
  cuts <- cuts[cuts>=lo & cuts<=hi]
  sum(vapply(seq_len(length(cuts)-1L),function(i)
    integrate(f,cuts[i],cuts[i+1L],rel.tol=2e-11,abs.tol=2e-13,
              subdivisions=300L)$value,numeric(1)))
}

posterior_row <- function(g) {
  m <- coordinate('log_beta0','mean'); s <- coordinate('log_beta0','sd')
  if (!is.finite(m)||!is.finite(s)||s<=0) stop('reduced oracle requires positive SD on log_beta0')
  loglik <- function(z) {
    ans <- 0
    for (i in seq_len(nrow(g))) {
      if (g$count[i] == 0) next
      p <- risk(z,g$concentration[i],g$bolus[i],g$time[i],g$failure[i],
        coordinate('log_beta1','mean'),coordinate('log_beta2','mean'),
        coordinate('log_beta3','mean'),coordinate('log_beta4','mean'))
      ans <- ans + g$count[i]*if(g$toxicity[i]==1) log(p) else log1p(-p)
    }
    ans
  }
  logpost <- function(z) loglik(z)+dnorm(z,m,s,log=TRUE)
  mode <- optimize(function(z)-logpost(z),c(m-14*s,m+14*s))$minimum
  offset <- logpost(mode); weight <- function(z) exp(logpost(z)-offset)
  zmean <- integrate_normal(function(z) z*weight(z),m,s,mode)/integrate_normal(weight,m,s,mode)
  z2 <- integrate_normal(function(z) z^2*weight(z),m,s,mode)/integrate_normal(weight,m,s,mode)
  answer <- data.frame(replicate=g$replicate[[1L]],prior_mean_log_beta0=m,
    prior_sd_log_beta0=s,posterior_sd_log_beta0=sqrt(max(0,z2-zmean^2)),
    posterior_mode_log_beta0=mode,
    log_marginal_likelihood=offset+log(integrate_normal(weight,m,s,mode)),
    n_observations=sum(g$count))
  for(k in 0:5) answer[[paste0('log_alpha',k)]] <- coordinate(paste0('log_alpha',k),'mean')
  answer$log_beta0 <- zmean
  for(k in 1:4) answer[[paste0('log_beta',k)]] <- coordinate(paste0('log_beta',k),'mean')
  answer
}
posterior <- do.call(rbind,lapply(split(obs,obs$replicate),posterior_row))
write.csv(posterior,file.path(dir,'posterior.csv'),row.names=FALSE)

response_metric <- function(metric,concentration,bolus) {
  alpha <- exp(vapply(0:5,function(k) coordinate(paste0('log_alpha',k),'mean'),numeric(1)))
  if (metric == 'p0') {
    return(-expm1(-alpha[1]*concentration^alpha[2]*bolus^alpha[3]))
  }
  if (metric == 'response_at_one') {
    d <- function(t) concentration^alpha[2]*(bolus^alpha[3]+(1-bolus^alpha[3])*t)
    h <- function(t) alpha[4]+alpha[5]*alpha[6]*d(t)^(alpha[6]-1)/(1+alpha[5]*d(t)^alpha[6])
    continuous <- integrate(h,0,1,rel.tol=2e-11,abs.tol=2e-13,subdivisions=300L)$value
    bolus_exposure <- alpha[1]*concentration^alpha[2]*bolus^alpha[3]
    return(-expm1(-bolus_exposure-continuous))
  }
  stop('unknown response metric')
}
prior_metric_row <- function(r) {
  m <- r$prior_mean_log_beta0; s <- r$prior_sd_log_beta0
  if (r$metric %in% c('p0','response_at_one')) {
    mean <- response_metric(r$metric,r$concentration,r$bolus)
    variance <- 0
    ess <- if (mean>0&&mean<1) Inf else NA_real_
  } else {
    failure <- if (r$metric=='toxicity_after_failure') 1 else 0
    time <- if (r$metric=='toxicity_at_zero_response') 0 else r$time
    toxfun <- function(z) risk(z,r$concentration,r$bolus,time,failure,
      coordinate('log_beta1','mean'),coordinate('log_beta2','mean'),
      coordinate('log_beta3','mean'),coordinate('log_beta4','mean'))
    if (s == 0) {
      mean <- toxfun(m); variance <- 0
      ess <- if (mean>0 && mean<1) Inf else NA_real_
    } else {
      mean <- integrate_normal(function(z) dnorm(z,m,s)*toxfun(z),m,s)
      second <- integrate_normal(function(z) dnorm(z,m,s)*toxfun(z)^2,m,s)
      variance <- max(0,second-mean^2)
      ess <- if (variance==0) { if (mean>0&&mean<1) Inf else NA_real_ } else mean*(1-mean)/variance-1
    }
  }
  data.frame(metric=r$metric,concentration=r$concentration,bolus=r$bolus,prior_predictive_mean=mean,
    prior_predictive_variance=variance,beta_moment_ess=ess)
}
predictive <- do.call(rbind,lapply(seq_len(nrow(metrics)),function(i) prior_metric_row(metrics[i,])))
write.csv(predictive,file.path(dir,'prior_predictive.csv'),row.names=FALSE)
