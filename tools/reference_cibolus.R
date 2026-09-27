# Independent base-R quadrature for the CiBolus response/toxicity model.
# Integrate the continuous hazard, avoiding the implementation's analytic
# cumulative-hazard formula; integrate density directly for interval masses.
options(digits=17, warn=2)

default <- c(.5,.7,.8,.08,1.4,1.6,.03,.9,.12,.25,.2)
published <- exp(c(-1.04,-1.60,-7.25,-4.74,-2.85,2.37,
                  -6.10,-3.79,-7.05,-5.42,-7.88))
pack <- function(x) paste(x,collapse='|')
bolus_exposure <- function(c,q,p) p[1]*c^p[2]*q^p[3]
hazard <- function(s,c,q,p) {
  d <- c^p[2]*(q^p[3]+(1-q^p[3])*s)
  p[4]+p[5]*p[6]*d^(p[6]-1)/(1+p[5]*d^p[6])
}
cumulative <- function(s,c,q,p) {
  if(s==0) return(0)
  integrate(function(t) hazard(t,c,q,p),0,s,
            rel.tol=1e-11,abs.tol=1e-13,subdivisions=200)$value
}
survival <- function(s,c,q,p) {
  exp(-bolus_exposure(c,q,p)-cumulative(s,c,q,p))
}
tox_exposure <- function(s,c,q,p,failure=FALSE) {
  p[7]+p[9]*c^p[8]*q+p[10]*c^p[8]*(1-q)*min(s,1)+p[11]*failure
}
tox <- function(s,c,q,p,failure=FALSE) -expm1(-tox_exposure(s,c,q,p,failure))
interval_mass <- function(lower,upper,c,q,p) {
  integrate(function(t) vapply(t,function(s)
    survival(s,c,q,p)*hazard(s,c,q,p),numeric(1)),lower,upper,
    rel.tol=1e-10,abs.tol=1e-12,subdivisions=200)$value
}
cases <- list(
  ordinary=list(c=.2,q=.1,p=default),
  no_bolus=list(c=.4,q=0,p=default),
  all_bolus=list(c=.7,q=1,p=default),
  near_constant=list(c=.2,q=.1,p=replace(default,3,1e-18)),
  underflowed_power=list(c=.1,q=.2,p=replace(default,c(2,6),c(1000,1))),
  steep_response=list(c=.4,q=.1,p=replace(default,c(2,3,6),c(.0001,.3,1000))),
  published_medians=list(c=.2,q=.1,p=published))
rows <- list()
for(name in names(cases)) {
  case <- cases[[name]]; c<-case$c; q<-case$q; p<-case$p
  for(s in c(0,.125,.5,1)) {
    H <- cumulative(s,c,q,p)
    rows[[length(rows)+1]] <- data.frame(case=name,concentration=c,bolus=q,
      parameters=pack(p),time=s,continuous_hazard=if(s==0 && q==0) NA else hazard(s,c,q,p),
      continuous_cumulative=H,bolus_probability=-expm1(-bolus_exposure(c,q,p)),
      response_probability=-expm1(-bolus_exposure(c,q,p)-H),
      toxicity=tox(s,c,q,p),failure_toxicity=tox(1,c,q,p,TRUE))
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/cibolus-probabilities.csv',row.names=FALSE)

endpoints <- c(.25,.5,.75,1)
joint_rows <- list()
for(name in c('ordinary','no_bolus','all_bolus','steep_response','published_medians')) {
  case <- cases[[name]]; c<-case$c; q<-case$q; p<-case$p
  mass <- c(-expm1(-bolus_exposure(c,q,p)),
    vapply(seq_along(endpoints),function(i)
      interval_mass(c(0,endpoints)[i],endpoints[i],c,q,p),numeric(1)),
    survival(1,c,q,p))
  stopifnot(abs(sum(mass)-1)<1e-9)
  pt <- c(tox(0,c,q,p),vapply(endpoints,tox,numeric(1),c=c,q=q,p=p),tox(1,c,q,p,TRUE))
  for(i in seq_along(mass)) {
    joint_rows[[length(joint_rows)+1]] <- data.frame(case=name,
      concentration=c,bolus=q,parameters=pack(p),endpoints=pack(endpoints),
      category=i-1,response_mass=mass[i],no_toxicity=mass[i]*(1-pt[i]),
      toxicity=mass[i]*pt[i])
  }
}
write.csv(do.call(rbind,joint_rows),'tests/fixtures/cibolus-joint.csv',row.names=FALSE)

patients <- data.frame(kind=c('bolus','exact','interval','no_response','bolus','exact'),
  concentration=c(.2,.3,.4,.2,.3,.4),bolus=c(.1,.2,.1,.2,.1,.2),
  lower=c(0,.4,.25,1,0,1),upper=c(0,.4,.5,1,0,1),toxicity=c(1,0,1,0,0,0))
ll_rows <- list()
for(i in seq_len(nrow(patients))) {
  r <- patients[i,]; c<-r$concentration; q<-r$bolus; p<-default
  response <- switch(r$kind,
    bolus=-expm1(-bolus_exposure(c,q,p)),
    exact=survival(r$upper,c,q,p)*hazard(r$upper,c,q,p),
    interval=interval_mass(r$lower,r$upper,c,q,p),
    no_response=survival(1,c,q,p))
  pt <- tox(r$upper,c,q,p,r$kind=='no_response')
  ll_rows[[i]] <- cbind(r,parameters=pack(p),
    log_likelihood=log(response)+if(r$toxicity) log(pt) else log1p(-pt))
}
write.csv(do.call(rbind,ll_rows),'tests/fixtures/cibolus-likelihood.csv',row.names=FALSE)

# Reduced posterior: only log(beta0) varies; response contributions cancel.
# Preserve complete response/toxicity records for the Python comparison.
mu <- -2; sd <- .6
fixed <- default; fixed[7] <- 0
offsets <- vapply(seq_len(nrow(patients)),function(i) {
  r <- patients[i,]
  tox_exposure(r$upper,r$concentration,r$bolus,fixed,r$kind=='no_response')
},numeric(1))
lp <- function(z) vapply(z,function(value) {
  B <- exp(min(value,700))+offsets
  sum(ifelse(patients$toxicity==1,log(-expm1(-B)),-B))+dnorm(value,mu,sd,log=TRUE)
},numeric(1))
mode <- optimize(function(z) -lp(z),c(-12,4))$minimum
shift <- lp(mode)
integral <- function(f=function(z) rep(1,length(z)),lower=-Inf) {
  limits <- sort(unique(c(lower,Inf,mode+c(-10,-3,-1,0,1,3,10)*sd)))
  limits <- limits[limits>=lower]
  sum(vapply(seq_len(length(limits)-1),function(i)
    integrate(function(z) exp(lp(z)-shift)*f(z),limits[i],limits[i+1],
      abs.tol=1e-12,rel.tol=1e-10,subdivisions=200)$value,numeric(1)))
}
Z <- integral()
zmean <- integral(f=function(z) z)/Z
bmean <- integral(f=function(z) exp(pmin(z,700)))/Z
non_tox_multiplier <- integral(f=function(z) exp(-exp(pmin(z,700))))/Z
c <- .2; q <- .1
k <- tox_exposure(1,c,q,fixed,FALSE)
limit <- .2
bcut <- -log1p(-limit)-k
result <- data.frame(log_beta0_prior_mean=mu,log_beta0_prior_sd=sd,
  log_beta0_mean=zmean,log_beta0_sd=sqrt(integral(f=function(z) (z-zmean)^2)/Z),
  beta0_mean=bmean,beta0_sd=sqrt(integral(f=function(z) exp(pmin(2*z,700)))/Z-bmean^2),
  concentration=c,bolus=q,response_at_one_toxicity_mean=1-exp(-k)*non_tox_multiplier,
  toxicity_limit=limit,overdose_probability=if(bcut<=0) 1 else integral(lower=log(bcut))/Z)
write.csv(result,'tests/fixtures/cibolus-posterior.csv',row.names=FALSE)
