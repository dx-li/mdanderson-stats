# Independent base-R references for the 2007 Dose Schedule Finder model.
# Integrate the triangular hazard numerically instead of using its analytic CDF.
options(digits=17, warn=2)

hazard <- function(u,a,b,c) {
  2*a/(b+c)*pmax(0,pmin(u/b,(b+c-u)/c))
}
cumulative <- function(u,a,b,c) {
  if(u<=0) return(0)
  end <- min(u,b+c)
  bounds <- sort(unique(c(0,min(b,end),end)))
  sum(vapply(seq_len(length(bounds)-1),function(i)
    integrate(function(t) hazard(t,a,b,c),bounds[i],bounds[i+1],
              abs.tol=1e-15,rel.tol=1e-12)$value,numeric(1)))
}
rows <- list()
parameters <- list(c(.12,2,3),c(.035,18,10),c(2.1,.3,1.7))
for(case in seq_along(parameters)) {
  p <- parameters[[case]]; a<-p[1]; b<-p[2]; c<-p[3]
  times <- unique(c(-1,0,1e-8,b/2,b,b+c/2,b+c,2*(b+c)))
  for(u in times) {
    H <- cumulative(u,a,b,c)
    rows[[length(rows)+1]] <- data.frame(case=case,area=a,peak=b,tail=c,
      elapsed=u,hazard=hazard(u,a,b,c),cumulative=H,risk=-expm1(-H))
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/dose-schedule-hazards.csv',row.names=FALSE)

area <- c(.07,.14); peak <- c(2,1); tail <- c(3,4)
cases <- list(
  first=list(time=1,event=1,s=c(0),d=c(0)),
  overlapping=list(time=3,event=1,s=c(0,1),d=c(0,1)),
  censored=list(time=7,event=0,s=c(0,1,4),d=c(0,1,0)),
  saturated=list(time=20,event=0,s=c(0,1,4),d=c(0,1,0)),
  initial_censoring=list(time=0,event=0,s=c(0),d=c(0)),
  delayed=list(time=6,event=1,s=c(2,3,5),d=c(1,0,1)))
pack <- function(x) paste(x,collapse='|')
rows <- list()
for(name in names(cases)) {
  case <- cases[[name]]; j <- case$d+1
  h <- sum(hazard(case$time-case$s,area[j],peak[j],tail[j]))
  H <- sum(vapply(seq_along(j),function(i)
    cumulative(case$time-case$s[i],area[j[i]],peak[j[i]],tail[j[i]]),numeric(1)))
  rows[[length(rows)+1]] <- data.frame(case=name,time=case$time,event=case$event,
    administration_times=pack(case$s),dose_indices=pack(case$d),area=pack(area),
    peak=pack(peak),tail=pack(tail),hazard=h,cumulative=H,risk=-expm1(-H),
    log_likelihood=if(case$event==1) log(h)-H else -H)
}
write.csv(do.call(rbind,rows),'tests/fixtures/dose-schedule-histories.csv',row.names=FALSE)

# Paper's approximate moment elicitation, using the stated tuning values.
p <- c(.2,.25,.3); q <- -log1p(-p); m <- 5
v <- log(1.5/(1.5-1))
rows <- data.frame(dose=0:2,probability=p,administrations=m,peak_mean=c(18,14,10),
  tail_mean=c(10,14,18),lambda_area=1.5,lambda_time=1.5,
  log_area_mean=log(diff(c(0,q))/m)-v/2,
  log_peak_mean=log(c(18,14,10))-v/2,
  log_tail_mean=log(c(10,14,18))-v/2,log_variance=v)
write.csv(rows,'tests/fixtures/dose-schedule-prior.csv',row.names=FALSE)

# Reduced one-dose posterior: log(a)~N(-1,.7^2), b=2 and c=3 fixed.
# Event likelihood factors as a^D * exp(-a*E) times known hazard factors.
times <- c(1,3,4,7)
events <- c(1,1,0,0)
administrations <- list(c(0),c(0,1),c(0,2),c(0,1,4))
E <- sum(vapply(seq_along(times),function(i)
  sum(vapply(times[i]-administrations[[i]],cumulative,numeric(1),a=1,b=2,c=3)),numeric(1)))
D <- sum(events)
mu <- -1; sd <- .7
lp <- function(z) D*z-E*exp(pmin(z,700))+dnorm(z,mu,sd,log=TRUE)
mode <- optimize(function(z) -lp(z),c(-10,5))$minimum
shift <- lp(mode)
integral <- function(power=0,f=function(z) rep(1,length(z)),lower=-Inf,upper=Inf) {
  limits <- sort(unique(c(lower,upper,mode+c(-10,-3,-1,0,1,3,10)*sd)))
  limits <- limits[limits>=lower & limits<=upper]
  sum(vapply(seq_len(length(limits)-1),function(i)
    integrate(function(z) exp(lp(z)-shift+power*z)*f(z),limits[i],limits[i+1],
              abs.tol=1e-12,rel.tol=1e-10,subdivisions=200)$value,numeric(1)))
}
Z <- integral()
zmean <- integral(f=function(z) z)/Z
amean <- integral(power=1)/Z
target <- .3
cutoff <- log(-log1p(-target)/2)
result <- data.frame(log_area_prior_mean=mu,log_area_prior_sd=sd,peak=2,tail=3,
  event_count=D,exposure=E,log_area_mean=zmean,
  log_area_sd=sqrt(integral(f=function(z) (z-zmean)^2)/Z),
  area_mean=amean,area_sd=sqrt(integral(power=2)/Z-amean^2),
  two_administration_risk=integral(f=function(z) -expm1(-2*exp(pmin(z,700))))/Z,
  toxicity_limit=target,overdose_probability=integral(lower=cutoff)/Z)
write.csv(result,'tests/fixtures/dose-schedule-posterior.csv',row.names=FALSE)
