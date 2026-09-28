# Independent quadrature/inversion for fixed-regimen triangular-hazard events.
# This validates the declared inverse-CDF event generator, not native RNG parity.
options(warn=2, digits=17)

hazard <- function(t,a,b,c) 2*a/(b+c)*pmax(0,pmin(t/b,(b+c-t)/c))
H <- function(t,administrations,a,b,c) {
  sum(vapply(administrations,function(s) {
    end <- min(t-s,b+c)
    if(end<=0) return(0)
    knots <- sort(unique(c(0,min(b,end),end)))
    sum(vapply(seq_len(length(knots)-1L),function(j)
      integrate(function(v) hazard(v,a,b,c),knots[j],knots[j+1L],
                abs.tol=1e-15,rel.tol=1e-12)$value,numeric(1)))
  },numeric(1)))
}
cases <- list(
  rising=list(s=0,a=.4,b=2,c=3,end=5,target=.01),
  peak=list(s=0,a=.4,b=2,c=3,end=5,target=.16),
  falling=list(s=0,a=.4,b=2,c=3,end=5,target=.3),
  no_event=list(s=0,a=.4,b=2,c=3,end=5,target=.5),
  overlapping=list(s=c(0,1),a=.4,b=2,c=3,end=4,target=.3),
  horizon_censor=list(s=c(0,1),a=.4,b=2,c=3,end=4,target=.7),
  late_after_gap=list(s=c(0,10),a=.4,b=2,c=3,end=15,target=.45),
  before_second=list(s=c(0,3,6),a=.4,b=2,c=3,end=10,target=.2),
  three_doses=list(s=c(0,3,6),a=.4,b=2,c=3,end=10,target=1.0),
  high_area=list(s=c(0,1),a=3,b=.3,c=1.7,end=4,target=.9),
  low_area=list(s=c(0,1),a=.05,b=2,c=3,end=6,target=.07))
rows <- list()
for(name in names(cases)) {
  x <- cases[[name]]
  u <- -expm1(-x$target)
  target <- -log1p(-u)
  maxH <- H(x$end,x$s,x$a,x$b,x$c)
  event <- target<=maxH
  time <- if(event) uniroot(function(t) H(t,x$s,x$a,x$b,x$c)-target,
    c(0,x$end),tol=1e-13,maxiter=200)$root else Inf
  if(name=='rising') stopifnot(abs(time-.5)<1e-11)
  if(name=='peak') stopifnot(abs(time-2)<1e-11)
  if(name=='falling') stopifnot(abs(time-(5-sqrt(3.75)))<1e-11)
  for(scale in c(1e-6,1,1e6)) {
    rows[[length(rows)+1L]] <- data.frame(scenario=name,time_scale=scale,
      administration_times=paste(x$s*scale,collapse='|'),
      area=x$a,peak=x$b*scale,tail=x$c*scale,horizon=x$end*scale,
      uniform=u,horizon_cumulative=maxH,event=event,event_time=time*scale,
      actual_administrations=sum(x$s<time & x$s<=x$end))
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/dose-schedule-events.csv',row.names=FALSE)
cat('Generated 33 inverse-event references using independent R quadrature.\n')
