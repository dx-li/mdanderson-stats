# Independent calendar observation, Gamma posterior and analytic OC references.
options(warn=2, digits=17)
standard <- function(enrollment,event,followup,looks=c(2L,4L,6L),
                     lrv=3,cmv=5,ll=.8,lc=.5,gl=.5,gc=.5,a=1,b=2) {
  list(enrollment=enrollment,event=event,followup=followup,looks=looks,
       lrv=lrv,cmv=cmv,ll=ll,lc=lc,gl=gl,gc=gc,a=a,b=b)
}
arrival <- c(.5,1,2,4,5,6)
cases <- list(
  no_events=standard(arrival,rep(Inf,6),12),
  early_no_go=standard(arrival,rep(.05,6),12),
  final_no_go=standard(arrival,rep(.05,6),12,looks=6L),
  final_consider=standard(rep(0,40),c(rep(3,20),rep(Inf,20)),4,
    looks=40L,ll=.9,lc=.5,a=1e-6,b=1e-6),
  boundary_events=standard(0:5,c(1,0,1.5,Inf,0,Inf),3,ll=.01,lc=.005),
  shifted_origin=standard(1000+arrival,rep(Inf,6),12),
  large_units=standard(arrival*1e200,rep(Inf,6),12e200,
    lrv=3e200,cmv=5e200,b=2e200),
  small_units=standard(arrival*1e-200,rep(Inf,6),12e-200,
    lrv=3e-200,cmv=5e-200,b=2e-200)
)
input <- output <- list()
for (name in names(cases)) {
  x <- cases[[name]]
  N <- length(x$enrollment)
  input[[name]] <- data.frame(case=name,index=seq_len(N)-1L,
    enrollment=x$enrollment,event=x$event,followup=x$followup,
    looks=paste(x$looks,collapse=';'),lrv=x$lrv,cmv=x$cmv,
    lambda_lrv=x$ll,lambda_cmv=x$lc,gamma_lrv=x$gl,gamma_cmv=x$gc,
    prior_shape=x$a,prior_scale=x$b)
  history <- list()
  for (n in x$looks) {
    clock <- x$enrollment[n]+if (n==N) x$followup else 0
    risk_time <- x$enrollment[n]-x$enrollment[seq_len(n)]+if (n==N) x$followup else 0
    observed <- pmin(x$event[seq_len(n)],risk_time)
    d <- sum(x$event[seq_len(n)]<=risk_time)
    T <- sum(observed)
    pl <- pgamma((x$b+T)/x$lrv*log(2),shape=x$a+d)
    pc <- pgamma((x$b+T)/x$cmv*log(2),shape=x$a+d)
    lower <- x$ll*(n/N)^x$gl
    upper <- x$lc*(n/N)^x$gc
    decision <- if (n<N) {
      if (pl<lower && pc<upper) 'stop_no_go' else 'continue'
    } else if (pl>lower && pc>upper) 'final_go'
    else if (pl<lower && pc<upper) 'final_no_go' else 'final_consider'
    history[[length(history)+1L]] <- data.frame(case=name,enrolled=n,events=d,
      total_time=T,analysis_time=clock,posterior_lrv=pl,posterior_cmv=pc,
      decision=decision)
    if (decision!='continue') break
  }
  output[[name]] <- do.call(rbind,history)
}
write_exact <- function(rows,path) {
  data <- do.call(rbind,rows)
  for (name in names(data)) if (is.numeric(data[[name]])) data[[name]] <- sprintf('%.17g',data[[name]])
  write.csv(data,path,row.names=FALSE)
}
write_exact(input,'tests/fixtures/bop2-dc-survival-trial-inputs.csv')
write_exact(output,'tests/fixtures/bop2-dc-survival-trial-reference.csv')

# N=1, look=1, prior IG(1,2), LRV=3, CMV=5, cutoffs .6/.4,
# followup=3: every event before followup is no-go; every survivor is go.
stopifnot(pgamma(5*log(2)/3,2)<.6,pgamma(5*log(2)/5,2)<.4,
          pgamma(5*log(2)/3,1)>.6,pgamma(5*log(2)/5,1)>.4)
mean_time <- 6/log(2)
survival <- exp(-3/mean_time)
analytic <- c(final_go=survival,final_no_go=1-survival,final_consider=0,
  stop_no_go=0,mean_enrollment=1,mean_events=1-survival,
  mean_total_time=mean_time*(1-survival),mean_duration=3.5)
write_exact(list(data.frame(metric=names(analytic),value=unname(analytic))),
  'tests/fixtures/bop2-dc-survival-analytic-oc.csv')
cat('Generated eight calendar references and one analytic survival OC case.\n')
