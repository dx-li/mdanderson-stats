# Independent beta-ordering integrals for hand-specified platform replays.
# These validate the declared Python scheduler, not native scheduler/RNG parity.
options(warn=2, digits=17)
base <- list(arms=c(0,1,1,2,0,2), response=c(1,0,0,1,0,1),
  entry=c(0,0,3), stop=c(6,3,6), cap=c(FALSE,TRUE,TRUE),
  fut=c(FALSE,FALSE,FALSE), final_n=c(NA,3,6), pfinal=.85,
  events=list(c(3,1),c(6,2)), stages=c('final','final'),
  probability=rbind(c(.5,.5,0),c(0,1,0),c(.5,.5,0),
                    c(0,0,1),c(.5,0,.5),c(.5,0,.5)))
cases <- list(entire_replace=c(base,list(mode='entire')),
              concurrent_replace=c(base,list(mode='concurrent')),
  unrelated_cap=list(arms=c(0,2,1,0,2,2),response=c(0,1,1,0,1,1),
    entry=c(0,0,0),stop=c(6,3,6),cap=c(FALSE,TRUE,TRUE),
    fut=c(FALSE,FALSE,FALSE),final_n=c(NA,3,6),pfinal=.7,mode='entire',
    events=list(c(3,1),c(6,2)),stages=c('final','final'),
    probability=rbind(c(1/3,1/3,1/3),c(0,.5,.5),c(0,1,0),
                      c(.5,0,.5),c(.5,0,.5),c(.5,0,.5))),
  concurrent_double_replace=list(arms=c(0,1,2,3,4,0,3,4),
    response=c(1,0,0,1,1,1,1,1),entry=c(0,0,0,3,3),stop=c(8,3,3,8,8),
    cap=rep(FALSE,5),fut=c(FALSE,TRUE,TRUE,FALSE,FALSE),
    final_n=c(NA,NA,NA,8,8),pfinal=.7,mode='concurrent',
    events=list(c(3,1,2),c(5,3,4),c(8,3,4)),stages=c('early','early','final'),
    probability=rbind(c(1/3,1/3,1/3,0,0),c(0,.5,.5,0,0),c(0,0,1,0,0),
                      c(0,0,0,.5,.5),c(0,0,0,0,1),c(1/3,0,0,1/3,1/3),
                      c(1/3,0,0,1/3,1/3),c(1/3,0,0,1/3,1/3))))

posterior <- function(s,arm,n) {
  rows <- seq_len(n)
  y <- s$response[rows][s$arms[rows]==arm]
  eligible <- s$arms[rows]==0
  if(s$mode=='concurrent') eligible <- eligible & (rows-1)>=s$entry[arm+1L]
  cy <- s$response[rows][eligible]
  a <- 1+sum(y); b <- 1+length(y)-sum(y)
  ca <- 1+sum(cy); cb <- 1+length(cy)-sum(cy)
  upper <- integrate(function(x)dbeta(x,a,b)*pbeta(x,ca,cb),0,1,rel.tol=1e-12)$value
  lower <- integrate(function(x)dbeta(x,a,b)*pbeta(x,ca,cb,lower.tail=FALSE),
                     0,1,rel.tol=1e-12)$value
  stopifnot(abs(upper+lower-1)<1e-12)
  c(control_successes=sum(cy),control_failures=length(cy)-sum(cy),
    treatment_greater=upper,control_greater=lower)
}

patients <- events <- terminal <- list()
for(name in names(cases)) {
  s <- cases[[name]]; k <- length(s$entry)
  for(n in seq_along(s$arms)) for(arm in 0:(k-1L)) {
    patients[[length(patients)+1L]] <- data.frame(scenario=name,patient=n,
      assigned_arm=s$arms[n],response=s$response[n],arm=arm,
      allocation=s$probability[n,arm+1L])
  }
  for(j in seq_along(s$events)) {
    event <- s$events[[j]]; n <- event[1]
    for(arm in event[-1]) {
      p <- posterior(s,arm,n)
      events[[length(events)+1L]] <- data.frame(scenario=name,enrolled=n,arm=arm,
        stage=s$stages[j],control_successes=p[1],control_failures=p[2],
        treatment_greater=p[3],control_greater=p[4])
    }
  }
  for(arm in 0:(k-1L)) {
    i <- arm+1L; y <- s$response[s$arms==arm]
    assessed <- !is.na(s$final_n[i])
    p <- if(assessed) posterior(s,arm,s$final_n[i]) else rep(NA_real_,4)
    terminal[[length(terminal)+1L]] <- data.frame(scenario=name,arm=arm,
      assigned=length(y),successes=sum(y),failures=length(y)-sum(y),
      entry_index=s$entry[i],stop_index=s$stop[i],cap_closed=s$cap[i],
      early_futility=s$fut[i],final_assessed=assessed,
      final_probability=p[3],final_control_successes=p[1],final_control_failures=p[2],
      final_efficacy=assessed && p[3]>=s$pfinal)
  }
}
stopifnot(abs(posterior(cases$entire_replace,2,6)[3]-.8)<1e-12,
          abs(posterior(cases$concurrent_replace,2,6)[3]-.9)<1e-12,
          abs(posterior(cases$unrelated_cap,2,6)[3]-34/35)<1e-12)
write.csv(do.call(rbind,patients),'tests/fixtures/plbarpo-control-trial-patients.csv',row.names=FALSE)
write.csv(do.call(rbind,events),'tests/fixtures/plbarpo-control-trial-monitor.csv',row.names=FALSE)
write.csv(do.call(rbind,terminal),'tests/fixtures/plbarpo-control-trial-final.csv',row.names=FALSE)
cat('Generated four explicit control platform replays and independent beta comparisons.\n')
