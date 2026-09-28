# Independent small calendar examples: direct beta tails and enumeration of
# possible pending endpoint totals. Open-accrual-clock freezing is an explicit
# Python replay convention, not a claim about native random-stream behavior.
options(warn=2, digits=17)

cases <- list(
  cap_with_pending=list(r=rep(0,6),t=rep(0,6),interval=rep(1,5),
    rd=rep(0,6),td=rep(100,6),rp=c(1,1),tp=c(1,1),rc=1,tc=.95),
  resume=list(r=rep(0,6),t=rep(0,6),interval=rep(1,5),
    rd=rep(0,6),td=c(10,9,0,0,0,0),rp=c(1,1),tp=c(1.8,.2),rc=1,tc=.8),
  toxicity_stop=list(r=rep(1,6),t=c(1,0,0,0,0,0),interval=rep(1,5),
    rd=rep(20,6),td=c(10,9,0,0,0,0),rp=c(1,1),tp=c(1.8,.2),rc=1,tc=.8),
  response_certain=list(r=rep(0,6),t=rep(1,6),interval=rep(1,5),
    rd=rep(0,6),td=rep(100,6),rp=c(.2,1.8),tp=c(1,1),rc=.8,tc=.8),
  tied_both=list(r=rep(0,6),t=rep(1,6),interval=rep(0,5),
    rd=rep(0,6),td=rep(0,6),rp=c(.2,1.8),tp=c(1.8,.2),rc=.8,tc=.8))

patients <- looks <- summaries <- list()
for(name in names(cases)) {
  x <- cases[[name]]
  a <- ra <- ta <- numeric(6)
  now <- paused <- 0
  actions <- c('continue','wait')
  for(n in 1:6) {
    if(n>1) now <- now+x$interval[n-1]
    a[n] <- now; ra[n] <- now+x$rd[n]; ta[n] <- now+x$td[n]
    repeat {
      rk <- ra[1:n]<=now; tk <- ta[1:n]<=now
      r <- sum(x$r[1:n][rk]); t <- sum(x$t[1:n][tk])
      rpending <- sum(!rk); tpending <- sum(!tk)
      action <- 'continue'
      if(n==6) action <- 'cap_complete'
      else if(n%%2==0) {
        # Enumerate hypothetical final counts rather than inspecting future bits.
        possibilities <- expand.grid(r=r:(r+rpending),t=t:(t+tpending))
        rstop <- pbeta(.5,x$rp[1]+possibilities$r,
                      x$rp[2]+n-possibilities$r)>x$rc
        tstop <- pbeta(.5,x$tp[1]+possibilities$t,
                      x$tp[2]+n-possibilities$t,lower.tail=FALSE)>x$tc
        action <- if(all(rstop)&all(tstop)) 'stop_both' else
          if(all(rstop)) 'stop_response' else if(all(tstop)) 'stop_toxicity' else
          if(any(rstop|tstop)) 'wait' else 'continue'
      }
      if(n%%2==0) looks[[length(looks)+1L]] <- data.frame(scenario=name,n=n,
        time=now,response_observed=r,toxicity_observed=t,
        response_pending=rpending,toxicity_pending=tpending,action=action)
      if(action!='wait') break
      next_time <- min(c(ra[1:n][!rk],ta[1:n][!tk]))
      stopifnot(next_time>now)
      paused <- paused+next_time-now; now <- next_time
    }
    if(!action %in% actions) break
  }
  for(i in 1:n) patients[[length(patients)+1L]] <- data.frame(scenario=name,
    patient=i-1L,arrival=a[i],response_available=ra[i],toxicity_available=ta[i],
    response_known=ra[i]<=now,toxicity_known=ta[i]<=now)
  summaries[[length(summaries)+1L]] <- data.frame(scenario=name,enrolled=n,
    decision=action,decision_time=now,duration=max(now,ra[1:n],ta[1:n]),
    paused_duration=paused,responses=sum(x$r[1:n]),toxicities=sum(x$t[1:n]),
    observed_responses=r,observed_toxicities=t)
}
write.csv(do.call(rbind,patients),'tests/fixtures/multc-calendar-patients.csv',row.names=FALSE)
write.csv(do.call(rbind,looks),'tests/fixtures/multc-calendar-looks.csv',row.names=FALSE)
write.csv(do.call(rbind,summaries),'tests/fixtures/multc-calendar-summary.csv',row.names=FALSE)
cat('Generated',length(patients),'patient rows,',length(looks),'look rows and',
    length(summaries),'calendar summaries.\n')
