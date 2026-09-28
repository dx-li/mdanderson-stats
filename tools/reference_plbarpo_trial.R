# Independent beta tails and hand-specified complete-outcome platform replays.
# The scheduling examples test the declared Python protocol, not native parity.
options(warn=2,digits=17)
cases <- list(
  cap_replace=list(arms=c(0,1,0,2,1,2),response=c(1,0,1,0,1,0),
    pfut=.99,peff=.99,pfinal=.8,
    events=list(c(2,0,1),c(3,0),c(4,1,2),c(5,1),c(6,2)),
    stage=c('early','cap','early','cap','final'),
    entered=c(TRUE,TRUE,TRUE),cap=c(TRUE,TRUE,TRUE),
    fut=c(FALSE,FALSE,FALSE),eff=c(FALSE,FALSE,FALSE),
    assessed=c(TRUE,TRUE,TRUE),final=c(TRUE,FALSE,FALSE),stop_n=c(3,5,6)),
  stop_replace=list(arms=c(0,0,1,1,2,2),response=c(0,0,1,1,1,0),
    pfut=.8,peff=.8,pfinal=.8,
    events=list(c(2,0),c(4,1),c(6,2)),stage=c('early','early','final'),
    entered=c(TRUE,TRUE,TRUE),cap=c(FALSE,FALSE,FALSE),
    fut=c(TRUE,FALSE,FALSE),eff=c(FALSE,TRUE,FALSE),
    assessed=c(FALSE,FALSE,TRUE),final=c(FALSE,FALSE,FALSE),stop_n=c(2,4,6)),
  all_futile=list(arms=c(0,1),response=c(0,0),pfut=.6,peff=.99,pfinal=.8,
    events=list(c(2,0,1)),stage='early',entered=c(TRUE,TRUE,FALSE),
    cap=c(FALSE,FALSE,FALSE),fut=c(TRUE,TRUE,FALSE),eff=c(FALSE,FALSE,FALSE),
    assessed=c(FALSE,FALSE,FALSE),final=c(FALSE,FALSE,FALSE),stop_n=c(2,2,0)))
patient_rows <- list();event_rows <- list();final_rows <- list()
for(name in names(cases)) {
  s <- cases[[name]]
  # At n=2 in cap_replace, the two posteriors are Beta(2,1), Beta(1,2).
  best0 <- integrate(function(x)dbeta(x,2,1)*pbeta(x,1,2),0,1,
                     rel.tol=1e-12)$value
  stopifnot(abs(best0-5/6)<1e-12)
  barn <- c(best0,1-best0)^(2/(2*6));barn <- barn/sum(barn)
  probabilities <- switch(name,
    cap_replace=rbind(c(.5,.5,0),c(0,1,0),c(barn,0),c(0,0,1),c(0,.5,.5),c(0,0,1)),
    stop_replace=rbind(c(1,0,0),c(1,0,0),c(0,1,0),c(0,1,0),c(0,0,1),c(0,0,1)),
    all_futile=rbind(c(.5,.5,0),c(0,1,0)))
  patient_rows[[length(patient_rows)+1L]] <- data.frame(scenario=name,
    patient=seq_along(s$arms),arm=s$arms,response=s$response,
    probability_arm0=probabilities[,1],probability_arm1=probabilities[,2],
    probability_arm2=probabilities[,3])
  for(j in seq_along(s$events)) {
    event <- s$events[[j]];n <- event[1]
    for(arm in event[-1]) {
      outcomes <- s$response[seq_len(n)][s$arms[seq_len(n)]==arm]
      a <- 1+sum(outcomes);b <- 1+length(outcomes)-sum(outcomes)
      lower <- pbeta(.5,a,b);upper <- pbeta(.5,a,b,lower.tail=FALSE)
      event_rows[[length(event_rows)+1L]] <- data.frame(scenario=name,enrolled=n,
        arm=arm,stage=s$stage[j],alpha=a,beta=b,futility_probability=lower,
        efficacy_probability=upper,final_probability=upper,pfut=s$pfut,
        peff=s$peff,pfinal=s$pfinal,futile=lower>s$pfut,
        efficacious=upper>=s$peff,final_efficacious=upper>=s$pfinal)
    }
  }
  for(arm in 0:2) {
    y <- s$response[s$arms==arm];i <- arm+1L
    final_rows[[length(final_rows)+1L]] <- data.frame(scenario=name,arm=arm,
      assigned=length(y),successes=sum(y),failures=length(y)-sum(y),
      entered=s$entered[i],cap_closed=s$cap[i],early_futility=s$fut[i],
      early_efficacy=s$eff[i],final_assessed=s$assessed[i],final_efficacy=s$final[i],
      any_efficacy=s$eff[i]||s$final[i],arm_stop_enrollment=s$stop_n[i])
  }
}
write.csv(do.call(rbind,patient_rows),'tests/fixtures/plbarpo-trial-patients.csv',row.names=FALSE)
write.csv(do.call(rbind,event_rows),'tests/fixtures/plbarpo-trial-monitor.csv',row.names=FALSE)
write.csv(do.call(rbind,final_rows),'tests/fixtures/plbarpo-trial-final.csv',row.names=FALSE)
cat('Generated three explicit platform replay references with independent beta tails.\n')
