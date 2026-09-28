# Independent BARD carryover/minimization/OBD ledgers under explicit settings.
# Uses the previously validated stage-one patient ledger, not Python Stage-2 outputs.
options(warn=2,digits=17)
ledger <- read.csv('tests/fixtures/bard-blrm-calendar-patients.csv')
h <- subset(ledger, scenario=='evaluable_cap_and_cohort_priority')
index <- seq_len(nrow(h))-1L
hf <- cbind(index%%2+1L,(index%/%2)%%2+1L)
factors <- rbind(c(1,1),c(2,2),c(1,2),c(2,1),c(1,1))
tox <- rbind(c(0,1),c(1,0),c(0,1),c(1,0),c(0,1))
resp <- rbind(c(1,0),c(1,1),c(0,1),c(0,0),c(1,1))
cases <- list(
 eligible_total_cap=list(pair=c(1,2),eligible=index%%3!=1,target=10,candidates=5),
 selected_doses_only=list(pair=c(2,3),eligible=rep(TRUE,nrow(h)),target=9,candidates=5),
 shortage=list(pair=c(1,2),eligible=index%%3!=1,target=11,candidates=2),
 already_full=list(pair=c(1,2),eligible=index%%3!=1,target=7,candidates=5))
patient_rows <- count_rows <- summary_rows <- posterior_rows <- list()
cell <- function(t,e) if(t && !e) 1L else if(!t && !e) 2L else if(t && e) 3L else 4L
for (name in names(cases)) {
 c <- cases[[name]]
 included <- which(c$eligible & h$dose %in% c$pair)
 history <- hf[included,,drop=FALSE]
 arms <- match(h$dose[included],c$pair)
 carry <- matrix(0L,2,4);added <- matrix(0L,2,4)
 for(k in seq_along(included)) {
  i <- included[k]; carry[arms[k],cell(h$dlt[i],h$response[i])] <- carry[arms[k],cell(h$dlt[i],h$response[i])]+1L
 }
 required <- c$target-length(included); enroll <- min(required,c$candidates)
 for(i in seq_len(enroll)) {
  scores <- vapply(1:2,function(arm) {
   difference <- vapply(1:2,function(f) {
    matching <- history[,f]==factors[i,f]
    sum(arms[matching]==1)-sum(arms[matching]==2)
   },numeric(1))
   sum(abs(difference+if(arm==1) 1 else -1))
  },numeric(1))
  # Explicit probability=1, tie_probability=1 isolates controller from RNG.
  arm <- if(scores[1]<=scores[2]) 1L else 2L
  category <- cell(tox[i,arm],resp[i,arm])
  added[arm,category] <- added[arm,category]+1L
  patient_rows[[length(patient_rows)+1L]] <- data.frame(case=name,candidate=i-1L,
   arm=arm,dose=c$pair[arm],toxicity=tox[i,arm],response=resp[i,arm],
   score1=scores[1],score2=scores[2],probability1=as.numeric(arm==1))
  arms <- c(arms,arm);history <- rbind(history,factors[i,])
 }
 counts <- carry+added
 for(a in 1:2) for(k in 1:4) count_rows[[length(count_rows)+1L]] <- data.frame(
  case=name,arm=a,category=k,carry=carry[a,k],added=added[a,k],total=counts[a,k])
 complete <- enroll==required;selected <- NA_integer_
 if(complete) {
  posterior <- counts+.25
  utility <- drop((posterior/rowSums(posterior))%*%c(0,30,50,100))
  overdose <- pbeta(.3,posterior[,1]+posterior[,3],posterior[,2]+posterior[,4],lower.tail=FALSE)
  adjusted <- overdose
  if(overdose[1]>overdose[2]) adjusted[] <- mean(overdose)
  loweff <- pbeta(.2,posterior[,3]+posterior[,4],posterior[,1]+posterior[,2])
  safe <- adjusted<=.95 & loweff<=.95
  if(any(safe)) selected <- which.max(ifelse(safe,utility,-Inf))
  for(a in 1:2) posterior_rows[[length(posterior_rows)+1L]] <- data.frame(
   case=name,arm=a,utility=utility[a],overdose=overdose[a],adjusted=adjusted[a],
   loweff=loweff[a],admissible=safe[a])
 }
 summary_rows[[length(summary_rows)+1L]] <- data.frame(case=name,carryover=length(included),
  required=required,enrolled=enroll,shortfall=required-enroll,completed=complete,
  selected_arm=selected)
}
write.csv(do.call(rbind,patient_rows),'tests/fixtures/bard-integrated-patients.csv',row.names=FALSE)
write.csv(do.call(rbind,count_rows),'tests/fixtures/bard-integrated-counts.csv',row.names=FALSE)
write.csv(do.call(rbind,posterior_rows),'tests/fixtures/bard-integrated-posterior.csv',row.names=FALSE)
write.csv(do.call(rbind,summary_rows),'tests/fixtures/bard-integrated-summary.csv',row.names=FALSE)
