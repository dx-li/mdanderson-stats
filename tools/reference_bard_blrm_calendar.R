# Independent fixed-parameter calendar references for BF-BLRM.
# The assignment ledgers below are enumerated by hand from the protocol rules;
# this script does not implement a second adaptive trial scheduler.
options(warn=2, digits=17)

# logit(p)= -3 + dose, reference=1 and beta=1. Dose 1 is below target;
# doses 2 and 3 are in (.16,.6), so the explicit lowest-PTT-tie rule targets 2.
dose <- 1:3
risk <- plogis(-3+dose)
ptt <- as.integer(risk>.16 & risk<.6)
pod <- as.integer(risk>=.6)
write.csv(data.frame(dose=dose,risk=risk,ptt=ptt,pod=pod),
          'tests/fixtures/bard-blrm-calendar-posterior.csv',row.names=FALSE)

# DLT window=2, cohort size=2, escalation maximum=8, evaluable cap=3.
# Every half-unit arrival is offered; decline indices 2,3,4,12,13,14 (zero based).
# After the first cohort completes at 2.5, escalate to dose 2. While its first
# cohort is pending, dose 1 has two evaluated patients and receives three
# pending backfill patients. The cap closes only when its next DLT assessment
# completes at 5.5. Six patients eventually receive dose 2, permitting final MTD.
main <- data.frame(arrival_index=c(0L,1L,5L,6L,7L,8L,9L,10L,11L,15L,16L),
  dose=c(1L,1L,2L,2L,1L,1L,1L,2L,2L,2L,2L),
  arrival=c(0,.5,2.5,3,3.5,4,4.5,5,5.5,7.5,8),
  backfill=c(FALSE,FALSE,FALSE,FALSE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE),
  dlt=FALSE,response=TRUE)
main$dlt_assessment <- main$arrival+2
main$response_assessment <- main$arrival+.25
main$scenario <- 'evaluable_cap_and_cohort_priority'

# Early DLT completes each cohort member at .25 after enrollment. The first
# cohort is complete at .75 before the next arrival at 1; dose 2 is then chosen.
# Enrollment cap is reached at 1.5, but response follow-up continues until 4.5.
early <- data.frame(arrival_index=0:3,dose=c(1L,1L,2L,2L),
  arrival=c(0,.5,1,1.5),backfill=FALSE,dlt=TRUE,response=TRUE)
early$dlt_assessment <- early$arrival+.25
early$response_assessment <- early$arrival+3
early$scenario <- 'early_dlt_late_response'

# The tape can end with a partial escalation cohort. Its assigned patient is
# fully followed up, with no fabricated extra arrival to fill that cohort.
short <- data.frame(arrival_index=0L,dose=1L,arrival=10,backfill=FALSE,
  dlt=FALSE,response=FALSE,dlt_assessment=12,response_assessment=13,
  scenario='arrival_tape_exhausted')

patients <- rbind(main,early,short)
write.csv(patients,'tests/fixtures/bard-blrm-calendar-patients.csv',row.names=FALSE)

# Independent counts at each actual DLT/response observation time, including
# grouped simultaneous observations. Look up the latest row <= any replay look.
rows <- list()
for(s in unique(patients$scenario)) {
  p <- patients[patients$scenario==s,]
  times <- sort(unique(c(p$arrival,p$dlt_assessment,p$response_assessment)))
  for(t in times) for(j in dose) rows[[length(rows)+1L]] <- data.frame(
    scenario=s,time=t,dose=j,
    assigned=sum(p$dose==j & p$arrival<=t),
    evaluated=sum(p$dose==j & p$dlt_assessment<=t),
    toxicities=sum(p$dose==j & p$dlt_assessment<=t & p$dlt),
    responses=sum(p$dose==j & p$response_assessment<=t & p$response))
}
write.csv(do.call(rbind,rows),'tests/fixtures/bard-blrm-calendar-counts.csv',row.names=FALSE)
write.csv(data.frame(
  scenario=c('evaluable_cap_and_cohort_priority','early_dlt_late_response',
             'arrival_tape_exhausted'),
  enrolled=c(11L,4L,1L),escalation=c(8L,4L,1L),backfill=c(3L,0L,0L),
  declined=c(6L,0L,0L),stop_time=c(8,1.5,10),final_time=c(10,4.5,13),
  duration=c(10,4.5,3),selected_dose=c(2L,NA_integer_,NA_integer_)),
  'tests/fixtures/bard-blrm-calendar-summary.csv',row.names=FALSE)
cat('Generated',nrow(patients),'hand-enumerated patient rows and',length(rows),
    'as-of count rows for three calendar paths.\n')
