# Original BOIN 2.7.2 waterfall titration with supplied patient outcomes.
# Reuse the source-pinned loader and exhaustive six-cell isotonic oracle from
# the existing workflow reference. Do not execute its case-generating calls.
# Native runif draws the entire initial staircase. The tape retains every
# draw, while the trace records only assigned patients. No R RNG parity claim.
options(warn=2, digits=17)
for (expr in parse('tools/reference_boin_waterfall_workflow.R')) {
  if (is.call(expr) && identical(expr[[1]], as.name('run_case'))) break
  eval(expr, envir=globalenv())
}
stopifnot(is.function(original_subtrial), is.function(enumerated_biviso))
summaries <- traces <- subtrials <- tapes <- list()

run_titration_case <- function(name, prelude, events, budgets=c(4L,2L),
                               cohort_size=3L) {
  current_subtrial <- 0L
  cohort_call <- 0L
  prelude_seen <- FALSE
  local_trace <- local_subtrials <- list()
  uniforms <- numeric()
  record <- function(cell, patients, toxicities, phase, append_uniform=TRUE) {
    local_trace[[length(local_trace)+1L]] <<- data.frame(case=name,
      subtrial=current_subtrial, phase=phase,
      dose_a=(cell-1L)%%2L+1L, dose_b=(cell-1L)%/%2L+1L,
      patients=patients, toxicities=toxicities)
    if (append_uniform) {
      uniforms <<- c(uniforms, rep(.1,toxicities), rep(.9,patients-toxicities))
    }
  }
  native$runif <- function(n, ...) {
    frame <- parent.frame()
    if (isTRUE(frame$titration.first.trial) && !prelude_seen) {
      prelude_seen <<- TRUE
      stopifnot(n==length(prelude), all(prelude %in% 0:1))
      stop_at <- if(any(prelude==1L)) which(prelude==1L)[1L] else n
      uniforms <<- c(uniforms,ifelse(prelude==1L,.1,.9))
      for (j in seq_len(stop_at)) {
        record(frame$dosespace[j],1L,prelude[j],'titration',append_uniform=FALSE)
      }
      return(ifelse(prelude==1L,.1,.9))
    }
    cohort_call <<- cohort_call+1L
    stopifnot(cohort_call<=length(events), events[cohort_call]>=0,
              events[cohort_call]<=n)
    cell <- frame$dosespace[frame$d]
    record(cell,n,events[cohort_call],'cohort')
    c(rep(.1,events[cohort_call]),rep(.9,n-events[cohort_call]))
  }
  native$waterfall.subtrial <- function(...) {
    args <- list(...)
    current_subtrial <<- current_subtrial+1L
    result <- original_subtrial(...)
    selected <- if(result$selectdose==99) 0L else args$dosespace[result$selectdose]
    local_subtrials[[current_subtrial]] <<- data.frame(case=name,
      subtrial=current_subtrial, space=paste(args$dosespace,collapse=';'),
      start=args$startdose, budget=args$ncohort,
      titration=isTRUE(args$titration.first.trial),
      selected_a=if(selected==0L) 0L else (selected-1L)%%2L+1L,
      selected_b=if(selected==0L) 0L else (selected-1L)%/%2L+1L,
      is_escalation=result$is.escalation,
      native_ntotal=result$ntotal, patients=flat(result$npts),
      toxicities=flat(result$ntox), eliminated=flat(result$elimi))
    result
  }
  native$reference_fit <- function(x,w,warn=TRUE) enumerated_biviso(x,w)
  out <- native$get.oc.comb.waterfall(matrix(.3,2,3),target=.3,ncohort=budgets,
    cohortsize=cohort_size,n.earlystop=12,ntrial=1,titration=TRUE)
  trace <- do.call(rbind,local_trace)
  actual_n <- actual_y <- matrix(0L,2,3)
  for (i in seq_len(nrow(trace))) {
    r <- trace$dose_a[i]; c <- trace$dose_b[i]
    actual_n[r,c] <- actual_n[r,c]+trace$patients[i]
    actual_y[r,c] <- actual_y[r,c]+trace$toxicities[i]
  }
  ceiling <- sum(budgets)*cohort_size+if(cohort_size>1L) length(prelude)-1L else 0L
  stopifnot(length(uniforms)<=ceiling)
  uniforms <- c(uniforms,rep(.9,ceiling-length(uniforms)))
  summaries[[length(summaries)+1L]] <<- data.frame(case=name,
    prelude=paste(prelude,collapse=';'), events=paste(events,collapse=';'),
    budgets=paste(budgets,collapse=';'), cohort_size=cohort_size,
    tape_size=ceiling, subtrials=current_subtrial,
    actual_patients=flat(actual_n), actual_toxicities=flat(actual_y),
    actual_total_patients=sum(trace$patients),
    actual_total_toxicities=sum(trace$toxicities),
    native_patients=flat(out$npatients), native_toxicities=flat(out$ntox),
    native_selected=flat(out$selpercent/100))
  traces[[length(traces)+1L]] <<- trace
  subtrials[[length(subtrials)+1L]] <<- do.call(rbind,local_subtrials)
  tapes[[length(tapes)+1L]] <<- data.frame(case=name,index=seq_along(uniforms)-1L,
    uniform=uniforms)
}

run_titration_case('all_safe',c(0L,0L,0L,0L),rep(0L,8))
run_titration_case('middle_dlt',c(0L,1L,0L,0L),rep(0L,8))
run_titration_case('first_dlt_safety',c(1L,0L,0L,0L),c(2L,rep(0L,7)))
run_titration_case('last_dlt',c(0L,0L,0L,1L),c(0L,1L,0L,1L,rep(0L,4)))
run_titration_case('global_budget_after_titration',c(0L,0L,0L,0L),rep(0L,4),
  budgets=c(1L,1L))
run_titration_case('single_patient_cohorts',c(0L,0L,0L,0L),rep(0L,8),
  budgets=c(4L,2L),cohort_size=1L)
write.csv(do.call(rbind,summaries),'tests/fixtures/boin-waterfall-titration.csv',row.names=FALSE)
write.csv(do.call(rbind,traces),'tests/fixtures/boin-waterfall-titration-traces.csv',row.names=FALSE)
write.csv(do.call(rbind,subtrials),'tests/fixtures/boin-waterfall-titration-subtrials.csv',row.names=FALSE)
write.csv(do.call(rbind,tapes),'tests/fixtures/boin-waterfall-titration-tapes.csv',row.names=FALSE)
cat('Generated',length(summaries),'original-R waterfall titration workflows.\n')
