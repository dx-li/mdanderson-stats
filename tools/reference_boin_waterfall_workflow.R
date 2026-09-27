# Original BOIN 2.7.2 waterfall control flow and final selection, using fixed
# cohort outcomes. Only Iso::biviso is replaced: the independent 2x3 oracle
# below exhaustively enumerates all active order equalities. No R installation
# is needed, and these fixtures are not a claim about the desktop executable.
options(warn=2, digits=17)
paths <- file.path('research/raw/BOINComb/R', c('get.boundary.R','get.oc.comb.R'))
stopifnot(unname(tools::md5sum(paths)) ==
  c('0808e5cbdfd5c7795470763d997783c2','342645d339cc8bcb47568f2bd76a1069'))
native <- new.env(parent=globalenv())
for (path in paths) sys.source(path, envir=native)
for (expr in as.list(body(native$get.oc.comb))[-1]) {
  if (is.call(expr) && length(expr)==3 && identical(expr[[1]],as.name('<-')) &&
      is.call(expr[[3]]) && identical(expr[[3]][[1]],as.name('function'))) {
    eval(expr, envir=native)
  }
}

# Every weighted isotonic optimum has an active set of edge equalities.
# Enumerating these sets, pooling connected components by weighted means,
# and retaining the feasible fit with least loss solves this six-cell problem
# independently of the production iterative bivariate implementation.
enumerated_biviso <- function(values, weights) {
  stopifnot(identical(dim(values),c(2L,3L)), all(weights>0),
            all(is.finite(values)), all(is.finite(weights)))
  edges <- rbind(c(1,2),c(3,4),c(5,6),c(1,3),c(3,5),c(2,4),c(4,6))
  best <- NULL
  best_loss <- Inf
  for (mask in 0:(2^nrow(edges)-1)) {
    component <- 1:6
    for (k in seq_len(nrow(edges))) {
      if (bitwAnd(as.integer(mask),bitwShiftL(1L,k-1L))!=0) {
        left <- component[edges[k,1]]
        right <- component[edges[k,2]]
        component[component==right] <- left
      }
    }
    fit <- numeric(6)
    for (group in unique(component)) {
      use <- component==group
      fit[use] <- sum(values[use]*weights[use])/sum(weights[use])
    }
    if (any(fit[edges[,1]]>fit[edges[,2]]+1e-13)) next
    loss <- sum(weights*(values-fit)^2)
    if (loss<best_loss) {
      best <- fit
      best_loss <- loss
    }
  }
  stopifnot(!is.null(best))
  matrix(best,2,3)
}

substitute_fit <- function(expr) {
  if (missing(expr)) return(quote(expr=))
  if (!is.call(expr)) return(expr)
  if (identical(expr[[1]],quote(Iso::biviso))) {
    expr[[1]] <- as.name('reference_fit')
    return(expr)
  }
  as.call(lapply(as.list(expr),substitute_fit))
}
body(native$get.oc.comb.waterfall) <- substitute_fit(body(native$get.oc.comb.waterfall))
original_subtrial <- native$waterfall.subtrial
summaries <- list()
traces <- list()
subtrials <- list()
fits <- list()
flat <- function(x) paste(as.vector(t(x)),collapse=';')

run_case <- function(name, events, budgets=c(4L,2L)) {
  cursor <- 0L
  current_subtrial <- 0L
  local_trace <- list()
  local_subtrials <- list()
  captured_fit <- NULL
  native$runif <- function(n, ...) {
    cursor <<- cursor+1L
    stopifnot(cursor<=length(events),n==3,events[cursor]>=0,events[cursor]<=n)
    frame <- parent.frame()
    cell <- frame$dosespace[frame$d]
    local_trace[[cursor]] <<- data.frame(case=name,subtrial=current_subtrial,
      cohort=cursor,dose_a=(cell-1L)%%2L+1L,dose_b=(cell-1L)%/%2L+1L,
      patients=n,toxicities=events[cursor])
    c(rep(.1,events[cursor]),rep(.9,n-events[cursor]))
  }
  native$waterfall.subtrial <- function(...) {
    args <- list(...)
    current_subtrial <<- current_subtrial+1L
    result <- original_subtrial(...)
    selected <- if(result$selectdose==99) 0L else args$dosespace[result$selectdose]
    local_subtrials[[current_subtrial]] <<- data.frame(case=name,
      subtrial=current_subtrial,space=paste(args$dosespace,collapse=';'),
      start=args$startdose,budget=args$ncohort,
      selected_a=if(selected==0L) 0L else (selected-1L)%%2L+1L,
      selected_b=if(selected==0L) 0L else (selected-1L)%/%2L+1L,
      is_escalation=result$is.escalation,
      patients=flat(result$npts),toxicities=flat(result$ntox),
      eliminated=flat(result$elimi))
    result
  }
  native$reference_fit <- function(x,w,warn=TRUE) {
    fit <- enumerated_biviso(x,w)
    frame <- parent.frame()
    captured_fit <<- data.frame(case=name,input=flat(x),weights=flat(w),
      fitted=flat(fit),patients=flat(frame$npts),toxicities=flat(frame$ntox),
      eliminated=flat(frame$elimi))
    fit
  }
  out <- native$get.oc.comb.waterfall(matrix(.3,2,3),target=.3,ncohort=budgets,
    cohortsize=3,n.earlystop=12,ntrial=1)
  actual <- do.call(rbind,local_trace)
  summaries[[length(summaries)+1L]] <<- data.frame(case=name,
    events=paste(events,collapse=';'),budgets=paste(budgets,collapse=';'),
    consumed_cohorts=cursor,subtrials=current_subtrial,
    native_patients=flat(out$npatients),native_toxicities=flat(out$ntox),
    native_selected=flat(out$selpercent/100),
    actual_total_patients=sum(actual$patients),
    actual_total_toxicities=sum(actual$toxicities),
    native_total_patients=out$totaln,native_total_toxicities=out$totaltox)
  traces[[length(traces)+1L]] <<- actual
  subtrials[[length(subtrials)+1L]] <<- do.call(rbind,local_subtrials)
  fits[[length(fits)+1L]] <<- captured_fit
}
run_case('safe_two_rows',rep(0L,6))
run_case('failed_first_subtrial',c(3L,rep(0L,5)))
run_case('failed_later_subtrial',c(0L,0L,0L,0L,3L,3L))
run_case('special_same_row',c(0L,3L,0L,0L,0L,0L))
run_case('special_same_row_fallback',c(0L,3L,0L,0L,3L,0L))
run_case('earlier_row_without_escalation',rep(1L,6))
write.csv(do.call(rbind,summaries),'tests/fixtures/boin-waterfall-workflows.csv',row.names=FALSE)
write.csv(do.call(rbind,traces),'tests/fixtures/boin-waterfall-workflow-traces.csv',row.names=FALSE)
write.csv(do.call(rbind,subtrials),'tests/fixtures/boin-waterfall-workflow-subtrials.csv',row.names=FALSE)
write.csv(do.call(rbind,fits),'tests/fixtures/boin-waterfall-workflow-fits.csv',row.names=FALSE)
cat('Generated',length(summaries),'original-R workflow cases with exhaustive isotonic fits.\n')
