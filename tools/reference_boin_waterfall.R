# Deterministic conduct references from original BOIN 2.7.2 nested routines.
# Source-only: no BOIN/Iso installation, random simulation or source redistribution.
# The original routines are loaded unchanged; a local runif supplies specified
# cohort outcomes and records the actual dose at which they are requested.
options(warn=2, digits=17)
source_dir <- 'research/raw/BOINComb/R'
paths <- file.path(source_dir, c('get.boundary.R', 'get.oc.comb.R'))
stopifnot(unname(tools::md5sum(paths)) ==
  c('0808e5cbdfd5c7795470763d997783c2', '342645d339cc8bcb47568f2bd76a1069'))
native <- new.env(parent=globalenv())
for (path in paths) sys.source(path, envir=native)
# Extract the nested function declarations without running the full simulator
# or its final Iso-dependent reporting path. Their arithmetic is unchanged.
for (expr in as.list(body(native$get.oc.comb))[-1]) {
  if (is.call(expr) && length(expr)==3 && identical(expr[[1]], as.name('<-')) &&
      is.call(expr[[3]]) && identical(expr[[3]][[1]], as.name('function'))) {
    eval(expr, envir=native)
  }
}
stopifnot(is.function(native$waterfall.subtrial),
          is.function(native$waterfall.subtrial.mtd))

summary_rows <- list()
trace_rows <- list()
run_case <- function(name, events, earlystop=12, extrasafe=FALSE,
                     start=1, bound=FALSE) {
  cursor <- 0L
  trace <- list()
  probability <- matrix(.3, 2, 3)
  space <- c(1L, 2L, 4L, 6L)
  native$runif <- function(n, ...) {
    cursor <<- cursor+1L
    stopifnot(cursor<=length(events), n==3, events[cursor]>=0, events[cursor]<=n)
    frame <- parent.frame()
    cell <- frame$dosespace[frame$d]
    trace[[cursor]] <<- data.frame(case=name, cohort=cursor,
      dose_a=(cell-1L)%%2L+1L, dose_b=(cell-1L)%/%2L+1L,
      patients=n, toxicities=events[cursor])
    c(rep(.1,events[cursor]),rep(.9,n-events[cursor]))
  }
  boundary <- native$get.boundary(.3, ncohort=150, cohortsize=1,
    cutoff.eli=.95, extrasafe=extrasafe)$boundary_tab
  out <- native$waterfall.subtrial(.3, p.true=probability, dosespace=space,
    npts=matrix(0,2,3), ntox=matrix(0,2,3), elimi=matrix(0,2,3),
    ncohort=length(events), cohortsize=3, n.earlystop=earlystop,
    startdose=start, extrasafe=extrasafe, totaln=0, temp=boundary, boundMTD=bound)
  flat <- function(x) paste(as.vector(t(x)),collapse=';')
  selected <- if(out$selectdose==99) 0L else space[out$selectdose]
  summary_rows[[length(summary_rows)+1L]] <<- data.frame(
    case=name, events=paste(events,collapse=';'), earlystop=earlystop,
    extrasafe=extrasafe, start=start, bound=bound, cohorts=cursor,
    patients=flat(out$npts), toxicities=flat(out$ntox), eliminated=flat(out$elimi),
    selected_a=if(selected==0L) 0L else (selected-1L)%%2L+1L,
    selected_b=if(selected==0L) 0L else (selected-1L)%/%2L+1L,
    is_escalation=out$is.escalation, total_patients=out$totaln,
    total_toxicities=out$totaltox)
  trace_rows[[length(trace_rows)+1L]] <<- do.call(rbind,trace)
}
run_case('safe_staircase', c(0,0,0,0))
run_case('lowest_dose_safety', c(3,0,0,0))
run_case('precision_at_retained_dose', c(0,1,1,1,1,1), earlystop=9)
run_case('precision_after_downward_move', c(0,1,0,2,0,0), earlystop=6)
run_case('subtrial_extra_safe', c(2,0,0,0), extrasafe=TRUE)
run_case('start_interior_then_deescalate', c(2,0,0,1,1,1), start=3)
dir.create('tests/fixtures', showWarnings=FALSE, recursive=TRUE)
write.csv(do.call(rbind,summary_rows), 'tests/fixtures/boin-waterfall-subtrials.csv', row.names=FALSE)
write.csv(do.call(rbind,trace_rows), 'tests/fixtures/boin-waterfall-traces.csv', row.names=FALSE)
cat('Generated',length(summary_rows),'original-R waterfall subtrial cases.\n')
