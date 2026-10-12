# Execute original desktop1.1.0 combination startup expressions, not a Python port.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R OUTPUT_CSV
# The supplied original program is never copied into the distribution.
options(warn=2, digits=17)
args <- commandArgs(TRUE)
stopifnot(length(args)==2,
          unname(tools::md5sum(args[1]))=='64708c960eaccbb1c0eff252a1ed1e88')
native <- new.env(parent=globalenv())
sys.source(args[1], envir=native)
is_head <- function(x, head) is.call(x) && identical(x[[1]], as.name(head))
statement <- function(block, predicate) {
  found <- Filter(predicate, as.list(block)[-1])
  stopifnot(length(found)==1L)
  found[[1]]
}
inner_assignment <- statement(body(native$get.oc.comb_singleMTD), function(x) {
  is_head(x,'<-') && identical(x[[2]], as.name('get.oc.comb.boin'))
})
inner <- eval(inner_assignment[[3]], native)
trial <- statement(body(inner), function(x) {
  is_head(x,'for') && identical(x[[2]], as.name('trial'))
})[[4]]
titration <- statement(trial, function(x) {
  is_head(x,'if') && identical(x[[2]], as.name('titration'))
})
cohorts <- statement(trial, function(x) {
  is_head(x,'for') && identical(x[[2]], as.name('pp'))
})
first_cohort <- as.list(cohorts[[4]])[[2]]
stopifnot(is_head(first_cohort,'if'))
runin_single <- statement(body(inner), function(x) {
  is_head(x,'if') && grepl('RunIn',paste(deparse(x[[2]]),collapse=''))
})
# The waterfall subtrial is nested directly in the contour wrapper.
waterfall_assignment <- statement(body(native$get.oc.comb_MTDcontour), function(x) {
  is_head(x,'<-') && identical(x[[2]],as.name('waterfall.subtrial'))
})
waterfall <- eval(waterfall_assignment[[3]], native)
runin_waterfall <- statement(body(waterfall), function(x) {
  is_head(x,'if') && grepl('RunIn',paste(deparse(x[[2]]),collapse=''))
})
encode <- function(x) paste(as.integer(t(x)),collapse=';')
rows <- list()
for (cap in 0:3) for (cohort in c(1L,2L,3L)) for (dlt in c(FALSE,TRUE)) {
  e <- new.env(parent=baseenv())
  e$p.true <- matrix(0,2,3)
  if (dlt) e$p.true[1,1] <- 1
  e$n <- e$y <- matrix(0,2,3)
  e$d <- c(1L,1L)
  e$titration <- TRUE
  e$titration.cap <- cap
  e$n.titr <- 0L
  e$cohortsize <- cohort
  e$npts <- 24L
  e$ft <- TRUE
  draws <- 0L
  e$runif <- function(n) {draws <<- draws+n;rep(.5,n)}
  e$rbinom <- function(n,size,prob) {
    stopifnot(n==1L,size==1L,all(prob==.5));0L
  }
  eval(titration,e)
  before <- e$n
  endpoint <- e$d
  actual_before <- draws
  eval(first_cohort,e)
  rows[[length(rows)+1L]] <- data.frame(
    cap=cap,cohort_size=cohort,first_dlt=dlt,
    titration_patients=encode(before),endpoint_a=endpoint[1],endpoint_b=endpoint[2],
    first_cohort_patients=encode(e$n),first_cohort_toxicities=encode(e$y),
    titration_draws=actual_before,total_outcome_draws=draws,reported_patients=sum(e$n),
    first_cohort_flag_after=e$ft
  )
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
runin <- list()
for (target in c(.2,.25,.3)) for (enabled in c(FALSE,TRUE)) {
  e <- new.env(parent=baseenv());e$target<-target;e$RunIn<-enabled
  e$b.d <- 11:22;eval(runin_single,e);single<-e$b.d
  e$b.d <- 11:22;eval(runin_waterfall,e);waterfall_values<-e$b.d
  runin[[length(runin)+1L]] <- data.frame(target=target,enabled=enabled,
    single=encode(single),waterfall=encode(waterfall_values))
}
write.csv(do.call(rbind,runin),file.path(dirname(args[2]),'run-in-boundaries.csv'),row.names=FALSE)
