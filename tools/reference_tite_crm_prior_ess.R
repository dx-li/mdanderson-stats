options(warn=2, digits=17)
args <- commandArgs(trailingOnly=TRUE)
repo_root <- if (length(args) >= 1L) normalizePath(args[[1]], mustWork=TRUE) else normalizePath(".", mustWork=TRUE)
out_dir <- if (length(args) >= 2L) args[[2]] else file.path(repo_root,"tests","fixtures")
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
# Assert exact local copies of the two pinned upstream sources.
expected <- c(
  "research/raw/dfcrm/dfcrm.R"="c955ae921232695c6201515bcc1a7720e4c9703f",
  "research/raw/BayesESS/source/R/internal.R"="813a734264059ff7e47eebf51eda02351818a8ce")
for (f in names(expected)) {
  source_path <- file.path(repo_root, f)
  actual <- system2("git",c("-C",shQuote(repo_root),"hash-object",shQuote(source_path)),stdout=TRUE)
  stopifnot(identical(unname(actual),unname(expected[f])))
}
native <- new.env(parent=globalenv())
sys.source(file.path(repo_root,"research/raw/dfcrm/dfcrm.R"),envir=native)
bayesess <- new.env(parent=globalenv())
sys.source(file.path(repo_root,"research/raw/BayesESS/source/R/internal.R"),envir=bayesess)
# Extract, do not hand-copy, BayesESS's nested TITE curvature function.
get_assignment <- function(fun,name) {
  exprs <- as.list(body(fun))
  hit <- which(vapply(exprs,function(x) is.call(x) && identical(x[[1]],as.name("<-")) &&
                        identical(x[[2]],as.name(name)),logical(1)))
  stopifnot(length(hit)==1L)
  exprs[[hit]]
}
eval(get_assignment(bayesess$essCRM,"getDiff"),envir=bayesess)
prior <- c(.05,.10,.20,.35)
truth <- c(.05,.15,.30,.45)
target <- .20
obswin <- 10
rate <- 2
scale <- sqrt(1.34)
n <- 8L
N_total <- n
# The outcome uniform tape induces toxicities at subjects 2, 5, and 8 for every dose.
outcome_uniform <- c(.9,.01,.9,.9,.01,.9,.9,.01)
delay_fraction <- c(NA,.2,NA,NA,.7,NA,NA,.9)
# Fixed schedule from obswin/rate, plus one explicit Poisson-arrival realization.
scenarios <- list(
  fixed=list(accrual="fixed",interarrivals=rep(obswin/rate,n)),
  poisson=list(accrual="poisson",interarrivals=c(5,2,3,6,2,7,2,9)))
paths <- ledgers <- curvature <- summaries <- list()
for (case in names(scenarios)) {
  spec <- scenarios[[case]]
  tape_state <- new.env(parent=emptyenv())
  tape_state$arrival_pos <- tape_state$event_pos <- tape_state$delay_pos <- tape_state$next_arrival <- 0L
  native$rbinom <- function(n,size,prob) {
    stopifnot(n==1L,size==1L,tape_state$arrival_pos<N_total)
    tape_state$arrival_pos <- tape_state$arrival_pos+1L
    as.integer(outcome_uniform[tape_state$arrival_pos] < prob)
  }
  native$runif <- function(n,min=0,max=1) {
    stopifnot(n==1L,tape_state$event_pos<3L)
    tape_state$event_pos <- tape_state$event_pos+1L
    tape_state$delay_pos <- tape_state$delay_pos+1L
    frac <- c(.2,.7,.9)[tape_state$delay_pos]
    min+(max-min)*frac
  }
  native$rexp <- function(n,rate) {
    stopifnot(n==1L)
    tape_state$next_arrival <- tape_state$next_arrival+1L
    spec$interarrivals[tape_state$next_arrival]
  }
  result <- native$onetite(truth,prior,target,n=n,x0=1,restrict=TRUE,
    obswin=obswin,tgrp=obswin,rate=rate,accrual=spec$accrual,surv="uniform",
    scheme="linear",method="bayes",model="empiric",scale=scale,seed=41)
  stopifnot(tape_state$arrival_pos==n,tape_state$event_pos==3L,
            tape_state$delay_pos==3L,
            tape_state$next_arrival==if (spec$accrual=="poisson") n else 0L)
  case_path <- data.frame(case=case,patient=seq_len(n),
    outcome_uniform=outcome_uniform,arrival=result$arrival,dose=result$level,
    tox=result$tox,toxicity_delay=result$toxicity.time,
    event_study_time=result$toxicity.study.time,beta_before=result$beta.hat)
  paths[[case]] <- case_path
  analysis <- max(result$arrival)+3
  observed <- as.integer(result$tox==1 & result$toxicity.study.time<=analysis)
  followup <- pmin(pmax(pmin(analysis,result$toxicity.study.time)-result$arrival,0),obswin)
  valid_weight <- ifelse(observed==1,1,followup/obswin)
  arrival_weight <- pmin(result$arrival,obswin)/obswin
  ledgers[[case]] <- data.frame(case=case,patient=seq_len(n),arrival=result$arrival,
    dose=result$level,eventual_tox=result$tox,event_study_time=result$toxicity.study.time,
    analysis_time=analysis,observed_tox=observed,followup=followup,
    valid_linear_weight=valid_weight,bayesess_arrival_weight=arrival_weight)
  d <- prior[result$level]
  curvature[[case]] <- data.frame(case=case,patient=seq_len(n),d=d,
    eventual_tox=result$tox,observed_tox=observed,
    bayesess_arrival_weight=arrival_weight,
    source_getDiff=sapply(seq_len(n),function(i) bayesess$getDiff(d[i],arrival_weight[i],result$tox[i])),
    explicit_followup_weight=valid_weight,
    observed_followup_getDiff=sapply(seq_len(n),function(i) bayesess$getDiff(d[i],valid_weight[i],observed[i])))
  summaries[[case]] <- data.frame(case=case,toxicity_uniforms=tape_state$arrival_pos,
    event_time_uniforms=tape_state$event_pos,arrival_intervals=tape_state$next_arrival,
    final_mtd=result$MTD,final_beta_mean=result$final.est,tox_count=sum(result$tox),
    source_curvature_sum=sum(curvature[[case]]$source_getDiff),
    observed_followup_curvature_sum=sum(curvature[[case]]$observed_followup_getDiff))
}
write.csv(do.call(rbind,paths),file.path(out_dir,"tite-crm-prior-ess-native-onetite-paths.csv"),row.names=FALSE)
write.csv(do.call(rbind,ledgers),file.path(out_dir,"tite-crm-prior-ess-followup-ledger.csv"),row.names=FALSE)
write.csv(do.call(rbind,curvature),file.path(out_dir,"tite-crm-prior-ess-curvature.csv"),row.names=FALSE)
write.csv(do.call(rbind,summaries),file.path(out_dir,"tite-crm-prior-ess-native-onetite-summary.csv"),row.names=FALSE)
# A source-formula sign case, kept separate from any followup-history approximation.
write.csv(data.frame(case="positive_nonDLT_curvature",d=.8,w=.01,y=0,
  getDiff=bayesess$getDiff(.8,.01,0)),file.path(out_dir,"tite-crm-prior-ess-curvature-sign-example.csv"),row.names=FALSE)
cat("final MTDs",sapply(summaries,function(x)x$final_mtd),"\n")
cat("source curvature sums",sapply(summaries,function(x)x$source_curvature_sum),"\n")
cat("observed-followup curvature sums",sapply(summaries,function(x)x$observed_followup_curvature_sum),"\n")
