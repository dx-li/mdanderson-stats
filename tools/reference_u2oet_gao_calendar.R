# Independent GAO fixed-coefficient/calendar oracle.
# Probability equations reproduce the separately validated base-R Appendix A
# reference; calendar transitions use explicit tapes and source U2OET rules.
options(warn = 2, digits = 17)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop('usage: Rscript reference_u2oet_gao_calendar.R <fixture-dir>')
dir <- args[[1L]]
tape <- read.csv(file.path(dir,'calendar_tape.csv'))

# Fixed 2x2-dose, binary-efficacy/binary-toxicity GAO coefficient fixture.
d1 <- c(1,2); d2 <- c(1,2)
ei <- matrix(c(-1.2,-.8),1,2); es <- matrix(c(.35,.25),1,2)
ti <- matrix(c(-2.0,-1.6),1,2); ts <- matrix(c(.45,.35),1,2)
lambda_e <- .8; lambda_t <- .7; kappa <- .3; rho <- .25
utility <- matrix(c(.1,0,1,.7),2,2,byrow=TRUE)

marginal <- function(a,b,intercepts,slopes,lambda,kappa) {
  eta <- intercepts + sweep(slopes,2,c(a,b),'*')
  s <- exp(eta[,1])+exp(eta[,2])+kappa*exp(rowSums(eta))
  h <- log1p(lambda*s)/lambda
  c(exp(-h[1]), -expm1(-h[1]))
}
rectangle <- function(ep,tp,rho) {
  if (rho == 0) return(outer(ep,tp))
  ec <- c(0,cumsum(ep)); tc <- c(0,cumsum(tp))
  if (abs(rho) == 1) stop('fixture uses nonsingular association')
  ez <- qnorm(ec); tz <- qnorm(tc); sd <- sqrt(1-rho*rho)
  out <- matrix(0,length(ep),length(tp))
  for (i in seq_along(ep)) for (j in seq_along(tp)) {
    f <- function(x) dnorm(x)*(pnorm((tz[j+1]-rho*x)/sd)-pnorm((tz[j]-rho*x)/sd))
    out[i,j] <- integrate(f,ez[i],ez[i+1],rel.tol=1e-11,abs.tol=1e-13,
                          subdivisions=1000L)$value
  }
  out
}
joint <- array(0,c(2,2,2,2)); utility_mean <- matrix(0,2,2)
pevent <- matrix(0,2,2); ptox <- matrix(0,2,2)
for (i in 1:2) for (j in 1:2) {
  ep <- marginal(d1[i],d2[j],ei,es,lambda_e,kappa)
  tp <- marginal(d1[i],d2[j],ti,ts,lambda_t,kappa)
  joint[i,j,,] <- rectangle(ep,tp,rho)
  utility_mean[i,j] <- sum(joint[i,j,,]*utility)
  pevent[i,j] <- sum(ep[2])
  ptox[i,j] <- sum(tp[2])
}
stopifnot(max(abs(apply(joint,c(1,2),sum)-1)) < 1e-11)
model <- do.call(rbind,lapply(1:2,function(i) data.frame(
  dose1=i-1L,dose2=0:1,efficacy_event=pevent[i,],toxicity_event=ptox[i,],
  mean_utility=utility_mean[i,])))
write.csv(model,file.path(dir,'model.csv'),row.names=FALSE)
marginals <- do.call(rbind,lapply(1:2,function(i) do.call(rbind,lapply(1:2,function(j) {
  ep <- marginal(d1[i],d2[j],ei,es,lambda_e,kappa)
  tp <- marginal(d1[i],d2[j],ti,ts,lambda_t,kappa)
  data.frame(dose1=i-1L,dose2=j-1L,efficacy0=ep[1],efficacy1=ep[2],
             toxicity0=tp[1],toxicity1=tp[2],association=rho)
}))))
write.csv(marginals,file.path(dir,'marginals.csv'),row.names=FALSE)
joint_rows <- do.call(rbind,lapply(1:2,function(i) do.call(rbind,lapply(1:2,function(j)
  do.call(rbind,lapply(1:2,function(e) data.frame(dose1=i-1L,dose2=j-1L,
    efficacy=e-1L,toxicity=0:1,probability=joint[i,j,e,])))))))
write.csv(joint_rows,file.path(dir,'joint.csv'),row.names=FALSE)

# One fixed posterior atom: posterior bad-event probabilities are exactly 0/1
# for each fixed-coefficient cell. This isolates calendar/cohort mechanics.
run_case <- function(tab, scope) {
  maxn <- nrow(tab); cohort <- 2L; surplus <- 2L; top <- 2L
  initial <- if (scope == 'ar2') {
    k <- which.max(as.vector(t(utility_mean))); c((k-1L)%/%2L+1L,(k-1L)%%2L+1L)
  } else c(1L,1L)
  min_e <- if(scope=='stop_no_pair') .95 else .25
  max_t <- if(scope=='stop_no_pair') .01 else .8
  cut_e <- .9; cut_t <- .9
  # For a one-atom posterior, P(inefficacy) / P(overdose) are indicators.
  acceptable <- (pevent >= min_e | cut_e >= 1) & (ptox <= max_t | cut_t >= 1)
  acceptable_flat <- as.vector(t(acceptable))
  utility_flat <- as.vector(t(utility_mean))
  rows <- matrix(numeric(),ncol=5)
  outcomes <- matrix(numeric(),ncol=2)
  allocations <- list(); snapshots <- list(); used <- 0L; early <- FALSE
  last <- c(NA_integer_,NA_integer_); trailing <- 0L; stop_time <- 0
  for (r in seq_len(maxn)) {
    now <- tab$arrival[r]
    complete <- array(0,c(2,2,2,2)); toxonly <- array(0,c(2,2,2)); ignored <- 0L
    for (p in seq_len(used)) {
      tox_seen <- outcomes[p,2] <= now
      e_seen <- outcomes[p,1] <= now
      if (!tox_seen) ignored <- ignored+1L
      else if (!e_seen) toxonly[rows[p,2],rows[p,3],rows[p,5]+1L] <-
        toxonly[rows[p,2],rows[p,3],rows[p,5]+1L]+1
      else complete[rows[p,2],rows[p,3],rows[p,4]+1L,rows[p,5]+1L] <-
        complete[rows[p,2],rows[p,3],rows[p,4]+1L,rows[p,5]+1L]+1
    }
    snapshots[[r]] <- data.frame(time=now,assigned=used,complete=sum(complete),
                                  toxicity_only=sum(toxonly),ignored=ignored)
    if (r == 1L) pair <- initial
    else {
      # Open cohorts continue at the prior pair while it remains acceptable.
      if (trailing %% cohort != 0L && !is.na(last[1]) && acceptable[last[1],last[2]]) {
        pair <- last
      } else {
        candidate <- which(acceptable_flat)
        if (used > 0L && length(candidate)) {
          max1 <- max(rows[seq_len(used),2]); max2 <- max(rows[seq_len(used),3])
          candidate <- candidate[((candidate-1L)%/%2L+1L) <= min(2L,max1+1L) &
                                 ((candidate-1L)%%2L+1L) <= min(2L,max2+1L)]
        }
        if (!length(candidate)) { early <- TRUE; stop_time <- now; break }
        ord <- candidate[order(-utility_flat[candidate],candidate,method='radix')]
        best <- ord[1]
        nbest <- if (used) sum(rows[seq_len(used),2]==((best-1L)%/%2L+1L) &
                              rows[seq_len(used),3]==((best-1L)%%2L+1L)) else 0
        other_counts <- if (length(ord)>1L) vapply(ord[-1],function(k) {
          if (!used) return(0)
          sum(rows[seq_len(used),2]==((k-1L)%/%2L+1L) &
              rows[seq_len(used),3]==((k-1L)%%2L+1L))
        },numeric(1)) else numeric()
        weights <- numeric(4L)
        use_ar <- length(ord)>1L && nbest-max(other_counts)>=surplus
        if (use_ar) {
          chosen <- ord[seq_len(min(top,length(ord)))]
          w <- utility_flat[chosen]; w <- w/max(w); w <- w/sum(w)
          weights[chosen] <- w
        } else weights[best] <- 1
        cum_alloc <- cumsum(weights)
        cum_alloc[4L] <- 1
        pair_index <- which(tab$u_alloc[r] < cum_alloc)[1L]
        if (is.na(pair_index)) pair_index <- 4L
        pair <- c((pair_index-1L)%/%2L+1L,(pair_index-1L)%%2L+1L)
      }
    }
    cum <- cumsum(as.vector(t(joint[pair[1],pair[2],,])))
    cum[4L] <- 1
    cell <- which(tab$u_cell[r] < cum)[1L]
    if (is.na(cell)) cell <- 4L
    e <- (cell-1L)%/%2L; t <- (cell-1L)%%2L
    de <- 6*tab$u_eff_delay[r]; dt <- 6*tab$u_tox_delay[r]
    rows <- rbind(rows,c(r,pair[1],pair[2],e,t))
    outcomes <- rbind(outcomes,c(now+de,now+dt))
    used <- used+1L; stop_time <- now
    trailing <- if (!is.na(last[1]) && all(last==pair)) trailing+1L else 1L
    last <- pair
  }
  analysis_time <- if (used) max(stop_time,max(outcomes[seq_len(used),])) else stop_time
  final_complete <- 0L; final_toxonly <- 0L; final_ignored <- 0L
  for (p in seq_len(used)) {
    tox_seen <- outcomes[p,2] <= analysis_time; e_seen <- outcomes[p,1] <= analysis_time
    if (!tox_seen) final_ignored <- final_ignored+1L
    else if (!e_seen) final_toxonly <- final_toxonly+1L
    else final_complete <- final_complete+1L
  }
  final_snapshot <- data.frame(time=analysis_time,assigned=used,complete=final_complete,
                               toxicity_only=final_toxonly,ignored=final_ignored)
  # Final data snapshot after follow-up; fitted coefficient atom is unchanged.
  finalmask <- acceptable
  if (scope == 'tried') {
    tried <- matrix(FALSE,2,2)
    if (used) for (p in seq_len(used)) tried[rows[p,2],rows[p,3]] <- TRUE
    finalmask <- finalmask & tried
  }
  selected <- if (early || !any(finalmask)) NA_integer_ else {
    ix <- which(as.vector(t(finalmask)))
    ix[which.max(utility_flat[ix])]
  }
  list(patients=rows, outcomes=outcomes, snapshots=do.call(rbind,snapshots),
       early=early, stop_time=stop_time, analysis_time=analysis_time,
       final_snapshot=final_snapshot,selected=selected, utility=utility_mean)
}

all_runs <- list(); patients <- list(); snapshots <- list(); final_snapshots <- list()
for (case in unique(tape$case)) {
  tab <- tape[tape$case==case,]; scope <- if(case=='scope_tried') 'tried' else if(case=='ar2') 'ar2' else 'acceptable'
  rr <- run_case(tab,if(case=='stop_no_pair') 'stop_no_pair' else scope)
  all_runs[[case]] <- data.frame(case=case,early=rr$early,stop_time=rr$stop_time,
      analysis_time=rr$analysis_time,selected=rr$selected)
  if (nrow(rr$patients)) {
    patients[[case]] <- data.frame(case=case,patient=rr$patients[,1],dose1=rr$patients[,2],
      dose2=rr$patients[,3],efficacy=rr$patients[,4],toxicity=rr$patients[,5],
      efficacy_time=rr$outcomes[,1],toxicity_time=rr$outcomes[,2])
  }
  snapshots[[case]] <- cbind(case=case,rr$snapshots)
  final_snapshots[[case]] <- cbind(case=case,rr$final_snapshot)
}
write.csv(do.call(rbind,all_runs),file.path(dir,'calendar_summary.csv'),row.names=FALSE)
write.csv(do.call(rbind,patients),file.path(dir,'patients.csv'),row.names=FALSE)
write.csv(do.call(rbind,snapshots),file.path(dir,'snapshots.csv'),row.names=FALSE)
write.csv(do.call(rbind,final_snapshots),file.path(dir,'final_snapshots.csv'),row.names=FALSE)
