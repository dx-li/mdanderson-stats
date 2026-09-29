# Independent base-R oracle for randomized BOP2-DC with two binary endpoints.
# Exhaustive 4-category outcome paths; this does not call Python code.
# Usage: Rscript reference_bop2_dc_randomized_paired_calibration.R [prefix]
options(digits = 17)
args <- commandArgs(trailingOnly = TRUE)
prefix <- if (length(args)) args[[1]] else "tests/fixtures/bop2-dc-randomized-paired/"
settings <- read.csv(paste0(prefix, "settings.csv"), colClasses="character", check.names=FALSE)
truths <- read.csv(paste0(prefix, "truths.csv"), colClasses="character", check.names=FALSE)
nums <- function(x) as.numeric(strsplit(x, ";", fixed=TRUE)[[1]])
ints <- function(x) as.integer(strsplit(x, ";", fixed=TRUE)[[1]])
flag <- function(x) as.integer(x) == 1L
roundoff_tied <- function(a,b,N) a==b || (max(abs(a),abs(b))>0 && abs(a-b)<=64*.Machine$double.eps*N*max(abs(a),abs(b)))
within_limit <- function(x,limit,N) x<=limit || roundoff_tied(x,limit,N)
parse_pair_grid <- function(x) {
  rows <- strsplit(x, ";", fixed=TRUE)[[1]]
  ans <- do.call(rbind,lapply(rows,function(row) as.numeric(strsplit(row, ",",fixed=TRUE)[[1]])))
  if (ncol(ans)!=2) stop("each calibration-grid row must have two endpoint controls")
  ans
}

# Integer-shape beta CDF from its finite binomial-sum identity.
beta_cdf_int <- function(x,a,b) {
  if (x <= 0) return(0); if (x >= 1) return(1)
  m <- a+b-1
  sum(vapply(a:m,function(k) choose(m,k)*x^k*(1-x)^(m-k),numeric(1)))
}
# Memoized scalar quadrature: shape/margin tuples recur across exhaustive paths.
tail_cache <- new.env(parent=emptyenv())
beta_diff <- function(ac,bc,at,bt,delta) {
  key <- paste(format(c(ac,bc,at,bt,delta),digits=16,scientific=TRUE),collapse=",")
  if (exists(key,envir=tail_cache,inherits=FALSE)) return(get(key,envir=tail_cache))
  if (delta==0 && ac==at && bc==bt) ans <- .5
  else if (delta>=1) ans <- 0
  else if (delta<=-1) ans <- 1
  else ans <- integrate(function(t) vapply(t,function(z) {
    if (z <= delta) return(0)
    exp(dbeta(z,at,bt,log=TRUE))*beta_cdf_int(z-delta,ac,bc)
  },numeric(1)),lower=max(0,delta),upper=1,rel.tol=2e-12,abs.tol=2e-14,
    subdivisions=1000L,stop.on.error=TRUE)$value
  assign(key,ans,envir=tail_cache); ans
}
obf <- function(lambda,n,N) 2*pnorm(qnorm((1+lambda)/2)/sqrt(n/N))-1

# Four cells: (both, first only, second only, neither). The marginal success
# indicators are cells {both, first-only} and {both, second-only}.
tail_pair <- function(cc,ct,pc,pt,endpoint,margins) {
  out <- matrix(NA_real_,2,2)
  sets <- list(c(1L,2L),c(1L,3L))
  for (j in 1:2) {
    ix <- sets[[j]]
    ac <- sum(pc[ix])+sum(cc[ix]); bc <- sum(pc[-ix])+sum(cc[-ix])
    at <- sum(pt[ix])+sum(ct[ix]); bt <- sum(pt[-ix])+sum(ct[-ix])
    for (k in 1:2) {
      delta <- margins[j,k]
      if (endpoint=="efficacy_toxicity" && j==2L) {
        # Raw margin is treatment-minus-control; safety benefit is its negative.
        aa <- ac; bb <- bc; ac <- at; bc <- bt; at <- aa; bt <- bb; delta <- -delta
      }
      out[j,k] <- beta_diff(ac,bc,at,bt,delta)
      if (endpoint=="efficacy_toxicity" && j==2L) {
        aa <- ac; bb <- bc; ac <- at; bc <- bt; at <- aa; bt <- bb
      }
    }
  }
  out
}
classify_interim <- function(endpoint,no,graduate) {
  if (endpoint=="multiple_efficacy") {
    if (all(no)) return("stop_no_go")
    if (any(graduate)) return("graduate")
  } else {
    if (any(no)) return("stop_no_go")
    if (all(graduate)) return("graduate")
  }
  "continue"
}
classify_final <- function(endpoint,go,no) {
  if (endpoint=="multiple_efficacy") {
    if (any(go)) return("final_go")
    if (all(no)) return("final_no_go")
  } else {
    if (all(go)) return("final_go")
    if (any(no)) return("final_no_go")
  }
  "final_consider"
}

candidate_out <- list(); path_out <- list(); monitor_out <- list(); oc_out <- list(); selected_out <- list()
for (si in seq_len(nrow(settings))) {
  s <- settings[si,,drop=FALSE]; id <- s$config_id[[1]]; endpoint <- s$endpoint[[1]]
  N <- as.integer(s$max_subjects); looks <- ints(s$looks)
  arms <- as.integer(strsplit(s$arm_assignments[[1]],"",fixed=TRUE)[[1]])
  pc <- nums(s$control_prior); pt <- nums(s$treatment_prior)
  lrv <- nums(s$lrv); cmv <- nums(s$cmv)
  if (N>6 || length(arms)!=N || any(!arms %in% 0:1) || !any(arms==0) || !any(arms==1) ||
      tail(looks,1)!=N || any(diff(looks)<=0) || length(pc)!=4 || length(pt)!=4 ||
      any(c(pc,pt)<=0) || any(c(pc,pt)!=round(c(pc,pt))) || length(lrv)!=2 || length(cmv)!=2)
    stop(paste("invalid setting",id))
  if (endpoint=="multiple_efficacy" && any(lrv>=cmv)) stop("efficacy LRV must be below CMV")
  if (endpoint=="efficacy_toxicity" && (lrv[1]>=cmv[1] || lrv[2]<=cmv[2]))
    stop("efficacy LRV<CMV and raw-toxicity LRV>CMV are required")
  margins <- rbind(c(lrv[1],cmv[1]),c(lrv[2],cmv[2]))
  tr <- truths[truths$config_id==id,,drop=FALSE]
  if (!setequal(tr$scenario,c("futile","effective"))) stop("need explicit futile/effective truths")
  for (r in seq_len(nrow(tr))) {
    for (nm in c("control_probability","treatment_probability")) {
      q <- nums(tr[[nm]][r]); if(length(q)!=4 || any(q<0) || abs(sum(q)-1)>1e-14) stop("bad joint truth")
    }
  }
  eff <- tr[tr$scenario=="effective",,drop=FALSE]
  qm <- function(q) c(q[1]+q[2],q[1]+q[3])
  d_eff <- qm(nums(eff$treatment_probability[1]))-qm(nums(eff$control_probability[1]))
  if (endpoint=="multiple_efficacy" && max(d_eff-cmv) < -1e-12) stop("effective truth misses both CMVs")
  if (endpoint=="efficacy_toxicity" && (d_eff[1]<cmv[1]-1e-12 || d_eff[2]>cmv[2]+1e-12))
    stop("effective efficacy/toxicity truth fails source-go conditions")
  gl <- list(parse_pair_grid(s$lambda_lrv_grid),parse_pair_grid(s$lambda_cmv_grid),
             parse_pair_grid(s$gamma_lrv_grid),parse_pair_grid(s$gamma_cmv_grid))
  if (any(unlist(gl[1:2])<0 | unlist(gl[1:2])>1) || any(unlist(gl[3:4])<0))
    stop("cutoff grids require lambda in [0,1] and nonnegative gamma")
  if (any(as.numeric(unlist(s[c("false_go_limit","false_no_go_limit","false_consider_limit")],use.names=FALSE))<0 |
          as.numeric(unlist(s[c("false_go_limit","false_no_go_limit","false_consider_limit")],use.names=FALSE))>1))
    stop("false-decision limits must lie in [0,1]")
  # Product order of the four grid rows; endpoint 2 controls are independently supplied.
  grid_size <- prod(vapply(gl,nrow,numeric(1)))
  if (grid_size>64 || 4^N>4096) stop("oracle fixture exceeds small exhaustive-work limit")
  axes <- lapply(gl,function(x) seq_len(nrow(x)))
  ixgrid <- expand.grid(rev(axes),KEEP.OUT.ATTRS=FALSE)
  ixgrid <- ixgrid[,rev(seq_along(axes)),drop=FALSE]
  names(ixgrid) <- c("il","ic","igL","igC")
  ixgrid$candidate_index <- seq_len(nrow(ixgrid))-1L
  # Keep each candidate as four independent length-two endpoint vectors.
  grid <- lapply(seq_len(nrow(ixgrid)),function(i) list(
    lambda_lrv=gl[[1]][ixgrid$il[i],],lambda_cmv=gl[[2]][ixgrid$ic[i],],
    gamma_lrv=gl[[3]][ixgrid$igL[i],],gamma_cmv=gl[[4]][ixgrid$igC[i],]))
  catpaths <- expand.grid(rep(list(1:4),N),KEEP.OUT.ATTRS=FALSE)
  catpaths$path <- apply(catpaths,1,paste0,collapse="")
  catpaths$path_id <- seq_len(nrow(catpaths))-1L
  candmetrics <- vector("list",length(grid))
  for (ci in seq_along(grid)) {
    g <- grid[[ci]]; decisions <- vector("list",nrow(catpaths))
    for (pi in seq_len(nrow(catpaths))) {
      cells <- as.integer(strsplit(catpaths$path[pi],"",fixed=TRUE)[[1]])
      cc <- ct <- integer(4); terminal <- "continue"; terminal_n <- N
      for (n in looks) {
        cc <- ct <- integer(4)
        for (i in seq_len(n)) if(arms[i]==0L) cc[cells[i]]<-cc[cells[i]]+1L else ct[cells[i]]<-ct[cells[i]]+1L
        post <- tail_pair(cc,ct,pc,pt,endpoint,margins)
        if(n<N) {
          no <- graduate <- logical(2)
          for(j in 1:2) {
            no[j] <- post[j,1]<g$lambda_lrv[j]*(n/N)^g$gamma_lrv[j] &&
                     post[j,2]<g$lambda_cmv[j]*(n/N)^g$gamma_cmv[j]
            graduate[j] <- flag(s$graduate_at_interim) &&
              post[j,1]>obf(g$lambda_lrv[j],n,N) && post[j,2]>obf(g$lambda_cmv[j],n,N)
            if(no[j] && graduate[j]) stop("overlapping no-go and graduation criteria")
          }
          action <- classify_interim(endpoint,no,graduate)
        } else {
          go <- no <- logical(2)
          for(j in 1:2) {
            go[j] <- post[j,1]>g$lambda_lrv[j] && post[j,2]>g$lambda_cmv[j]
            no[j] <- post[j,1]<g$lambda_lrv[j] && post[j,2]<g$lambda_cmv[j]
          }
          action <- classify_final(endpoint,go,no)
        }
        monitor_out[[length(monitor_out)+1L]] <- data.frame(config_id=id,candidate_index=ixgrid$candidate_index[ci],
          path_id=catpaths$path_id[pi],look=n,control_n=sum(cc),treatment_n=sum(ct),
          control_success1=cc[1]+cc[2],control_success2=cc[1]+cc[3],
          treatment_success1=ct[1]+ct[2],treatment_success2=ct[1]+ct[3],
          tail1_lrv=post[1,1],tail1_cmv=post[1,2],tail2_lrv=post[2,1],tail2_cmv=post[2,2],decision=action)
        if(action!="continue") {terminal<-action;terminal_n<-n;break}
      }
      decisions[[pi]] <- list(action=terminal,n=terminal_n,cells=cells)
      path_out[[length(path_out)+1L]] <- data.frame(config_id=id,candidate_index=ixgrid$candidate_index[ci],
        path_id=catpaths$path_id[pi],path=catpaths$path[pi],terminal_decision=terminal,terminal_look=terminal_n)
    }
    bytruth <- list()
    for(scenario in c("futile","effective")) {
      q <- tr[tr$scenario==scenario,,drop=FALSE]
      pC <- nums(q$control_probability[1]); pT <- nums(q$treatment_probability[1])
      stopm <- setNames(rep(0,length(looks)),looks); gradm <- stopm; samplem <- stopm
      fg <- fc <- fn <- en <- 0
      for(d in decisions) {
        prob <- prod(vapply(seq_len(N),function(i) if(arms[i]==0L) pC[d$cells[i]] else pT[d$cells[i]],numeric(1)))
        key <- as.character(d$n); samplem[key] <- samplem[key]+prob; en <- en+prob*d$n
        if(d$action=="stop_no_go") stopm[key] <- stopm[key]+prob
        else if(d$action=="graduate") gradm[key] <- gradm[key]+prob
        else if(d$action=="final_go") fg <- fg+prob
        else if(d$action=="final_consider") fc <- fc+prob
        else if(d$action=="final_no_go") fn <- fn+prob
      }
      bytruth[[scenario]] <- list(stop=stopm,graduate=gradm,sample=samplem,go=fg,consider=fc,no=fn,en=en)
    }
    f <- bytruth$futile; e <- bytruth$effective
    fgr <- sum(f$graduate)+f$go; fngr <- sum(e$stop)+e$no; cgr <- sum(e$graduate)+e$go
    fcr <- max(f$consider,e$consider)
    feasible <- within_limit(fgr,as.numeric(s$false_go_limit),N) &&
      within_limit(fngr,as.numeric(s$false_no_go_limit),N) &&
      within_limit(fcr,as.numeric(s$false_consider_limit),N)
    candmetrics[[ci]] <- data.frame(FGR=fgr,FNGR=fngr,CGR=cgr,FCR=fcr,EN_futile=f$en,
      EN_effective=e$en,feasible=feasible)
    candidate_out[[length(candidate_out)+1L]] <- data.frame(config_id=id,candidate_index=ixgrid$candidate_index[ci],
      lambda_lrv_1=g$lambda_lrv[1],lambda_cmv_1=g$lambda_cmv[1],lambda_lrv_2=g$lambda_lrv[2],lambda_cmv_2=g$lambda_cmv[2],
      gamma_lrv_1=g$gamma_lrv[1],gamma_cmv_1=g$gamma_cmv[1],gamma_lrv_2=g$gamma_lrv[2],gamma_cmv_2=g$gamma_cmv[2],candmetrics[[ci]])
    for(scenario in c("futile","effective")) {
      z <- bytruth[[scenario]]
      for(n in looks) {
        samplep <- z$sample[as.character(n)]
        oc_out[[length(oc_out)+1L]] <- data.frame(config_id=id,candidate_index=ixgrid$candidate_index[ci],
          scenario=scenario,look=n,stop_no_go=z$stop[as.character(n)],graduate=z$graduate[as.character(n)],
          final_go=if(n==N) z$go else 0,final_consider=if(n==N) z$consider else 0,
          final_no_go=if(n==N) z$no else 0,sample_size_probability=samplep,expected_sample_size=z$en)
      }
    }
  }
  mm <- do.call(rbind,candmetrics); ok <- which(mm$feasible)
  for(objective in c("cgr","ess_futile")) {
    if(!length(ok)) {
      selected_out[[length(selected_out)+1L]] <- data.frame(config_id=id,objective=objective,selected_index=NA_integer_,
        FGR=NA_real_,FNGR=NA_real_,CGR=NA_real_,FCR=NA_real_,EN_futile=NA_real_,EN_effective=NA_real_)
      next
    }
    rel_equal <- function(a,b) roundoff_tied(a,b,N)
    primary <- if(objective=="cgr") mm$CGR[ok] else mm$EN_futile[ok]
    pbest <- if(objective=="cgr") max(primary) else min(primary)
    pkeep <- vapply(primary,function(x) rel_equal(x,pbest),logical(1))
    finalists <- ok[pkeep]
    secondary <- if(objective=="cgr") -mm$EN_futile[finalists] else mm$CGR[finalists]
    sbest <- max(secondary); sk <- vapply(secondary,function(x) rel_equal(x,sbest),logical(1))
    winner <- finalists[which(sk)[1]]; z <- mm[winner,,drop=FALSE]
    selected_out[[length(selected_out)+1L]] <- data.frame(config_id=id,objective=objective,selected_index=winner-1L,
      FGR=z$FGR,FNGR=z$FNGR,CGR=z$CGR,FCR=z$FCR,EN_futile=z$EN_futile,EN_effective=z$EN_effective)
  }
}
write.csv(do.call(rbind,candidate_out),paste0(prefix,"candidate-metrics.csv"),row.names=FALSE)
write.csv(do.call(rbind,selected_out),paste0(prefix,"selected.csv"),row.names=FALSE)
write.csv(do.call(rbind,path_out),paste0(prefix,"path-decisions.csv"),row.names=FALSE)
write.csv(do.call(rbind,monitor_out),paste0(prefix,"monitor-paths.csv"),row.names=FALSE)
write.csv(do.call(rbind,oc_out),paste0(prefix,"operating-characteristics.csv"),row.names=FALSE)
