# Independent base-R references for Multc Lean / Multc99 Phase IIa.
# No third-party packages. Run from the repository root with Rscript.
options(digits=17)

# Direct integration over the HISTORICAL beta variable, separately for each
# tail, independent of Python's beta-difference implementation.
tail <- function(a,b,ha,hb,delta,lower=TRUE) {
  cuts <- sort(unique(c(0,1,max(0,min(1,-delta)),max(0,min(1,1-delta)))))
  sum(vapply(seq_len(length(cuts)-1),function(j)
    integrate(function(s) dbeta(s,ha,hb)*
      pbeta(pmax(0,pmin(1,s+delta)),a,b,lower.tail=lower),
      cuts[j],cuts[j+1],rel.tol=1e-10,abs.tol=1e-12,
      subdivisions=300)$value,numeric(1)))
}
rows <- list()
for(v in list(c(.6,1.4,30,70,0,0,0),c(.6,1.4,30,70,0,5,0),
              c(.6,1.4,30,70,0,6,0),c(.6,1.4,30,70,.1,9,2),
              c(.5,1.5,20,60,-.08,8,3),c(.8,1.2,3,7,.15,7,2),
              c(.8,1.2,3,7,-.2,7,2),c(.8,1.2,.7,2.3,0,5,1))) {
  a<-v[1]; b<-v[2]; ha<-v[3]; hb<-v[4]; delta<-v[5]; n<-v[6]; y<-v[7]
  rows[[length(rows)+1]] <- data.frame(a=a,b=b,ha=ha,hb=hb,delta=delta,n=n,y=y,
    below=tail(a+y,b+n-y,ha,hb,delta),
    above=tail(a+y,b+n-y,ha,hb,delta,FALSE))
}
write.csv(do.call(rbind,rows),'tests/fixtures/multc-beta-tails.csv',row.names=FALSE)

# Native comparison note: uniform experimental priors; fixed historical .5.
# The toxicity tail at y is the response tail at n-y by symmetry.
rows <- list()
for(n in 1:15) {
  ys <- 0:n
  r <- pbeta(.5,1+ys,1+n-ys)
  t <- pbeta(.5,1+ys,1+n-ys,lower.tail=FALSE)
  rows[[n]] <- data.frame(n=n,response=max(c(-1,ys[r>.95])),
    toxicity=min(c(n+1,ys[t>.95])))
}
write.csv(do.call(rbind,rows),'tests/fixtures/multc99-phaseiia-bounds.csv',row.names=FALSE)

# Enumerate every complete length-six paired-outcome sequence (4^6=4096),
# then identify its first stopping look. This is independent of a count-state
# recursion and verifies cohort monitoring, outcome association and Wald means.
outcomes <- matrix(c(1,1,1,0,0,1,0,0),ncol=2,byrow=TRUE)
probabilities <- list(c(.1,.45,.15,.3),c(.25,.30,0,.45))
rows <- list(); moments <- list()
for(cohort in c(1,2)) for(scenario in 1:2) {
  p <- probabilities[[scenario]]
  mass <- matrix(0,7,4,dimnames=list(NULL,c('response','toxicity','both','cap')))
  er <- et <- 0
  visit <- function(path,weight) {
    if(length(path)<6) {
      for(k in 1:4) if(p[k]>0) visit(c(path,k),weight*p[k])
      return(invisible(NULL))
    }
    data <- outcomes[path,,drop=FALSE]
    for(n in seq(cohort,6,cohort)) {
      if(n<2) next
      r <- sum(data[seq_len(n),1]); t <- sum(data[seq_len(n),2])
      sr <- pbeta(.45,1+r,1+n-r)>.8
      st <- pbeta(.30,1+t,1+n-t,lower.tail=FALSE)>.85
      # Reaching the cap is recorded separately, even if a rule also crosses.
      reason <- if(n==6) 4 else if(sr && st) 3 else if(sr) 1 else if(st) 2 else 0
      if(reason>0) {
        mass[n+1,reason] <<- mass[n+1,reason]+weight
        er <<- er+weight*r; et <<- et+weight*t
        break
      }
    }
  }
  visit(integer(),1)
  pmf <- rowSums(mass); en <- sum((0:6)*pmf)
  for(n in 0:6) for(reason in colnames(mass))
    rows[[length(rows)+1]] <- data.frame(cohort=cohort,scenario=scenario,n=n,
      reason=reason,probability=mass[n+1,reason])
  moments[[length(moments)+1]] <- data.frame(cohort=cohort,scenario=scenario,
    expected_n=en,sd_n=sqrt(sum(((0:6)-en)^2*pmf)),responses=er,toxicities=et)
}
write.csv(do.call(rbind,rows),'tests/fixtures/multc-path-enumeration.csv',row.names=FALSE)
write.csv(do.call(rbind,moments),'tests/fixtures/multc-path-moments.csv',row.names=FALSE)
