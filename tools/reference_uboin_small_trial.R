# Exact enumeration of a two-dose, two-cohort U-BOIN trial in base R.
# Explicit settings: prior .25/cell, tried-only candidates, N=6, s1=3,
# s2=9, cohort=3, winner allocation; toxicity .3, efficacy .2, delta .05.
options(digits=17)
patterns <- as.matrix(expand.grid(rep(list(0:3),4)))
patterns <- patterns[rowSums(patterns)==3,,drop=FALSE]
gumbel <- function(t,e,c=.2) {
  rho <- 2*plogis(c)-1
  c((1-e)*(1-t),(1-e)*t,e*(1-t),e*t)+
    e*(1-e)*t*(1-t)*c(1,-1,-1,1)*rho
}
truth <- rbind(gumbel(.15,.35),gumbel(.35,.65))
selection <- function(n) {
  a <- n+.25
  admissible <- rowSums(n)>0 &
    pbeta(.3,a[,2]+a[,4],a[,1]+a[,3],lower.tail=FALSE)<=.95 &
    pbeta(.2,a[,3]+a[,4],a[,1]+a[,2])<=.9
  if(!any(admissible)) return(0L)
  u <- as.vector(a %*% c(30,0,100,50))/rowSums(a)
  u[!admissible] <- -Inf
  which.max(u)
}
selected <- numeric(3); expected <- matrix(0,2,4); early <- 0
le <- log(.85/.75)/log(.25*.85/(.15*.75))
for(i in seq_len(nrow(patterns))) {
  first <- patterns[i,]; w1 <- dmultinom(first,prob=truth[1,])
  n <- rbind(first,rep(0,4)); m <- sum(first[c(2,4)])
  if(pbeta(.3,m+1,4-m,lower.tail=FALSE)>.95) {
    selected[1] <- selected[1]+w1; expected <- expected+w1*n; early <- early+w1
    next
  }
  dose <- if(m/3<=le) 2L else selection(n)
  if(dose==0) {
    selected[1] <- selected[1]+w1; expected <- expected+w1*n; early <- early+w1
    next
  }
  for(k in seq_len(nrow(patterns))) {
    final <- n; final[dose,] <- final[dose,]+patterns[k,]
    weight <- w1*dmultinom(patterns[k,],prob=truth[dose,])
    chosen <- selection(final)
    selected[chosen+1] <- selected[chosen+1]+weight
    expected <- expected+weight*final
  }
}
out <- data.frame(selection0=selected[1],selection1=selected[2],selection2=selected[3],
                  patients1=sum(expected[1,]),patients2=sum(expected[2,]),early=early)
write.csv(out,'tests/fixtures/uboin-small-trial.csv',row.names=FALSE)
