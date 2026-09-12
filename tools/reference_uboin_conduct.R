# Independent base-R stage-I boundaries, using the published U-BOIN rules.
# The toxicity safety threshold is the upper limit, not the BOIN target.
options(digits=17)
rows <- list()
for(target in c(.15,.2,.25,.3)) {
  safe <- .6*target; toxic <- 1.4*target; upper <- target+.05
  le <- log((1-safe)/(1-target))/log(target*(1-safe)/(safe*(1-target)))
  ld <- log((1-target)/(1-toxic))/log(toxic*(1-target)/(target*(1-toxic)))
  for(n in c(3,6,9,12,18,24,30)) {
    m <- 0:n
    eliminate <- m[pbeta(upper,m+1,n-m+1,lower.tail=FALSE)>.95]
    rows[[length(rows)+1]] <- data.frame(target=target,upper=upper,n=n,
      escalate=max(m[m/n<=le]),deescalate=min(m[m/n>=ld]),
      eliminate=if(length(eliminate)) min(eliminate) else n+1)
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/uboin-stage-one.csv',row.names=FALSE)
