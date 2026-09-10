# Independent integration over a uniform efficacy quantile, not vendor code.
options(digits=17)
e <- c(.1,.2,.3,.4); t <- c(.4,.3,.2,.1)
ec <- c(0,cumsum(e)); tb <- qnorm(c(0,cumsum(t)))
rows <- list()
for(rho in c(.1,.75,-.6)) for(a in 1:4) for(b in 1:4) {
  f <- function(u) {
    x <- qnorm(u)
    pnorm((tb[b+1]-rho*x)/sqrt(1-rho*rho))-pnorm((tb[b]-rho*x)/sqrt(1-rho*rho))
  }
  p <- integrate(f,ec[a],ec[a+1],abs.tol=1e-13,rel.tol=1e-11,subdivisions=1000)$value
  rows[[length(rows)+1]] <- c(rho,a-1,b-1,p)
}
answer <- as.data.frame(do.call(rbind,rows))
names(answer) <- c('rho','efficacy','toxicity','probability')
write.table(answer,stdout(),sep=',',quote=FALSE,row.names=FALSE)
