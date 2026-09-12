# Independent analytic Dirichlet moments and Beta marginals, base R only.
# Rows are efficacy ascending; columns are toxicity ascending.
options(digits=17)
cases <- list(
  binary=list(n=matrix(c(2,1,5,2),2,byrow=TRUE),
              a=matrix(c(.1,.2,.3,.4),2,byrow=TRUE),
              u=matrix(c(30,0,100,50),2,byrow=TRUE),dt=1,re=1),
  categorical=list(n=matrix(c(2,1,0,3,2,1,5,1,2),3,byrow=TRUE),
                   a=matrix((1:9)/45,3,byrow=TRUE),
                   u=matrix(c(30,15,0,50,30,0,100,45,15),3,byrow=TRUE),dt=2,re=2),
  prior_only=list(n=matrix(0,2,3),a=matrix(c(.2,.1,.3,.4,.5,.6),2,byrow=TRUE),
                  u=matrix(c(30,15,0,100,70,50),2,byrow=TRUE),dt=2,re=1))
rows <- list()
for(name in names(cases)) {
  x <- cases[[name]]; a <- x$n+x$a; total <- sum(a)
  mean <- sum(a*x$u)/total
  # Evaluate second moment via Dirichlet cross moments, independently of
  # the centered-variance expression used by the Python implementation.
  av <- as.vector(a); uv <- as.vector(x$u)
  second <- sum(outer(av,av)*outer(uv,uv))+sum(av*uv^2)
  variance <- second/(total*(total+1))-mean^2
  toxic <- sum(a[,(x$dt+1):ncol(a),drop=FALSE])
  response <- sum(a[(x$re+1):nrow(a),,drop=FALSE])
  rows[[length(rows)+1]] <- data.frame(case=name,mean=mean,variance=variance,
    overdose=pbeta(.3,toxic,total-toxic,lower.tail=FALSE),
    loweff=pbeta(.2,response,total-response))
}
write.csv(do.call(rbind,rows),'tests/fixtures/uboin-posterior.csv',row.names=FALSE)
