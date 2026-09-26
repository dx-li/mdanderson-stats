# Independent base-R references for BCHM. No JAGS or package installation.
options(digits=17)

# Released BCHM 1.00 integrated Gaussian cluster likelihood, omitting constants
# that are identical for every assignment of the incoming subgroup.
log_cluster <- function(W,S,Q,mu,v,d) {
  postv <- 1/(1/v+W/d)
  postm <- (mu/v+S/d)*postv
  (-Q/d+postm^2/postv+log(postv)-mu^2/v-log(v))/2
}
cases <- list(
  unequal=list(W=c(7,90),S=c(2,20),Q=c(4/7,400/90),y=5,n=12,
               mu=.2,v=20,d=.01,alpha=.001),
  native=list(W=c(25,50),S=c(2,17),Q=c(4/25,289/50),y=3,n=25,
              mu=.2,v=10,d=.001,alpha=1e-60),
  diffuse=list(W=c(4,12),S=c(0,12),Q=c(0,12),y=2,n=5,
               mu=.4,v=2,d=.5,alpha=3),
  small_new=list(W=c(100,100),S=c(50,50),Q=c(25,25),y=50,n=100,
                 mu=.5,v=1,d=.1,alpha=1e-100))
rows <- list()
for(name in names(cases)) {
  a <- cases[[name]]; x <- a$y/a$n
  ll <- log_cluster(a$W+a$n,a$S+a$y,a$Q+a$n*x*x,a$mu,a$v,a$d)-
        log_cluster(a$W,a$S,a$Q,a$mu,a$v,a$d)
  fresh <- log_cluster(a$n,a$y,a$n*x*x,a$mu,a$v,a$d)
  lp <- c(log(a$W)+ll,log(a$alpha)+fresh)
  prob <- exp(lp-max(lp)); prob <- prob/sum(prob)
  for(j in seq_along(prob)) rows[[length(rows)+1]] <- data.frame(
    case=name,W=paste(a$W,collapse=";"),S=paste(a$S,collapse=";"),
    y=a$y,n=a$n,mu=a$mu,prior_variance=a$v,data_variance=a$d,
    concentration=a$alpha,choice=j,probability=prob[j])
}
write.csv(do.call(rbind,rows),'tests/fixtures/bchm-assignment.csv',row.names=FALSE)

# Equal subgroup sizes yield an ordinary partition law with concentration
# alpha/n and powered Gaussian likelihoods. Enumerate all five partitions of 3.
y <- c(1,2,7); n <- rep(10,3); mu <- .2; v <- 2; d <- .2; alpha <- 4
partitions <- rbind(c(1,1,1),c(1,1,2),c(1,2,1),c(1,2,2),c(1,2,3))
lp <- apply(partitions,1,function(z) {
  k <- max(z)
  out <- k*log(alpha/n[1])+sum(lgamma(tabulate(z)))
  for(j in seq_len(k)) {
    idx <- z==j
    out <- out+log_cluster(sum(n[idx]),sum(y[idx]),sum(y[idx]^2/n[idx]),mu,v,d)
  }
  out
})
weights <- exp(lp-max(lp)); weights <- weights/sum(weights)
rows <- list()
for(i in 1:3) for(j in 1:3) rows[[length(rows)+1]] <- data.frame(
  i=i,j=j,similarity=sum(weights*(partitions[,i]==partitions[,j])))
write.csv(do.call(rbind,rows),'tests/fixtures/bchm-partition.csv',row.names=FALSE)

# Second-stage integration with fixed tau1=3 but a RANDOM shared mu. This
# checks the off-diagonal similarity weights; fixing mu as well would not.
# A Gamma(shape=3e10,rate=1e10) prior approaches this precision limit.
y <- c(0,6); n <- c(8,10); mu0 <- qlogis(mean(y/n)); tau2 <- .7
C <- matrix(c(1,.2,.2,1),2); tau1 <- 3; target <- .35
int <- function(f,a=-Inf,b=Inf) integrate(f,a,b,rel.tol=2e-8,
                                        abs.tol=1e-13,subdivisions=200)$value
rows <- lapply(1:2,function(i) {
  inner <- function(mu,j,kind) {
    sd <- 1/sqrt(tau1*C[i,j])
    f <- function(z) {
      p <- plogis(mu+sd*z)
      dbinom(y[j],n[j],p)*dnorm(z)*if(kind=="mean") p else 1
    }
    int(f,if(kind=="tail") (qlogis(target)-mu)/sd else -Inf,Inf)
  }
  outer <- function(kind) int(function(z) vapply(z,function(zz) {
    mu <- mu0+zz/sqrt(tau2)
    dnorm(zz)*inner(mu,i,kind)*inner(mu,3-i,"norm")
  },numeric(1)))
  norm <- outer("norm")
  data.frame(group=i,y=y[i],n=n[i],mean=outer("mean")/norm,
             efficacy=outer("tail")/norm)
})
write.csv(do.call(rbind,rows),'tests/fixtures/bchm-borrowing-limit.csv',row.names=FALSE)

# Native rounded probabilities are used for the final strict decision.
p <- c(.0005,.0015,.1005,.1015,.5005,.5015,.6005,.6015,.9985,.9995)
write.csv(data.frame(probability=p,rounded=round(p,3)),
          'tests/fixtures/bchm-rounding.csv',row.names=FALSE)
