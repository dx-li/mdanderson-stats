# Independent base-R EffTox references; no packages or simulation.
# Thall & Cook (2004), DOI 10.1111/j.0006-341X.2004.00218.x, eqs 1-3.
# Cook (2006), NewTradeOffFunctions.pdf, bivariate Lp contour.
options(digits=17)
doses <- c(1,2,4)
x <- log(doses)-mean(log(doses))
# Rows E=0,1; columns T=0,1 at each dose.
counts <- list(matrix(c(4,1,2,1),2,byrow=TRUE),
               matrix(c(2,1,4,1),2,byrow=TRUE),
               matrix(c(1,2,4,3),2,byrow=TRUE))
joint <- function(theta) {
  t <- plogis(theta[1]+theta[2]*x)
  e <- plogis(theta[3]+theta[4]*x+theta[5]*x*x)
  rho <- tanh(theta[6]/2)
  lapply(seq_along(x),function(k) {
    outer(c(1-e[k],e[k]),c(1-t[k],t[k]))+
      matrix(c(1,-1,-1,1),2)*rho*e[k]*(1-e[k])*t[k]*(1-t[k])
  })
}
ll <- function(theta) sum(vapply(seq_along(x),function(k)
  sum(counts[[k]]*log(joint(theta)[[k]])),numeric(1)))
base <- c(-1,.8,.2,1.1,-.3,0)
rows <- list()
for(psi in c(-5,0,2)) {
  theta <- base; theta[6] <- psi
  for(k in seq_along(x)) for(e in 0:1) for(t in 0:1)
    rows[[length(rows)+1]] <- data.frame(psi=psi,dose=k,e=e,t=t,
      probability=joint(theta)[[k]][e+1,t+1],loglikelihood=ll(theta))
}
write.csv(do.call(rbind,rows),'tests/fixtures/efftox-joint.csv',row.names=FALSE)

# One-dimensional posterior integrations. Independence when psi=0 makes
# the two-intercept case exactly separable. Association-only and positive
# toxicity-slope cases retain the full bivariate likelihood.
int <- function(f,lo=-Inf,hi=Inf) integrate(f,lo,hi,rel.tol=1e-10,
  abs.tol=1e-12,subdivisions=300)$value
rows <- list()
for(case in c('independent_intercepts','association','positive_slope')) {
  theta <- base
  indices <- if(case=='independent_intercepts') c(1,3) else
    if(case=='association') 6 else 2
  means <- if(case=='independent_intercepts') c(-1,.2) else
    if(case=='association') .4 else -1
  sds <- if(case=='independent_intercepts') c(.9,1.1) else
    if(case=='association') 1.3 else .8
  if(case=='positive_slope') theta[6] <- 1.2
  for(a in seq_along(indices)) {
    j <- indices[a]; m <- means[a]; s <- sds[a]
    lower <- if(case=='positive_slope') -m/s else -Inf
    offset <- ll(theta)
    density <- function(z) vapply(z,function(zz) {
      par <- theta; par[j] <- m+s*zz
      exp(ll(par)-offset)*dnorm(zz)
    },numeric(1))
    norm <- int(density,lower)
    expectation <- function(f) int(function(z) density(z)*f(m+s*z),lower)/norm
    pm <- expectation(identity)
    pv <- expectation(function(v) (v-pm)^2)
    rows[[length(rows)+1]] <- data.frame(case=case,parameter=j,quantity='mean',
      dose=0,value=pm)
    rows[[length(rows)+1]] <- data.frame(case=case,parameter=j,quantity='sd',
      dose=0,value=sqrt(pv))
    if(j %in% c(1,3)) for(k in seq_along(x)) {
      shift <- if(j==1) theta[2]*x[k] else theta[4]*x[k]+theta[5]*x[k]^2
      prob <- expectation(function(v) plogis(v+shift))
      cut <- qlogis(if(j==1) .3 else .45)-shift
      tail <- if(j==1) int(density,lower,(cut-m)/s)/norm else
        int(density,(cut-m)/s,Inf)/norm
      rows[[length(rows)+1]] <- data.frame(case=case,parameter=j,
        quantity='probability',dose=k,value=prob)
      rows[[length(rows)+1]] <- data.frame(case=case,parameter=j,
        quantity='acceptable',dose=k,value=tail)
    }
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/efftox-posterior.csv',row.names=FALSE)

# Official tutorial contour plus distinctly concave and convex cases.
rows <- list()
for(a in list(c(.5,.65,.7,.25),c(.4,.8,.8,.1),c(.5,.5,.7,.3))) {
  e0 <- a[1]; t1 <- a[2]; em <- a[3]; tm <- a[4]
  p <- uniroot(function(p) ((1-em)/(1-e0))^p+(tm/t1)^p-1,
    c(1e-6,100),tol=1e-13)$root
  for(v in list(c(e0,0),c(1,t1),c(em,tm),c(1,0),c(.2,.05),
                c(.4,.1),c(.6,.15),c(.8,.2),c(.9,.3))) {
    u <- 1-(((1-v[1])/(1-e0))^p+(v[2]/t1)^p)^(1/p)
    rows[[length(rows)+1]] <- data.frame(e0=e0,t1=t1,em=em,tm=tm,p=p,
      e=v[1],t=v[2],utility=u)
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/efftox-contour.csv',row.names=FALSE)
