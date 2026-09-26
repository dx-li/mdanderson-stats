# Independent base-R references; no Python, original source, or external packages.
# Run from repository root: Rscript tools/reference_mtadf.R
options(digits=17)

# Weighted min-max characterization, independent of the Python PAVA stack.
increasing <- function(y,w) {
  vapply(seq_along(y), function(j) {
    max(vapply(seq_len(j), function(a) {
      min(vapply(j:length(y), function(b)
        sum(y[a:b]*w[a:b])/sum(w[a:b]), numeric(1)))
    }, numeric(1)))
  }, numeric(1))
}
cases <- list(
  published_pool=list(y=c(.1,.3,.2,.8),w=c(10,10,5,5)),
  plateau=list(y=c(.1,.5,.5,.2),w=c(3,6,12,3)),
  decreasing=list(y=c(.8,.6,.4,.2),w=c(3,2,8,1)),
  increasing=list(y=c(0,.2,.7,1),w=c(1,2,3,4)),
  multiple_peaks=list(y=c(.6,.1,.7,.2,.9),w=c(2,11,3,8,5)),
  uneven=list(y=c(.8,.1,.7,.2),w=c(1,30,2,20)),
  flat=list(y=rep(.25,5),w=c(1,3,10,2,4)))
rows <- list()
for (name in names(cases)) {
  y <- cases[[name]]$y; w <- cases[[name]]$w; J <- length(y)
  fits <- lapply(seq_len(J), function(k) {
    left <- increasing(y[1:k],w[1:k])
    right <- if (k==J) numeric(0) else
      rev(increasing(rev(y[(k+1):J]),rev(w[(k+1):J])))
    c(left,right)
  })
  scores <- vapply(fits,function(f) sum((f-y)^2),numeric(1))
  chosen <- which.min(scores); fit <- fits[[chosen]]
  rows[[length(rows)+1]] <- data.frame(
    scenario=name,dose=seq_len(J)-1,value=y,weight=w,fitted=fit,
    chosen_split=chosen-1,peak=which.max(fit)-1,score=scores[chosen])
}
write.csv(do.call(rbind,rows),'tests/fixtures/mtadf-isotonic-reference.csv',row.names=FALSE)

settings <- data.frame(
  scenario=c('default','low_limit','high_limit','concentrated','diffuse'),
  phi=c(.3,.1,.8,.3,.3),cutoff=c(.8,.95,.5,.8,.8),
  margin=c(.05,.02,.1,.05,.05),concentration=c(.5,.5,.5,2,.1))
rows <- list()
for (i in seq_len(nrow(settings))) {
  s <- settings[i,]; mass <- 1-s$cutoff+s$margin
  a <- uniroot(function(a) pbeta(s$phi,a,s$concentration-a)-mass,
               c(s$concentration*1e-9,s$concentration*(1-1e-9)),tol=1e-13)$root
  b <- s$concentration-a
  n <- c(0,3,20,10000); x <- c(0,1,0,10000)
  rows[[i]] <- data.frame(s,alpha=a,beta=b,subjects=n,toxicities=x,
                         overdose=pbeta(s$phi,a+x,b+(n-x),lower.tail=FALSE),
                         row.names=NULL)
}
write.csv(do.call(rbind,rows),'tests/fixtures/mtadf-prior-reference.csv',row.names=FALSE)
