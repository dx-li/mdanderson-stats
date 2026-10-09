# Independent posterior tables for the recovered native compact-boundary loop.
# Base R only. No production Python or third-party statistical packages.
options(digits=17)
profiles <- list(
  fixed=list(a=1,b=1,h=.5,ha=NA,hb=NA,delta=0,cutoff=.95),
  fractional=list(a=.6,b=1.4,h=.3,ha=NA,hb=NA,delta=.1,cutoff=.85),
  disabled=list(a=1,b=1,h=.5,ha=NA,hb=NA,delta=0,cutoff=1),
  prior=list(a=1,b=1,h=.5,ha=NA,hb=NA,delta=0,cutoff=.49),
  tie=list(a=1,b=1,h=.5,ha=NA,hb=NA,delta=0,cutoff=.5),
  historical=list(a=.6,b=1.4,h=NA,ha=3,hb=7,delta=0,cutoff=.85),
  shifted=list(a=.8,b=1.2,h=NA,ha=.7,hb=2.3,delta=.15,cutoff=.85),
  negative=list(a=.8,b=1.2,h=NA,ha=3,hb=7,delta=-.2,cutoff=.85),
  zero_support=list(a=1,b=1,h=0,ha=NA,hb=NA,delta=0,cutoff=0),
  certain=list(a=1,b=1,h=1,ha=NA,hb=NA,delta=0,cutoff=0)
)
cases <- list()
add <- function(name,cap,minimum,cohort,r,t) {
  cases[[length(cases)+1]] <<- list(name=name,cap=cap,minimum=minimum,
    cohort=cohort,response=r,toxicity=t)
}
for(cohort in c(1,2,3,6)) {
  add(paste0('fixed_cohort_',cohort),24,1,cohort,'fixed','fractional')
  add(paste0('minimum_cohort_',cohort),24,12,cohort,'fractional','fixed')
}
add('response_disabled',12,3,3,'disabled','fixed')
add('toxicity_disabled',12,3,3,'fixed','disabled')
add('both_disabled',12,6,3,'disabled','disabled')
add('response_prior',12,6,3,'prior','fixed')
add('toxicity_prior',12,6,3,'fixed','prior')
add('both_prior',12,6,3,'prior','prior')
add('strict_prior_tie',12,6,3,'tie','tie')
add('random_history',12,3,3,'historical','historical')
add('positive_response_margin',12,2,2,'shifted','historical')
# Native toxicity is converted to the LOWER-tail probability for NONtoxicities.
# A negative complemented margin means a positive toxicity margin in the design.
add('positive_toxicity_margin',12,2,2,'historical','negative')
add('negative_response_margin',12,2,2,'negative','historical')
add('support_zero_cutoff',6,1,1,'zero_support','zero_support')
add('support_certain_cutoff',6,3,3,'certain','certain')
add('minimum_at_cap',12,12,3,'fixed','fixed')
tail <- function(a,b,p) {
  if(!is.na(p$h)) return(pbeta(p$h+p$delta,a,b))
  if(p$delta==0 && p$ha==p$hb && a==b) return(.5)
  cuts <- sort(unique(c(0,1,max(0,min(1,-p$delta)),max(0,min(1,1-p$delta)))))
  sum(vapply(seq_len(length(cuts)-1),function(j)
    integrate(function(s) dbeta(s,p$ha,p$hb)*
      pbeta(s+p$delta,a,b),cuts[j],cuts[j+1],
      rel.tol=1e-10,abs.tol=1e-12,subdivisions=300)$value,numeric(1)))
}
rows <- list()
for(case in cases) for(endpoint in c('response','toxicity')) {
  p <- profiles[[case[[endpoint]]]]
  for(n in 0:case$cap) for(i in 0:n) {
    rows[[length(rows)+1]] <- data.frame(case=case$name,cap=case$cap,
      minimum=case$minimum,cohort=case$cohort,endpoint=endpoint,
      a=p$a,b=p$b,h=p$h,ha=p$ha,hb=p$hb,delta=p$delta,cutoff=p$cutoff,
      n=n,i=i,probability=tail(p$a+i,p$b+n-i,p))
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/multc-lean-boundary-r-tails.csv',
  row.names=FALSE,na='')
