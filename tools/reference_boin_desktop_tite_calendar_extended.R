# External original desktop TITE calendar with controlled arrival/outcome services.
# Records observations and original end-state; no source text or templates copied.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R_DIR OUTPUT_CSV
args<-commandArgs(TRUE);stopifnot(length(args)==2)
files<-c('get.oc.tite.R','select.mtd.tite.boin.R')
stopifnot(all(unname(tools::md5sum(file.path(args[1],files)))==
 c('e6c481052cc4a9a497e016bd3903a7aa','8cee6b3372ddf8362e0ab4cc189f4a43')))
native<-new.env(parent=globalenv())
for(f in files) sys.source(file.path(args[1],f),envir=native)
native$file.exists<-function(...) FALSE
original<-body(native$get.oc.tite)
body(native$get.oc.tite)<-as.call(c(list(as.name('{'),quote(on.exit({
 native$endstate<-list(doses=dv,entry=t.enter,events=t.event,tox=y,
                       duration=t.decision,waits=npend,stop=earlystop)
}))),as.list(original)[-1]))
native$rexp<-function(n,rate) rep(native$gap,n)
native$gen.tite<-function(dist,n,pi,alpha,T) {
 tox<-as.integer(rep(c(.1,.5,.9),length.out=n)<pi)
 list(tox=tox,t.tox=ifelse(tox==1,T*native$delay,T),ntox.st=sum(tox))
}
encode<-function(x) paste(x,collapse=';');rows<-list()
for(target in c(.25,.3)) for(gap in c(.1,.5)) for(delay in c(.2,.9)) {
 for(early in c(6L,12L)) for(runin in c(FALSE,TRUE)) for(safe in c(FALSE,TRUE)) {
  for(size in c(1L,3L)) for(maxpen in c(.35,.65)) for(prior in c('uniform','late')) for(truth in c('gradient','all','none','latent')) {
   p<-switch(truth,gradient=c(.1,target,.6),all=rep(.95,3),none=rep(.01,3),latent=rep(.55,3))
   native$gap<-gap;native$delay<-delay;native$endstate<-NULL
   result<-NULL;error<-''
   invisible(capture.output(tryCatch({result<-native$get.oc.tite(target,p,6L,size,maxt=1,prior.p=if(prior=='uniform') NA else c(.1,.2,.7),
      dist2=2,accrual=3,maxpen=maxpen,n.earlystop=early,RunIn=runin,
      extrasafe=safe,ntrial=1,seed=137)},error=function(e) error<<-conditionMessage(e))))
   e<-native$endstate
   rows[[length(rows)+1L]]<-data.frame(target=target,gap=gap,delay=delay,early=early,
    runin=runin,extra_safe=safe,truth=truth,cohort_size=size,max_pending=maxpen,prior=prior,patients=encode(result$npatients),
    toxicities=encode(result$ntox),selection=encode(result$selpercent),duration=result$duration,
    doses=encode(e$doses),entry=encode(e$entry),delays=encode(e$events),outcomes=encode(e$tox),
    waits=e$waits,stop=e$stop,error=error)
  }
 }
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE,na='')
