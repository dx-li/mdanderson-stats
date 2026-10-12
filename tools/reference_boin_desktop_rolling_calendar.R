# Original desktop Rolling Six calendar with controlled input services.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R_DIR OUTPUT_CSV
args<-commandArgs(TRUE);stopifnot(length(args)==2)
original_path<-file.path(args[1],'get.oc.R6.R')
stopifnot(unname(tools::md5sum(original_path))=='0417ef2a3bd5ae5df1294f4bce11993b')
native<-new.env(parent=globalenv());sys.source(original_path,envir=native)
original<-body(native$get.oc.R6)
body(native$get.oc.R6)<-as.call(c(list(as.name('{'),quote(on.exit({
 native$endstate<-list(doses=dv,entry=t.enter,events=t.event,tox=y,
                      decision=t.decision,stop=earlystop)
}))),as.list(original)[-1]))
native$rexp<-function(n,rate) rep(native$gap,n)
native$gen.tite<-function(dist,n,pi,alpha,T) {
 u<-c(.1,.5,.9)[(native$offset+seq_len(n)-1L)%%3L+1L];native$offset<-native$offset+n
 tox<-as.integer(u<pi)
 list(tox=tox,t.tox=ifelse(tox==1,T*native$delay,T),ntox.st=sum(tox))
}
encode<-function(x) paste(x,collapse=';');rows<-list()
for(size in c(1L,3L)) for(gap in c(.1,.5,2)) for(delay in c(.2,.9)) {
 for(start in c(1L,2L)) for(truth in c('gradient','all','none','latent')) {
  p<-switch(truth,gradient=c(.05,.25,.6),all=rep(.95,3),none=rep(.01,3),latent=rep(.55,3))
  native$gap<-gap;native$delay<-delay;native$offset<-0L;native$endstate<-NULL
  result<-NULL;error<-'';setTimeLimit(elapsed=3,transient=TRUE)
  invisible(capture.output(tryCatch({result<-native$get.oc.R6(.25,p,size,maxt=1,
     dist2=2,accrual=3,startdose=start,ntrial=1,seed=137)},error=function(e) error<<-conditionMessage(e))))
  setTimeLimit(cpu=Inf,elapsed=Inf,transient=FALSE)
  e<-native$endstate
  rows[[length(rows)+1L]]<-data.frame(size=size,gap=gap,delay=delay,start=start,truth=truth,
    patients=encode(result$npatients),toxicities=encode(result$ntox),selection=encode(result$selpercent),
    duration=if(is.null(result)) NA else result$duration,doses=encode(e$doses),entry=encode(e$entry),
    delays=encode(e$events),outcomes=encode(e$tox),decision=e$decision,stop=e$stop,error=error)
 }
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE,na='')
