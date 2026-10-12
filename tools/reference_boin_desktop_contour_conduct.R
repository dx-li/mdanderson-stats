# Original desktop contour conduct wrappers; originals remain external inputs.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R_DIRECTORY OUTPUT_CSV
options(warn=2,digits=17)
args<-commandArgs(TRUE);stopifnot(length(args)==2)
files<-c('get.boundary.R','next.subtrial2_for_winApp.R','next.comb2_for_winApp.R')
expected<-c('4c67779cb7735c845f9b490e3d2ed57b','a0c5f84c89cf270c9fbe7e8f2a590760','098857e8a4024951483549ea71696b3a')
stopifnot(all(unname(tools::md5sum(file.path(args[1],files)))==expected))
native<-new.env(parent=globalenv())
for(file in files) sys.source(file.path(args[1],file),envir=native)
encode<-function(x) paste(as.vector(t(x)),collapse=';')
records<-list()
spaces<-list(matrix(c(1,1,2,1,2,2,2,3),ncol=2,byrow=TRUE),
             matrix(c(1,2,1,3),ncol=2,byrow=TRUE))
for(target in c(.25,.3)) for(toxic in c(1.4*target,.5)) for(runin in c(FALSE,TRUE)) {
 for(safe in c(FALSE,TRUE)) for(subtrial in 1:2) {
  space<-spaces[[subtrial]]
  for(position in 1:nrow(space)) for(size in c(3L,6L,12L)) for(events in 0:size) {
   n<-y<-matrix(0L,2,3)
   if(subtrial==2) for(k in 1:nrow(spaces[[1]])) n[spaces[[1]][k,1],spaces[[1]][k,2]]<-3L
   if(position>1) for(k in 1:(position-1)) n[space[k,1],space[k,2]]<-3L
   dose<-space[position,];n[dose[1],dose[2]]<-size;y[dose[1],dose[2]]<-events
   result<-NULL;error<-''
   messages<-capture.output(tryCatch({result<-native$next.comb2_for_winAPP(target,n,y,dose,
     ncohort=c(4L,2L),cohortsize=3L,n.earlystop=12L,p.tox=toxic,
     extrasafe=safe,boDoThreePlusThreeRunIn=runin)},error=function(e) error<<-conditionMessage(e)))
   # The managed outer controller classifies printed status, not result$status.
   # Record independent host classifications without redistributing report text.
   text<-paste(messages,collapse=' ')
   managed<-if(grepl('due to exhausting',text,fixed=TRUE)) 3L else
       if(grepl('Terminate the trial because the lowest dose',text,fixed=TRUE)) 2L else
       if(grepl('no more subtrials',text,fixed=TRUE)) 5L else 0L
   if(managed==0L && !is.null(result) && (any(result$drecom<1) || result$drecom[1]>2 || result$drecom[2]>3)) managed<-4L
   records[[length(records)+1L]]<-data.frame(target=target,toxic_probability=toxic,runin=runin,
       extra_safe=safe,patients=encode(n),toxicities=encode(y),current_a=dose[1],current_b=dose[2],
       next_a=if(is.null(result)) NA else result$drecom[1],
       next_b=if(is.null(result)) NA else result$drecom[2],
       status=if(is.null(result)) '' else result$status,managed_status=managed,error=error)
  }
 }
}
write.csv(do.call(rbind,records),args[2],row.names=FALSE,na='')
