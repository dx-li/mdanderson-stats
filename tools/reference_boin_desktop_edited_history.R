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
records<-list();set.seed(193)
for(case in 1:120) for(safe in c(FALSE,TRUE)) {
 n<-matrix(sample(c(0L,3L,6L),6,replace=TRUE),2,3)
 y<-matrix(rbinom(6,as.vector(n),.35),2,3)
 dose<-c(1L,sample(2:3,1));n[dose[1],dose[2]]<-max(3,n[dose[1],dose[2]])
 result<-NULL;error<-''
 messages<-capture.output(tryCatch({result<-native$next.comb2_for_winAPP(.25,n,y,dose,
   ncohort=c(8L,8L),cohortsize=3L,n.earlystop=12L,extrasafe=safe)},
   error=function(e) error<<-conditionMessage(e)))
 text<-paste(messages,collapse=' ')
 managed<-if(grepl('due to exhausting',text,fixed=TRUE)) 3L else
   if(grepl('Terminate the trial because the lowest dose',text,fixed=TRUE)) 2L else
   if(grepl('no more subtrials',text,fixed=TRUE)) 5L else 0L
 if(managed==0L && !is.null(result) && (any(result$drecom<1) || result$drecom[1]>2 || result$drecom[2]>3)) managed<-4L
 records[[length(records)+1L]]<-data.frame(extra_safe=safe,patients=encode(n),toxicities=encode(y),
   current_a=dose[1],current_b=dose[2],next_a=if(is.null(result)) NA else result$drecom[1],
   next_b=if(is.null(result)) NA else result$drecom[2],managed_status=managed,error=error)
}
write.csv(do.call(rbind,records),args[2],row.names=FALSE,na='')
