# Native high-target table edge contracts; no original source redistributed.
args<-commandArgs(TRUE);stopifnot(length(args)==2)
path<-file.path(args[1],'get.boundary.tite.R')
stopifnot(unname(tools::md5sum(path))=='19803f88bc78c49cc9fe7104c0258c94')
rows<-list()
for(safe in c(FALSE,TRUE)) {
 native<-new.env(parent=globalenv());sys.source(path,envir=native)
 result<-NULL;error<-''
 invisible(capture.output(tryCatch({result<-native$get.boundary.tite(.6,4L,3L,
   n.earlystop=12,extrasafe=safe,print=FALSE,output=FALSE)},
   error=function(e) error<<-conditionMessage(e))))
 rows[[length(rows)+1L]]<-data.frame(target=.6,extra_safe=safe,error=error,
   decision=if(is.null(result)) '' else tail(result$boundary_tab[result$boundary_tab[,1]=='3',6],1))
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
