# Independent base-R scalar reference for detection and gel quantification.
options(digits=17, warn=2)
base <- outer(0:5,0:7,function(r,c) 2+.1*r+.05*c)
base[3,4] <- base[3,4]+10
base[5,7] <- base[5,7]+20
images <- list(base,1.5*base+3,.75*base+1)
average <- Reduce('+',images)/length(images)
# Python region convention (1,6,1,8) means these one-based index sets.
ri <- 2:6; ci <- 2:8
threshold <- as.numeric(quantile(average[ri,ci],.75,type=7))
candidates <- list()
for(r in ri) for(c in ci) {
  neighbors <- c(if(r>min(ri)) average[r-1,c],if(r<max(ri)) average[r+1,c],
                 if(c>min(ci)) average[r,c-1],if(c<max(ci)) average[r,c+1])
  if(average[r,c]>threshold && all(average[r,c]>=neighbors))
    candidates[[length(candidates)+1]] <- c(row=r,col=c,intensity=average[r,c])
}
candidates <- do.call(rbind,candidates)
candidates <- candidates[order(-candidates[,'intensity'],candidates[,'row'],candidates[,'col']),,drop=FALSE]
selected <- matrix(numeric(0),ncol=3)
for(i in seq_len(nrow(candidates))) {
  item <- candidates[i,]
  if(nrow(selected)==0 || all(pmax(abs(selected[,1]-item[1]),abs(selected[,2]-item[2]))>2))
    selected <- rbind(selected,item)
}
coordinates <- data.frame(row=selected[,1]-1,col=selected[,2]-1,
                          intensity=selected[,3],threshold=threshold)
write.csv(coordinates,'tests/fixtures/pinnacle-peaks.csv',row.names=FALSE)
pixel_rows <- list()
for(i in seq_along(images)) for(r in 1:6) for(c in 1:8)
  pixel_rows[[length(pixel_rows)+1]] <- data.frame(image=i-1,row=r-1,col=c-1,
      value=images[[i]][r,c],average=average[r,c])
write.csv(do.call(rbind,pixel_rows),'tests/fixtures/pinnacle-images.csv',row.names=FALSE)
rows <- list()
for(method in c('none','local_minimum','local_quantile','global_quantile')) {
  for(i in seq_along(images)) {
    image <- images[[i]]
    raw <- bg <- numeric(nrow(selected))
    for(j in seq_len(nrow(selected))) {
      r<-selected[j,1]; c<-selected[j,2]
      window <- function(radius) image[seq(max(min(ri),r-radius),min(max(ri),r+radius)),
                                       seq(max(min(ci),c-radius),min(max(ci),c+radius))]
      raw[j] <- max(window(1))
      bg[j] <- switch(method,none=0,local_minimum=min(window(2)),
                      local_quantile=as.numeric(quantile(window(2),.25,type=7)),
                      global_quantile=as.numeric(quantile(image[ri,ci],.25,type=7)))
    }
    corrected <- raw-bg
    for(normalization in c('none','mean_pinnacle','pinnacle_sum','image_volume')) {
      factor <- switch(normalization,none=1,mean_pinnacle=mean(corrected),
                       pinnacle_sum=sum(corrected),image_volume=sum(image[ri,ci]))
      for(j in seq_len(nrow(selected)))
        rows[[length(rows)+1]] <- data.frame(background=method,normalization=normalization,
          image=i-1,peak=j-1,raw=raw[j],baseline=bg[j],corrected=corrected[j],
          factor=factor,normalized=corrected[j]/factor)
    }
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/pinnacle-quantification.csv',row.names=FALSE)
