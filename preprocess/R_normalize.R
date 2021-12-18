source("R_functions/AFS.R")
source("R_functions/ALSFS.R")
source("R_functions/LS_Smoothing.R")
source("R_functions/BoundaryCorrection.R")

fl <- list.files(path = "temp_for_R",full.names = T,pattern = ".csv")
for(f in fl) {
    print(f)
    x <- read.csv(f)
    res <- data.frame(x$wv[x$flux != 0], AFS(x[x$flux != 0,], 0.7, 0.4))
    colnames(res)[1] <- 'wv'
    colnames(res)[2] <- 'res'
    output <- merge(x, res, all=T)
    output[is.na(output)] <- 1
    write.csv(output, file=f, row.names=F)
}
