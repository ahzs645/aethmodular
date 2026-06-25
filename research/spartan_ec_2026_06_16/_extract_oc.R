m <- readRDS("data/tool_pls_OC.RDS")
nc <- dim(m$fitted.values)[3]
Y <- m$model$Value
X <- m$model$Spectrum
wv <- m$wavenumbers
cat("class", class(m), "| n", length(Y), "| ncomp", m$ncomp, "| method", m$method, "| Xdim", paste(dim(X),collapse="x"), "\n")
cat("Y(OC) summary:"); print(summary(Y))
write.csv(cbind(id=0:(nrow(X)-1), as.data.frame(X)), "data/rds_OC_X.csv", row.names=FALSE)
write.csv(data.frame(id=0:(length(Y)-1), Y_measured=Y), "data/rds_OC_Ymeasured.csv", row.names=FALSE)
md <- as.data.frame(m$metadata); write.csv(md, "data/rds_OC_metadata.csv", row.names=FALSE)
co <- m$coefficients[,1,18]; write.csv(data.frame(wavenumber=wv, b=as.numeric(co)), "data/rds_OC_coef_k18.csv", row.names=FALSE)
cat("wrote rds_OC_X/Ymeasured/metadata/coef\n"); cat("metadata cols:", paste(names(md),collapse=", "), "\n")
