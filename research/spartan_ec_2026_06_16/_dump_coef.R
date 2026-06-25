m <- readRDS("data/tool_pls_EC.RDS")
wn <- m$wavenumbers
for (k in c(3,18,25)) {
  co <- m$coefficients[,1,k]
  write.csv(data.frame(wavenumber=wn, b=as.numeric(co)), sprintf("data/rds_EC_coef_k%d.csv",k), row.names=FALSE)
}
cat("wavenumbers head:", head(wn,3), "... tail:", tail(wn,2), "\n")
cat("dumped coef at k=3,18,25\n")
