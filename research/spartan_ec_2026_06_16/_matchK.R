m <- readRDS("data/tool_pls_EC.RDS")
wn <- round(m$wavenumbers, 4)
exp <- read.csv("data/tool_EC_coeffs_lot251_biomass.csv")
exp <- exp[exp$Wavenumber != 0, ]
eb <- exp$b[match(wn, round(exp$Wavenumber, 4))]
best <- c(k=NA, r=-1)
for (k in 1:dim(m$coefficients)[3]) {
  r <- cor(m$coefficients[,1,k], eb, use="complete.obs")
  if (r > best["r"]) best <- c(k=k, r=r)
}
cat(sprintf("exported CSV best matches RDS at K=%d (r=%.4f)\n", best["k"], best["r"]))
# also report a few specific k
for (k in c(18,30,40,50,60,80)) cat(sprintf("  K=%2d r=%.4f\n", k, cor(m$coefficients[,1,k], eb, use="complete.obs")))
