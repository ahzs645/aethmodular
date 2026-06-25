# Dissect the tool's exported PLS model (pls::mvr object) to recover the EXACT
# training X, Y, preprocessing, and coefficients it used — so we can test whether
# our local fit is equivalent on identical data.
args <- commandArgs(trailingOnly = TRUE)
rds <- args[1]; outdir <- args[2]
m <- readRDS(rds)

cat("=== class ===\n"); print(class(m))
cat("=== names ===\n"); print(names(m))
cat("=== ncomp ===\n"); print(m$ncomp)
cat("=== method ===\n"); print(m$method)
cat("=== scaled? (m$scale) ===\n"); print(if (is.null(m$scale)) "NULL/none" else head(m$scale))
cat("=== call ===\n"); print(m$call)

# dimensions
if (!is.null(m$coefficients)) { cat("=== dim(coefficients) [nvar,nresp,ncomp] ===\n"); print(dim(m$coefficients)) }
if (!is.null(m$fitted.values)) { cat("=== dim(fitted.values) ===\n"); print(dim(m$fitted.values)) }

# recover Y = fitted + residuals at final ncomp
nc <- dim(m$fitted.values)[3]
fit <- m$fitted.values[, 1, nc]
res <- m$residuals[, 1, nc]
Y <- fit + res
ids <- rownames(m$fitted.values)
cat("=== n obs ===\n"); print(length(Y))
cat("=== Y summary (measured) ===\n"); print(summary(Y))
cat("=== first rownames (ids) ===\n"); print(head(ids, 5))

# the model frame: response + predictor matrix
cat("=== str(m$model, max.level=1) ===\n")
if (!is.null(m$model)) print(str(m$model, max.level = 1)) else cat("no $model\n")

# write Y (measured) with ids
ydf <- data.frame(id = ids, Y_measured = Y)
write.csv(ydf, file.path(outdir, "rds_EC_Ymeasured.csv"), row.names = FALSE)

# write the predictor matrix X if present in the model frame
X <- NULL
if (!is.null(m$model)) {
  for (nm in names(m$model)) {
    if (is.matrix(m$model[[nm]]) && ncol(m$model[[nm]]) > 50) { X <- m$model[[nm]]; cat("X from m$model$", nm, " dim ", paste(dim(X), collapse="x"), "\n", sep="") }
  }
}
if (!is.null(X)) {
  cat("=== X colnames head (wavenumbers) ===\n"); print(head(colnames(X), 5))
  write.csv(cbind(id = ids, as.data.frame(X)), file.path(outdir, "rds_EC_X.csv"), row.names = FALSE)
  cat("wrote rds_EC_X.csv  dim ", paste(dim(X), collapse="x"), "\n")
} else cat("could not locate X predictor matrix in model frame\n")

# write coefficients at final ncomp (raw, per-wavenumber) for direct comparison
co <- m$coefficients[, 1, nc]
cdf <- data.frame(wavenumber = names(co), b = as.numeric(co))
write.csv(cdf, file.path(outdir, "rds_EC_coef_finalncomp.csv"), row.names = FALSE)
cat("wrote rds_EC_coef_finalncomp.csv  (ncomp=", nc, ")\n", sep="")
cat("=== Xmeans/Ymeans present? ===\n"); print(c(Xmeans = !is.null(m$Xmeans), Ymeans = !is.null(m$Ymeans)))
if (!is.null(m$Ymeans)) { cat("Ymeans: "); print(m$Ymeans) }
