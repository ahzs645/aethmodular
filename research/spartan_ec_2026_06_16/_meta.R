m <- readRDS("data/tool_pls_EC.RDS")
cat("=== str(m$metadata) ===\n"); print(str(m$metadata))
if (!is.null(m$metadata)) {
  md <- as.data.frame(m$metadata)
  cat("=== metadata head ===\n"); print(head(md,3))
  write.csv(md, "data/rds_EC_metadata.csv", row.names=FALSE)
  cat("wrote data/rds_EC_metadata.csv  dim", paste(dim(md),collapse="x"), "\n")
}
