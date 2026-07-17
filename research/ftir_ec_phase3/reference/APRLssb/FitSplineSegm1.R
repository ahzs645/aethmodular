FitSplineSegm1 <- function(x, y, df, fixed=2220, init.bound=3710, dx=-10) {
  bound <- FindBound(x, y, df, fixed, init.bound, dx)
  out <- FitSpline(x, y, df, c(fixed, bound))
  baseline <- out[["baseline"]]
  list(bounds = c(fixed,bound),
       param = out[["df"]],
       baseline = baseline,
       absorbance = y-baseline)
}
devtools::use_data(FitSplineSegm1, overwrite=TRUE)
