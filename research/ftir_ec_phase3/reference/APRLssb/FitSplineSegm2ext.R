FitSplineSegm2ext <- function(x, y, df, fixed=1820, interval=c(1600, 1520), n.ext=20) {
  ## if(diff(x[1:2]) > 0) {
  ##   desc <- order(x,decreasing=TRUE)
  ##   x <- x[desc]
  ##   y <- y[desc]
  ## }
  if(fixed < max(interval) || diff(x[1:2]) > 0) {
    stop("check FitSplineSegm2ext definition")
  }
  imin <- FindMinPos(x, y, interval)
  bound <- x[imin]
  index <- seq(length(x))
  h <- head(index,imin+n.ext)  # head
  r <- tail(index,-(imin+n.ext)) # rest
  out <- FitSpline(x[h], y[h], df, c(fixed, bound))
  baseline <- c(out[["baseline"]],y[r])
  list(bounds = c(fixed,bound),
       param = out[["df"]],       
       baseline = baseline,
       absorbance = y-baseline)
}
devtools::use_data(FitSplineSegm2ext)
