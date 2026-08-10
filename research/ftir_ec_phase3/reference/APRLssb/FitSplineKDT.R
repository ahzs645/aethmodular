## functions from Kuzmiakova, Dillner, Takahama 2016

FitSplineSegm1 <- function(x, y, df, fixed=2220, init.bound=3710, dx=-10) {
  bound <- FindBound(x, y, df, fixed, init.bound, dx)
  out <- FitSpline(x, y, df, c(fixed, bound))
  baseline <- out[["baseline"]]
  list(bounds = c(fixed,bound),
       param = out[["df"]],
       baseline = baseline,
       absorbance = y-baseline)
}

FitSplineSegm2 <- function(x, y, df, fixed=1820, interval=c(1600, 1520)) {
  ## if(diff(x[1:2]) > 0) {
  ##   desc <- order(x,decreasing=TRUE)
  ##   x <- x[desc]
  ##   y <- y[desc]
  ## }
  if(fixed < max(interval) || diff(x[1:2]) > 0) {
    stop("check FitSplineSegm2 definition")
  }
  imin <- FindMinPos(x, y, interval)
  bound <- x[imin]
  index <- seq(length(x))
  h <- head(index,imin)  # head
  r <- tail(index,-imin) # rest
  out <- FitSpline(x[h], y[h], df, c(fixed, bound))
  baseline <- c(out[["baseline"]],y[r])
  list(bounds = unname(c(fixed,bound)),
       param = out[["df"]],       
       baseline = baseline,
       absorbance = y-baseline)
}

FitSplineSegm2r <- function(x, y, df, fixed=1820, interval=c(1600, 1520), endpoints=c(2000, 1500)) {
  ## if(diff(x[1:2]) > 0) {
  ##   desc <- order(x,decreasing=TRUE)
  ##   x <- x[desc]
  ##   y <- y[desc]
  ## }
  if(fixed < max(interval) || diff(x[1:2]) > 0) {
    stop("check FitSplineSegm2r definition")
  }
  y[] <- RotateTrans(x, y, range(which(MakeMask(x, endpoints)))) # need rotation to define minimum point
  imin <- FindMinPos(x, y, interval)
  bound <- x[imin]
  index <- seq(length(x))
  h <- head(index,imin)  # head
  r <- tail(index,-imin) # rest
  out <- FitSpline(x[h], y[h], df, c(fixed, bound))
  baseline <- c(out[["baseline"]],y[r])
  list(bounds = unname(c(fixed,bound)),
       param = out[["df"]],       
       baseline = baseline,
       absorbance = y-baseline)
}

FitSplineSegm2ext <- function(x, y, df, fixed=1820, interval=c(1600, 1520), n.ext=20) {
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
