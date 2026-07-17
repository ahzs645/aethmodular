################################################################################
##
## level1.R
## Authors: Satoshi Takahama (satoshi.takahama@epfl.ch),
##   Adele Kuzmiakova (adele.kuzmiakova@gmail.com)
## Mar 2016
##
## -----------------------------------------------------------------------------
##
## This file is part of APRLssb
##
## see LICENSE
##
################################################################################

##
## {SCRIPT_DESCRIPTION}
##

## -----------------------------------------------------------------------------


#' FitSpline
#'
#' Fit smoothing spline
#'
#' @param x [numeric] wavenumber
#' @param y [numeric] spectrum
#' @param df [numeric, length=1] effective degrees of freedom
#' @param interval [numeric, length=2], bounds
#' @param weights [numeric] computed from \code{x} and \code{interval}
#'
#' @return [list] fitted baseline and model parameters
#' @export

FitSpline <- function(x, y, df, interval, weights=ifelse(MakeMask(x, interval), 0, 1)) {
  ## spline
  out <- smooth.spline(x, y = y, w = weights, df = df, all.knots=TRUE)
  c(list(baseline=fitted(out)), out[c("df","lambda")])
}

#' @describeIn FitSpline \code{FitBl} is an alias for \code{FitBaseline}
#' @export
FitBl <- FitSpline

#' MakeMask
#'
#' Generate mask indicating binary classification. Different from findInterval because exclusive bounds are used.
#'
#' @param x [numeric] wavenumber
#' @param interval [numeric, length=2] endpoints
#'
#' @return [logical] vector of membership
#' @export

MakeMask <- function(x, interval)
  x > min(interval) & x < max(interval)

#' RotateTrans
#'
#' Apply rotation and translation to spectrum.
#'
#' @param x [numeric] wavenumber
#' @param y [numeric] spectrum
#' @param interval [numeric, length=2] endpoints
#'
#' @return [numeric] transformed spectrum
#' @export

RotateTrans <- function(x, y, interval=c(1, length(x))) {

  M <- cbind(x, y)
  a <- min(interval)
  b <- max(interval)
  ##
  alpha <- -atan((M[a,2]-M[b,2])/(M[a,1]-M[b,1]))
  rotm <- matrix(c(cos(alpha),sin(alpha),-sin(alpha),cos(alpha)),ncol=2)
  M1 <- sweep(M,2,M[a,],`-`)#t(t(M)-M[a,])
  M2 <- M1 %*% t(rotm)#t(rotm %*% t(M1))
  M2[, 2]
}

#' FindMinPos
#'
#' Iteratively find W1
#'
#' @param x [numeric] wavenumber
#' @param y [numeric] spectrum
#' @param p [numeric] parameter
#' @param fixed [numeric] fixed bound
#' @param init.bound [numeric] initial value of bound to be found by iteration
#' @param dx [numeric] increment by which iteration should be sought
#'
#' @return [numeric] value of bound
#' @export


FindBound <- function(x, y, p, fixed, init.bound, dx=diff(x)[1]) {

  SumAnalyte <- function(bound) {
    ## lexically scoped: x, y, df, fixed
    baseline <- FitBl(x, y, p, c(fixed,bound))[["baseline"]]
    analyte <- y - baseline
    sum(analyte[MakeMask(x,bound+c(0,dx))], na.rm = TRUE)
  }

  bound <- init.bound
  abs.sum <- SumAnalyte(bound)

  while (abs.sum < 0){
    bound <- bound + dx
    abs.sum <- SumAnalyte(bound)
  }

  bound
}

#' FindMinLoc
#'
#' Find minimum location
#'
#' @param x [numeric] wavenumber
#' @param y [numeric] spectrum
#' @param interval [numeric, length=2] bounds
#'
#' @return [numeric] a value
#' @export

FindMinLoc <- function(x, y, interval){
  ix <- MakeMask(x,interval)
  x[ix][which.min(y[ix])]
}

#' FindMinPos
#'
#' Find minimum position
#'
#' @param x [numeric] wavenumber
#' @param y [numeric] spectrum
#' @param interval [numeric, length=2] bounds
#'
#' @return [integer] a positional index
#' @export


FindMinPos <- function(x, y, interval){
  ix <- MakeMask(x,interval)
  seq(length(y))[ix][which.min(y[ix])]
}

#' ComputeNAFSample
#'
#' Compute Negative Analyte Fraction
#'
#' @param x [numeric] wavenumber
#' @param y [numeric] absorbance
#' @param interval [numeric, length=2] mask bounds
#'
#' @return [numeric] A single value that characterizes the NAF for the sample
#' @export


ComputeNAFSample <- function(x, y, interval) {
  mask <- ifelse(MakeMask(x, interval), 1, 0)
  analyte <- y*mask
  Vecnorm(analyte[analyte < 0],"1")/Vecnorm(analyte,"1")
}

