\name{logitchoice}
\alias{logitchoice}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Fit a logit choice model with l2 regularization on the coefficients.}
\description{The regularization path is computed along a grid of values
  for the regularization parameter lambda. No standardization is applied to any
  of the inputs prior to estimation. Unless the model is fit without 
  regularization (lambda = 0), we recommend that the user performs some kind
  of standardization.}
\usage{
logitchoice(X, Y, grouping, lambda=NULL, nLambda=50, lambdaMinRatio=0.01,
tol=1e-5, alpha=0.8, maxIter=5000, verbose=FALSE, numCores=1)
}

\arguments{
  \item{X}{Matrix of features or predictors with dimension nobs x nvars;
    each row is an observation vector for a particular alternative. Thus if
    there are k alternatives for an observation, this observation consists of
    k rows in \code{X}.}
  \item{Y}{Target variable of length nobs. Must contain a single 1 for each
  choice situation, so that sum(Y) = number of observations}
  \item{grouping}{Grouping information that identifies the choice situations, i.e.
  if an observation consists of k alternatives, all those k rows in X are assigned to
  the same group.}
  \item{lambda}{A user supplied \code{lambda} sequence. Typical usage
    is to have the program compute its own \code{lambda} sequence based on
    \code{nLambda} and \code{lambdaMinRatio}. Supplying a value of
    \code{lambda} overrides this.}
  \item{nLambda}{The number of \code{lambda} values. Default is 50.}
  \item{lambdaMinRatio}{Smallest value for \code{lambda}, as a fraction of
    \code{lambdaMax}. The default is \code{0.01}.}
  \item{tol}{Convergence tolerance in the adaptive FISTA algorithm.}
  \item{alpha}{Backtracking parameter in majorization-minimization scheme.}
  \item{maxIter}{Maximum number of iterations in adaptive FISTA. Default 5000.}
  \item{verbose}{Prints progress. False by default.}
  \item{numCores}{Number of threads to run. For this to work, the
    package must be installed with OpenMP enabled. Default is 1 thread.}
}
\details{
  The sequence of models implied by \code{lambda} is fit by FISTA (fast
  iterative soft thresholding) with adaptive step size and adaptive
  momentum restart.
}
\value{
  An object of class \code{logitchoice} with the components
  \item{call}{The user function call.}
  \item{betahat}{The fitted coefficients, with dimension nVars x
    \code{nLambda}.}
  \item{yhat}{The fitted values, with dimension nobs x
    \code{nLambda}.}
  \item{residual}{Residuals for each \code{lambda}.}
  \item{lambda}{The actual \code{lambda} sequence used.}
  \item{objValues}{Objective values for each lambda.}
  \item{numIters}{Number of algorithm iterations taken for each fitted \code{lambda} value.}
}

\author{Michael Lim\cr Maintainer: Michael Lim \email{michael626@gmail.com}}

\seealso{
\code{predict.logitchoice},
\code{coef.logitchoice}
}
\examples{
groupSizes = sample(6:18, 100, replace=TRUE)
numGroups = length(groupSizes)
n = sum(groupSizes)
p = 20 
X = matrix(rnorm(n*p), nrow=n)
X = scale(X)
Y = rep(0, n)
Y[cumsum(groupSizes)] = 1
grouping = rep(1:numGroups, groupSizes)
fit = logitchoice(X, Y, grouping)
max(abs(fit$yhat - predict(fit, X, grouping)))
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ models }
\keyword{ choice }% __ONLY ONE__ keyword per line
\keyword{ logit }
