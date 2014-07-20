\name{coef.logitchoice}
\alias{coef.logitchoice}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Return main effect and interaction coefficients.
}
\description{Returns the model coefficients for each lambda.
}
\usage{
\method{coef}{logitchoice}(object, lambdaIndex = NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{Fitted \code{"logitchoice"} model object.}
  \item{lambdaIndex}{Index of \code{lambda} value at which to extract
    coefficients. If NULL, return the coefficients for all values of
    \code{lambda} in the path.}
  \item{\dots}{Not used.}
}
\details{Returns the actual main effect and interaction
  coefficients. These satisfy the sum constraints in the original linear
interaction model.}
\value{A matrix of dimension nvars x length(lambdaIndex).
}

\author{Michael Lim and\cr Maintainer: Michael Lim \email{michael626@gmail.com}}

\seealso{
\code{predict.logitchoice},
\code{logitchoice}
}

\examples{
  groupSizes = sample(6:18, 100, replace=TRUE)
  numGroups = length(groupSizes)
  n = sum(groupSizes)
  p = 10
  X = matrix(rnorm(n*p), nrow=n)
  X = scale(X)
  Y = rep(0, n)
  Y[cumsum(groupSizes)] = 1
  grouping = rep(1:numGroups, groupSizes)
  fit = logitchoice(X, Y, grouping)
  stopifnot(sum(fit$yhat) == numGroups*length(fit$lambda))
max(abs(fit$betahat - coef(fit)))
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ models }
\keyword{ choice }% __ONLY ONE__ keyword per line
\keyword{ logit }