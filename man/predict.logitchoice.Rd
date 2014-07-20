\name{predict.logitchoice}
\alias{predict.logitchoice}
\title{Make predictions from a "logitchoice" object.}
\description{
Similar to other predict methods, this function returns fitted values on
the response scale.
}
\usage{
\method{predict}{logitchoice}(object, X, grouping, lambda=NULL, ...)
}

%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{Fitted \code{"logitchoice"} model object.}
  \item{X}{Matrix of new values for which to make predictions. Must have
  the same number of variables as during training the model.}
  \item{grouping}{Grouping information for choice situations as in logitchoice.}
  \item{lambda}{User input lambda sequence. Must be subset of the
    lambdas used in fitting. If NULL (the default), predict at all the
    lambdas used during fitting.}
  \item{\dots}{Not used. Other arguments to predict.}
}
\details{
If lambda is not specified, makes predictions at all the fitted lambda values. Users may provide their own lambda sequence, but this must be a subset of the values used to fit the models. 
}
\value{A matrix of predicted values, with columns corresponding to each
  fitted model.}

\author{
Michael Lim\cr Maintainer: Michael Lim \email{michael626@gmail.com}
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{logitchoice}, \code{predict.logitchoice}
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
max(abs(fit$yhat - predict(fit, X, grouping)))
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ models }
\keyword{ choice }% __ONLY ONE__ keyword per line
\keyword{ logit }
