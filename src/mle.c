#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#ifdef _OPENMP
# include <omp.h>
#endif

static const double EPS = 1e-8;

void compute_gradient(const double *restrict x, const double *restrict res, const double *restrict beta, const double *restrict lambda, const int *restrict numRows, const int *restrict numCols, const int *restrict numGroups, const int *restrict numCores, double *restrict result) {
  int n = *numRows, p = *numCols, nGroups = *numGroups;
  int i, j, offset;
  double temp, lam = *lambda;

#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(*numCores);
#endif

# pragma omp parallel for shared(x, res, beta, lam, n, p, nGroups, result) private(i, j, offset, temp)
  for (j=0; j<p; j++) {
    offset = j * n;
    temp = 0.0;
    for (i=0; i<n; i++) {
      temp += x[offset + i] * res[i];
    }
    result[j] = -temp / nGroups + lam * beta[j];
  }
}

int check_convergence(const double *restrict gradient, const int len, const double tol) {
  int i;
  for (i=0; i<len; i++) {
    if (fabs(gradient[i]) > tol) {
      return 0;
    }
  }
  return 1;
}

void compute_linear(const double *restrict x, const double *restrict beta, const int *restrict numRows, const int *restrict numCols, double *restrict result) {
  int n = *numRows, p = *numCols;
  int i, j, offset;
  double scalar;
  for (j=0; j<p; j++) {
    scalar = beta[j];
    offset = j * n;
    for (i=0; i<n; i++) {
      result[i] += x[offset + i] * scalar;
    }
  }
}

void compute_probabilities(const double *restrict x, const double *restrict beta, const int *restrict groupSizes, const int *restrict numRows, const int *restrict numCols, const int *restrict numGroups, double *restrict result) {
  int n = *numRows, nGroups = *numGroups;
  int i, j, g, start, offset;
  double scalar;
  compute_linear(x, beta, numRows, numCols, result);
  //result now contains the linear components, so apply the exponential
  for (i=0; i<n; i++) {
    result[i] = exp(result[i]);
  }
  //now normalize by group
  i = offset = 0;
  for (g=0; g<nGroups; g++) {
    scalar = 0.0;
    offset += groupSizes[g];
    start = i;
    while (i < offset) {
      scalar += result[i];
      ++i;
    }
    for (j=start; j<i; j++) {
      result[j] /= scalar;
    }
  }
}

double compute_loglik(const int *restrict y, const double *restrict probabilities, const double *restrict beta, const double lambda, const int n, const int p, const int numGroups) {
  int i;
  double result = 0.0;
  double penalty = 0.0;
  for (i=0; i<n; i++) {
    result += y[i] <= 0 ? 0.0 : log(probabilities[i]);
  }
  for (i=0; i<p; i++) {
    penalty += beta[i] * beta[i];
  }
  return -result / numGroups + lambda * penalty / 2;
}

double compute_stepsize(const double *restrict gradient, const double *restrict gradientOld, const double *restrict beta, const double *restrict betaOld, const int gradientLength){
  int i;
  double normBeta = 0.0, normGradient = 0.0;
  for (i=0; i<gradientLength; i++){
    normBeta += (beta[i]-betaOld[i])*(beta[i]-betaOld[i]);
    normGradient += (gradient[i]-gradientOld[i])*(gradient[i]-gradientOld[i]);
  }
  return (normBeta > EPS && normGradient > EPS) ? sqrt(normBeta/normGradient) : 1.0;
}

double update_theta(const double *restrict beta, const double *restrict intermediate, const double *restrict intermediateOld, const int gradientLength, const double theta){
  int i;
  double value = 0.0;
  for (i=0; i<gradientLength; i++){
    value += (beta[i]-intermediate[i]) * (intermediate[i]-intermediateOld[i]);
  }
  return value > 0.0 ? 1.0 : theta;
}

void compute_update(const double *restrict beta, double *restrict betaUpdated, const double *restrict gradient, const double stepsize, const int len){
  int i;
  for (i=0; i<len; i++){
    betaUpdated[i] = beta[i] - stepsize * gradient[i];
  }
}

void optimize_step(const double *restrict x, const int *restrict y, const double *restrict res, double *restrict probabilities, const int *restrict groupSizes, const int n, const int p, const int nGroups, const double *restrict beta, double *restrict betaUpdated, const double *restrict gradient, double *restrict stepsize, const double alpha, const double lambda, const int maxIter){
  int i, iter = 0;
  double loglik, loglikUpdated;
  loglik = compute_loglik(y, probabilities, beta, lambda, n, p, nGroups);
  double delta, gradientTimesDelta, deltaTimesDelta;
  while (iter < maxIter){
    gradientTimesDelta = 0.0;
    deltaTimesDelta = 0.0;
    compute_update(beta, betaUpdated, gradient, *stepsize, p);
    for (i=0; i<p; i++){
      delta = log(exp(betaUpdated[i]) / exp(beta[i]));
      gradientTimesDelta += gradient[i] * delta;
      deltaTimesDelta += delta * delta;
    }
    memset(probabilities, 0, n * sizeof *probabilities);
    compute_probabilities(x, betaUpdated, groupSizes, &n, &p, &nGroups, probabilities);
    loglikUpdated = compute_loglik(y, probabilities, betaUpdated, lambda, n, p, nGroups);
    if (loglikUpdated <= loglik + gradientTimesDelta + deltaTimesDelta/(2*(*stepsize))) {
      break;
    }
    *stepsize *= alpha;
    ++iter;
  }
}

void solver(const double *restrict x, const int *restrict y, double *restrict res, double *restrict probabilities, double *restrict beta, const double *restrict lam, const int *restrict groupSizes, const int *restrict numRows, const int *restrict numCols, const int *restrict numGroups, const double *restrict alpha, int *restrict converged, int *restrict numIters, const int *restrict maxIters, const double *restrict tol, double *restrict objValue, const int *restrict numCores) {
  int n = *numRows, p = *numCols, nGroups = *numGroups, maxIter = *maxIters, iter, i;
  double lambda = *lam, tolerance = *tol, momentum, theta, thetaOld, stepsize;
  //initialize the gradient and intermediate variables
  double *restrict betaOld = malloc(p * sizeof *betaOld);
  double *restrict gradient = malloc(p * sizeof *gradient);
  double *restrict gradientOld = malloc(p * sizeof *gradientOld);
  double *restrict intermediate = malloc(p * sizeof * intermediate);
  double *restrict intermediateOld = malloc(p * sizeof *intermediate);
  memcpy(intermediate, beta, p * sizeof *beta);
  //start accelerated fista
  iter = 0;
  theta = 1.0;
  *converged = 0;
  objValue[iter] = compute_loglik(y, probabilities, beta, lambda, n, p, nGroups);
  while (iter < maxIter - 1) {
    memcpy(gradientOld, gradient, p * sizeof *gradient);
    compute_gradient(x, res, beta, lam, numRows, numCols, numGroups, numCores, gradient);
    //check convergence
    if (check_convergence(gradient, p, tolerance)) {
      *converged = 1;
      break;
    }
    memcpy(intermediateOld, intermediate, p * sizeof *intermediate);
    stepsize = iter > 0 ? compute_stepsize(gradient, gradientOld, beta, betaOld, p) : 1.0;
    optimize_step(x, y, res, probabilities, groupSizes, n, p, nGroups, beta, intermediate, gradient, &stepsize, *alpha, lambda, maxIter);
    thetaOld = update_theta(beta, intermediate, intermediateOld, p, theta);
    theta = (1+sqrt(1+4*pow(thetaOld, 2))) / 2;
    momentum = (thetaOld-1) / theta;
    /* update beta */
    memcpy(betaOld, beta, p * sizeof *beta);
    for (i=0; i<p; i++){
      beta[i] = intermediate[i] + momentum*(intermediate[i]-intermediateOld[i]);
    }
    /* update probabilities and residual */
    memset(probabilities, 0, n * sizeof *probabilities);
    compute_probabilities(x, beta, groupSizes, numRows, numCols, numGroups, probabilities);
    for (i=0; i<n; i++) {
      res[i] = y[i] - probabilities[i];
    }
    ++iter;
    objValue[iter] = compute_loglik(y, probabilities, beta, lambda, n, p, nGroups);
  }
  *numIters = iter;
  free(betaOld);
  free(gradient);
  free(gradientOld);
  free(intermediate);
  free(intermediateOld);
}
