#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <R.h>
#include <Rinternals.h>
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

SEXP R_compute_gradient(SEXP R_x, SEXP R_res, SEXP R_beta, SEXP R_lambda, SEXP R_numRows, SEXP R_numCols, SEXP R_numGroups, SEXP R_numCores, SEXP R_result) {
  PROTECT(R_x = coerceVector(R_x, REALSXP));
  PROTECT(R_res = coerceVector(R_res, REALSXP));
  PROTECT(R_beta = coerceVector(R_beta, REALSXP));
  PROTECT(R_lambda = coerceVector(R_lambda, REALSXP));
  PROTECT(R_numRows = coerceVector(R_numRows, INTSXP));
  PROTECT(R_numCols = coerceVector(R_numCols, INTSXP));
  PROTECT(R_numGroups = coerceVector(R_numGroups, INTSXP));
  PROTECT(R_numCores = coerceVector(R_numCores, INTSXP));
  PROTECT(R_result = coerceVector(R_result, REALSXP));
  double *restrict x = REAL(R_x);
  double *restrict res = REAL(R_res);
  double *restrict beta = REAL(R_beta);
  double *restrict lambda = REAL(R_lambda);
  int *restrict numRows = INTEGER(R_numRows);
  int *restrict numCols = INTEGER(R_numCols);
  int *restrict numGroups = INTEGER(R_numGroups);
  int *restrict numCores = INTEGER(R_numCores);
  double *restrict result = REAL(R_result);
  compute_gradient(x, res, beta, lambda, numRows, numCols, numGroups, numCores, result);
  UNPROTECT(9);
  return R_result;
}

int check_convergence(const double *restrict beta, const double *restrict betaOld, const int len, const double tol) {
  int i;
  for (i=0; i<len; i++) {
    if (fabs((beta[i] - betaOld[i]) / beta[i]) > tol) {
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
    
SEXP R_compute_probabilities(SEXP R_x, SEXP R_beta, SEXP R_groupSizes, SEXP R_numRows, SEXP R_numCols, SEXP R_numGroups, SEXP R_result) {
  PROTECT(R_x = coerceVector(R_x, REALSXP));
  PROTECT(R_beta = coerceVector(R_beta, REALSXP));
  PROTECT(R_groupSizes = coerceVector(R_groupSizes, INTSXP));
  PROTECT(R_numRows = coerceVector(R_numRows, INTSXP));
  PROTECT(R_numCols = coerceVector(R_numCols, INTSXP));
  PROTECT(R_numGroups = coerceVector(R_numGroups, INTSXP));
  PROTECT(R_result = coerceVector(R_result, REALSXP));
  double *restrict x = REAL(R_x);
  double *restrict beta = REAL(R_beta);
  int *restrict groupSizes = INTEGER(R_groupSizes);
  int *restrict numRows = INTEGER(R_numRows);
  int *restrict numCols = INTEGER(R_numCols);
  int *restrict numGroups = INTEGER(R_numGroups);
  double *restrict result = REAL(R_result);
  compute_probabilities(x, beta, groupSizes, numRows, numCols, numGroups, result);
  UNPROTECT(7);
  return R_result;
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
    //check convergence
    if (check_convergence(beta, betaOld, p, tolerance)) {
      *converged = 1;
      break;
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

SEXP R_solver(SEXP R_x, SEXP R_y, SEXP R_res, SEXP R_probabilities, SEXP R_beta, SEXP R_lam, SEXP R_groupSizes, SEXP R_numRows, SEXP R_numCols, SEXP R_numGroups, SEXP R_alpha, SEXP R_converged, SEXP R_numIters, SEXP R_maxIters, SEXP R_tol, SEXP R_objValue, SEXP R_numCores) {
  PROTECT(R_x = coerceVector(R_x, REALSXP));
  PROTECT(R_y = coerceVector(R_y, INTSXP));
  PROTECT(R_res = coerceVector(R_res, REALSXP));
  PROTECT(R_probabilities = coerceVector(R_probabilities, REALSXP));
  PROTECT(R_beta = coerceVector(R_beta, REALSXP));
  PROTECT(R_lam = coerceVector(R_lam, REALSXP));
  PROTECT(R_groupSizes = coerceVector(R_groupSizes, INTSXP));
  PROTECT(R_numRows = coerceVector(R_numRows, INTSXP));
  PROTECT(R_numCols = coerceVector(R_numCols, INTSXP));
  PROTECT(R_numGroups = coerceVector(R_numGroups, INTSXP));
  PROTECT(R_alpha = coerceVector(R_alpha, REALSXP));
  PROTECT(R_converged = coerceVector(R_converged, INTSXP));
  PROTECT(R_numIters = coerceVector(R_numIters, INTSXP));
  PROTECT(R_maxIters = coerceVector(R_maxIters, INTSXP));
  PROTECT(R_tol = coerceVector(R_tol, REALSXP));
  PROTECT(R_objValue = coerceVector(R_objValue, REALSXP));
  PROTECT(R_numCores = coerceVector(R_numCores, INTSXP));
  double *restrict x = REAL(R_x);
  int *restrict y = INTEGER(R_y);
  double *restrict res = REAL(R_res);
  double *restrict probabilities = REAL(R_probabilities);
  double *restrict beta = REAL(R_beta);
  double *restrict lam = REAL(R_lam);
  int *restrict groupSizes = INTEGER(R_groupSizes);
  int *restrict numRows = INTEGER(R_numRows);
  int *restrict numCols = INTEGER(R_numCols);
  int *restrict numGroups = INTEGER(R_numGroups);
  double *restrict alpha = REAL(R_alpha);
  int *restrict converged = INTEGER(R_converged);
  int *restrict numIters = INTEGER(R_numIters);
  int *restrict maxIters = INTEGER(R_maxIters);
  double *restrict tol = REAL(R_tol);
  double *restrict objValue = REAL(R_objValue);
  int *restrict numCores = INTEGER(R_numCores);
  solver(x, y, res, probabilities, beta, lam, groupSizes, numRows, numCols, numGroups, alpha, converged, numIters, maxIters, tol, objValue, numCores);
  UNPROTECT(17);
  SEXP result = PROTECT(allocVector(VECSXP, 6));
  SET_VECTOR_ELT(result, 0, R_beta);
  SET_VECTOR_ELT(result, 1, R_res);
  SET_VECTOR_ELT(result, 2, R_probabilities);
  SET_VECTOR_ELT(result, 3, R_objValue);
  SET_VECTOR_ELT(result, 4, R_converged);
  SET_VECTOR_ELT(result, 5, R_numIters);
  const char *names[6] = {"betahat", "res", "yhat", "objValue", "converged", "numIters"};
  SEXP sNames = PROTECT(allocVector(STRSXP, 6));
  int i;
  for (i=0; i<6; i++) {
    SET_STRING_ELT(sNames, i, mkChar(names[i]));
  }
  setAttrib(result, R_NamesSymbol, sNames);
  UNPROTECT(2);
  return result;
}
