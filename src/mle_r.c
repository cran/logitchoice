#include <R.h>
#include <Rinternals.h>

void compute_gradient(const double *restrict x, const double *restrict res, const double *restrict beta, const double *restrict lambda, const int *restrict numRows, const int *restrict numCols, const int *restrict numGroups, const int *restrict numCores, double *restrict result);

void compute_probabilities(const double *restrict x, const double *restrict beta, const int *restrict groupSizes, const int *restrict numRows, const int *restrict numCols, const int *restrict numGroups, double *restrict result);

void solver(const double *restrict x, const int *restrict y, double *restrict res, double *restrict probabilities, double *restrict beta, const double *restrict lam, const int *restrict groupSizes, const int *restrict numRows, const int *restrict numCols, const int *restrict numGroups, const double *restrict alpha, int *restrict converged, int *restrict numIters, const int *restrict maxIters, const double *restrict tol, double *restrict objValue, const int *restrict numCores);

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

