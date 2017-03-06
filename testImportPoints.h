#ifndef TEST_IMPORT_POINTS_H
#define TEST_IMPORT_POINTS_H

#include <armadillo>
      
void testImportPoints(const arma::mat& inK, const arma::mat& inKS, const arma::mat& inKreg,
	const arma::mat& y, const arma::mat& z,	const arma::vec& inPoints,
	const arma::vec& S, double lambda,	const arma::mat& t, const arma::mat& ihs,
	const arma::mat& alphaNew, int tempInt, double lastL,
	int& bestN, double& fval_bestN);

#endif
