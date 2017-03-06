#ifndef IMPORT_VECTOR_MACHINE_H
#define IMPORT_VECTOR_MACHINE_H

#include <armadillo>
#include <boost/optional.hpp>

typedef struct ivm_params
{
// kernel parameter
	boost::optional<double> sigma;

// regularization parameter
	boost::optional<double> lambda;
	
// temporal filtering, 1: no filtering, 0.5: average between old and new
// parameters (params.tempInit < 1 is more stable, but converges slower)
// default: 0.95
	boost::optional<int> tempInt;

// threshold on the function value for backward selection, if an import 
// vector is removed
// default: 0.001
	boost::optional<double> epsilonBack;
	
// maximum number of iterations (maximum number of import vectors, inf
// tests all points)
// default: inf
	boost::optional<int> maxIter;

	boost::optional<int> C;
	
// maximum number of points tested for adding to the subset (inf takes all
// points)
// default: inf
	boost::optional<unsigned int> Nadd;

// stopping criterion for convergence proof
// default: 0.001
	boost::optional<double> epsilon;

// skip backward selection of import vector (computional cost descrease
// significantly)
// default = 0;
//	boost::optional<int> deselect;

// number of folds in crossvalidation
// default = 5;
	boost::optional<int> CV;

// display output (0: no output, 1: output)
// default: 0;
	boost::optional<int> output;
	
} ivm_params_t;

typedef struct ivm_model
{
	arma::uvec indTrain;
	arma::mat P;
	ivm_params_t params;
	double trainError;
	long trainTime;
	arma::mat IV;
	double kernelSigma;
	double lambda;
	int C;
	arma::icolvec c;
	arma::uvec S;
	int nIV;
	arma::mat alpha;
	double fval;
	
} ivm_model_t;

typedef struct ivm_result
{
	arma::mat P;

	ivm_model_t model;
	int nIV;
	int Ntest;
	double oa;
	double aa;
	arma::vec aa_c;
	double kappa;
	
	arma::mat confMat;
	arma::mat confMatNorm;

} ivm_result_t;

void init_params(ivm_params_t& params);
void ivm_learn(const arma::mat& phi, const arma::icolvec& c, const ivm_params_t& par, ivm_model_t& model);
void ivm_predict(const ivm_model_t& model, arma::mat& phit, arma::icolvec& ctest, ivm_result_t& result);

#endif
