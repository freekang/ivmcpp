#include <armadillo>
#include <boost/optional.hpp>
#include <climits>

#include "ivm.h"

void test_ivm_learn()
{
	arma::mat phi(2, 30);
	for(int i=0;i<phi.n_rows;++i)
	{
		for(int k=0;k<phi.n_cols;++k)
		{
			phi(i, k) = 2*k+1+i;
		}
	}

	arma::icolvec labels(30);
	labels.rows(0, 9).fill(1);
	labels.rows(10, 19).fill(2);
	labels.rows(20, 29).fill(3);
	
	ivm_params_t params;
	init_params(params);
	params.sigma = 0.25;
	params.lambda = std::exp(-5);
	
	ivm_model_t model;
	ivm_learn(phi, labels, params, model);
std::cout << "model.nIV: " << model.nIV << std::endl;
model.IV.print("model.IV");
    printf("Train Accuracy = %.1f %% (%.0f/%d)\n", 
        100 - model.trainError / labels.n_elem * 100,
        labels.n_elem - model.trainError, labels.n_elem);
	
}

void test_ivm_predict()
{
	arma::mat phi(2, 30);
	for(int i=0;i<phi.n_rows;++i)
	{
		for(int k=0;k<phi.n_cols;++k)
		{
			phi(i, k) = 2*k+61+i;
		}
	}

	arma::icolvec labels(30);
	labels.rows(0, 9).fill(2);
	labels.rows(10, 19).fill(1);
	labels.rows(20, 29).fill(3);
	
	ivm_params_t params;
	init_params(params);
	params.sigma = 0.25;
	params.lambda = std::exp(-5);
	
	phi.load("phit.mat", arma::raw_ascii);
	labels.load("ct.mat", arma::raw_ascii);
	
	
	ivm_model_t model;
	arma::mat alpha;
	alpha.load("alpha.mat", arma::raw_ascii);
	model.alpha = alpha;
	arma::mat IV;
	IV.load("iv.mat", arma::raw_ascii);
	model.IV = IV;
	
	model.kernelSigma = *params.sigma;
	model.C = 3;
	
	//phi.print("phi");
	//labels.print("labels");
	//alpha.print("alpha");
	
	ivm_result_t result;
	ivm_predict(model, phi, labels, result);
}

int main(int argc, char *argv[])
{
	//arma::mat A("1 2 3 4 5;6 7 8 9 10");
	//arma::mat B("11 12;13 14");
	//arma::mat C("21 22 23 24 25;26 27 28 29 30");
	//A.print("A");
	//B.print("B");

	//// at column 2, insert a copy of B;
	//// A will now have 12 columns
	////A.insert_cols(5, B);
	//A.insert_rows(2, C);
	//A.print("A");

	//return 0;
	
	//test_computeKernel();
	test_ivm_learn();
	test_ivm_predict();
}
//g++ -g -std=c++11 -o ivm_test ivm_test.cpp ivm.cpp testImportPoints.cpp -I/usr/include/eigen3 -larmadillo
