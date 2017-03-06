#include "testImportPoints.h"

#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
#include <fstream>
#include <sstream>
//#include "math.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
//#include "time.h"
#include <iostream>

#include <armadillo>

using namespace Eigen;

MatrixXd U;
MatrixXd V;
MatrixXd CC;
MatrixXd AinvU;
MatrixXd VAinv;
MatrixXd CVAinvU;
MatrixXd KTRKinv;

// Sherman-Morrisson Woodbury formula
void update_alpha(const MatrixXd &Ainv, const MatrixXd &KS,
                  const VectorXd &R, const VectorXd &z, 
                  const double lambda, const VectorXd &delta_K, 
                  const VectorXd &delta_Kreg, const VectorXd &delta_kreg, 
                  const int N, const int Q, VectorXd &alpha, const MatrixXd &KRZ,
                  const MatrixXd OldKNormR){

   
    // tmp matrices
    //U = 1.0/((double)N) * KS.block(0, 0, N, Q).transpose() * R.asDiagonal() * delta_K + lambda * delta_Kreg;
    U = OldKNormR * delta_K + lambda * delta_Kreg;
    V = U.transpose();
    CC = 1.0/((double)N) * delta_K.transpose() * R.asDiagonal() * delta_K + lambda * delta_kreg;
    AinvU = Ainv * U;
    VAinv = AinvU.transpose();
    CVAinvU = (CC - V * AinvU).inverse(); 
    
    KTRKinv.block(0, 0, Q, Q) = Ainv + AinvU * CVAinvU * VAinv;
    KTRKinv.block(0, Q, Q, 1) = -AinvU * CVAinvU;
    KTRKinv.block(Q, 0, 1, Q) = -CVAinvU * VAinv;
    KTRKinv.block(Q, Q, 1, 1) = CVAinvU;
    
    //alpha = KTRKinv * KS.transpose() * R.asDiagonal() * z;
    alpha = KTRKinv * KRZ;
}
 
 void Dump(const arma::mat& in, int idx, const std::string& desc)
{
	int m = in.n_rows;
	int n = in.n_cols;
	std::cout << desc << ",par=" << idx << "," << m << "x" << n << std::endl;
	for(int i=0;i<(m*n);++i)
	{
		std::cout << "\t" << i << "," << in.memptr()[i] << std::endl;
	}
}
      
void testImportPoints(const arma::mat& inK, const arma::mat& inKS, const arma::mat& inKreg,
	const arma::mat& y, const arma::mat& z,	const arma::vec& inPoints,
	const arma::vec& S, double lambda,	const arma::mat& t, const arma::mat& ihs,
	const arma::mat& alphaNew, int tempInt, double lastL,
	int& bestN, double& fval_bestN)
{
/*	Dump(inK, 0, "inK");
	Dump(inKS, 1, "inKS");
	Dump(inKreg, 2, "inKreg");
	Dump(y, 3, "y");
	Dump(z, 4, "z");
	Dump(inPoints, 5, "inPoints");

	Dump(S, 6, "S");
	std::cout << "7,lambda," << lambda << std::endl;
	//Dump(inPoints, 7, "inPoints");
	Dump(t, 8, "t");
	Dump(ihs, 9, "ihs");
	Dump(alphaNew, 10, "alphaNew");
	std::cout << "11,tempInt," << tempInt << std::endl;
	std::cout << "12,lastL," << lastL << std::endl;
*/
	// declarations
	//double* bestN;
    //double* fval_bestN;
	int p, c, q, n, col, row;
    double reg, fval, fval_best;
    float fval_float;
            
    clock_t start, end;
    double cpuTime;
      
	MatrixXd K;
	MatrixXd KS;
	MatrixXd Kreg;
    VectorXd delta_K;
    VectorXd delta_Kreg;
    VectorXd delta_kreg;        
	MatrixXd Y;
    MatrixXd YTmp;
	MatrixXd Z;
	MatrixXd R;
	MatrixXd T;
	MatrixXd onesY;
	MatrixXd KRKT;
	MatrixXd dat;
    MatrixXd dat_trans;
    MatrixXd OldAlpha;
	MatrixXd Alpha;
    VectorXd alpha;  
	MatrixXd A;
    VectorXd r;
    VectorXd sum_y;
    MatrixXd Ainv;
    MatrixXd KRZ;
    MatrixXd OldKNormR;
    
    // dimensions
    const unsigned long int N = inK.n_rows; // number of all points
    const unsigned long int Q = inKS.n_cols; // size of the subset
    const unsigned long int C = y.n_cols; // number of classes
    const unsigned long int P = inPoints.n_elem; // number of points to be tested
    
    // resize
    U.resize(Q, 1);
    V.resize(1, Q);
    CC.resize(1, 1);
    KTRKinv.resize(Q+1, Q+1);
    
    // input
	double *mxK      = const_cast<double*>(inK.memptr()); // kernel matrix
	double *mxKS     = const_cast<double*>(inKS.memptr());  //mxGetPr(prhs[1]); // subset kernel matrix
	double *mxKreg   = const_cast<double*>(inKreg.memptr()); //mxGetPr(prhs[2]); // regularization kernel matrix
	double *mxY      = const_cast<double*>(y.memptr()); //mxGetPr(prhs[3]); // probabilities
	double *mxZ      = const_cast<double*>(z.memptr()); //mxGetPr(prhs[4]); // z
	double *points   = const_cast<double*>(inPoints.memptr()); //mxGetPr(prhs[5]); // points to be tested
	double *subset   = const_cast<double*>(S.memptr()); //mxGetPr(prhs[6]); // subset
	
	double *mxtargets  = const_cast<double*>(t.memptr()); //mxGetPr(prhs[8]); // targets ??type
	double *mxAinv   = const_cast<double*>(ihs.memptr()); //mxGetPr(prhs[9]); // Hessian
    double *mxOldAlpha   = const_cast<double*>(alphaNew.memptr()); //mxGetPr(prhs[10]); // old parameters
    
//    double tempInt     = tempInt;  // temporal integration    
//for(int i=0;i<12;++i)
//{
	//std::cout << "testAinv:" << i << "=" << mxAinv[i] << std::endl;
//}    
    //if (nrhs < 13){
        //fval_best = 1e16;
    //}
    //else{
       // double *mxfval_best = mxGetPr(prhs[10]);
        fval_best   = lastL; //mxGetScalar(prhs[12]);
//    }
    
	if (y.n_rows != t.n_rows){
		printf("targets and probabilities do not have the same dimensions.");
	}
    
    // output
    //plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    //bestN = mxGetPr(plhs[0]);
    //plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    //fval_bestN = mxGetPr(plhs[1]);
    
	// convert subset kernel matrix KS to MatrixXD	
	KS.resize(N, Q + 1);
	for (col = 0; col < Q; ++col) {
		for (row = 0; row < N; ++row) {
			KS(row, col) = mxKS[(col * N) + row];
		}
	}
	
	// convert kernel matrix K to MatrixXD	
	K.resize(N, N);
	for (col = 0; col < N; ++col) {
		for (row = 0; row < N; ++row) {
			K(row, col) = mxK[(col * N) + row];
		}
	}
	
	// convert regularization kernel matrix KQ to MatrixXD
	Kreg.resize(Q + 1, Q + 1);
	for (col = 0; col < Q; ++col) {
		for (row = 0; row < Q; ++row) {
			Kreg(row, col) = mxKreg[(col * Q) + row];
		}
	}
	
    Ainv.resize(Q * C, Q);
	for (col = 0; col < Q; ++col) {
		for (row = 0; row < (Q * C); ++row) {
			Ainv(row, col) = mxAinv[(col * (Q * C)) + row];
		}
	}
    
	// convert Y and Z to MatrixXD and VectorXD	
	Y.resize(N, C);
	Z.resize(N, C);
	for (c = 0; c < C; ++c) {
		for (row = 0; row < N; ++row) {
			Y(row, c) = mxY[(c * N) + row];
			Z(row, c) = mxZ[(c * N) + row];
		}
	}
	
	// convert Y and Z to MatrixXD and VectorXD	
	T.resize(N, C);
	for (col = 0; col < C; ++col) {
		for (row = 0; row < N; ++row) {
			T(row, col) = mxtargets[(col * N) + row];
		}
	}
            
	// compute probability weights
    R.resize(N, C);
    onesY.setOnes(N, C);
    R = Y.array() * (onesY - Y).array();
     
    // precompute KRZ
    KRZ.resize(Q+1, C);
    for (int c = 0; c < C; ++c){
        KRZ.block(0, c, Q+1, 1) = KS.transpose() * R.col(c).asDiagonal() * Z.col(c);
    }
    
    // precompute OldKNormR
    OldKNormR.resize(C * Q+1, N);
    for (int c = 0; c < C; ++c){
        OldKNormR.block(c * Q, 0, Q, N) = 1.0 / N * KS.block(0, 0, N, Q).transpose() * 
            R.col(c).asDiagonal();
    }
    
    // test each points to be in the subset
    r.resize(N);
    KRKT.resize(Q + 1, Q + 1);
    dat.resize(Q + 1, N);
    dat_trans.resize(N, Q + 1);
    Alpha.resize(Q + 1, C);
    alpha.resize(Q + 1, 1);
    A.resize(C, N);
    sum_y.resize(N);
    delta_kreg.resize(1);
    bestN = 0;
    fval_bestN = fval_best;
    MatrixXd b;
    
    for (p = 0; p < P; ++p) {
        sum_y.setZero(N);
        
		// for alpha update: get new kernel matrix rows
		delta_K = K.col((int) points[p] - 1);	
        
	    // for alpha update: get parts of regularization matrix
        delta_Kreg.resize(Q);
	    for (q = 0; q < Q; ++q) {
			delta_Kreg(q) = K((int) subset[q] - 1, (int) points[p] - 1);
		}

        delta_kreg(0) = K((int) points[p] - 1, (int) points[p] - 1);
                
        // get subset kernel matrix
		KS.col(Q) = K.col((int) points[p] - 1);	

	    // get regularization matrix
	    for (q = 0; q < Q; ++q) {
			Kreg(Q, q) = K((int) points[p] - 1, (int) subset[q] - 1);
			Kreg(q, Q) = K((int) subset[q] - 1, (int) points[p] - 1);
		}
        Kreg(Q, Q) = K((int) points[p] - 1, (int) points[p] - 1);
        
        // compute paramters for all classes
		for (int c = 0; c < C; ++c){
            
            // parameters
            if (Q == 0){
                Alpha.block(0, c, Q + 1, 1) = (1.0 / N * KS.transpose() * R.col(c).asDiagonal()
                * KS + lambda * Kreg).inverse() * KS.transpose() * R.col(c).asDiagonal() * Z.col(c);
            }
            else{
                
                b = KS.col(Q).transpose() * R.col(c).asDiagonal() * Z.col(c);
                KRZ(Q, c) = b(0,0);
                update_alpha(Ainv.block(c * Q, 0, Q, Q), KS, R.col(c), Z.col(c), lambda, delta_K, 
                             delta_Kreg, delta_kreg, N, Q, alpha, KRZ.col(c), 
                             OldKNormR.block(c * Q, 0, Q, N));
                Alpha.block(0, c, Q + 1, 1) = alpha;  
            }
                
            // new activities
            A.block(c, 0, 1, N) = (Alpha.col(c).transpose() * KS.transpose()).array().exp();
			
            sum_y = sum_y + A.row(c).transpose();
		}

        
		// normalize activities yielding probabilities
        for (int c = 0; c < C; ++c){
            Y.col(c) = A.row(c).array() / sum_y.transpose().array(); //?
        }
        
        // set max and min value of probabilities
        for (int n = 0; n < N; ++n){
			for (int c = 0; c < C; ++c){
				if (Y(n, c) < 1e-8){
					Y(n, c) = 1e-8;
				}
				if (Y(n, c) > (1.0 - 1e-8)){
					Y(n, c) = 1.0 - 1e-8;
				}
			}
		}

        fval_float = 0;
        for (size_t i=0; i<T.size(); i++){
            fval_float += (float) T(i) * std::log((float) Y(i));
        }
        fval_float *= -1.0/N;
        fval_float += reg;
        fval = fval_float;
                
        // regularization value
        reg = 0.0;
        for (int c = 0; c < C; ++c){
            reg = reg + lambda / 2.0 * (Alpha.col(c).transpose() * Kreg * Alpha.col(c)).trace();
        }
     
        // objective function value
        fval = -1.0 / N * (T.array() * Y.array().log()).sum() + reg;
        
        if (fval_best >= fval) {
			bestN = points[p];
            fval_bestN = fval;
            fval_best = fval;
        }
        
	}
}
