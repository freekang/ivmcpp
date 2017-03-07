#include "ivm.h"

#include <climits>

#include "testImportPoints.h"

arma::uvec std_setdiff(arma::uvec& x, arma::uvec& y)
{

  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;

  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));

  return arma::conv_to<arma::uvec>::from(out);
}

arma::uvec rand_sample(const arma::uvec& src, int num)
{
    // Generate a vector with all possible numbers and shuffle it.
    arma::uvec result(src.n_elem);
    for (int i=0; i <src.n_elem;++i)
    {
        result(i) = i;
	}
    std::random_shuffle(result.begin(), result.end());

    // Truncate to the requested size.
    result.resize(num);
    
    return src.elem(result);
}

void compute_kernel(const arma::mat& x, const arma::mat& c, double sigma, arma::mat& outX)
{
	arma::uword dx = x.n_rows;
	arma::uword nx = x.n_cols;
	arma::uword dc = c.n_rows;
	arma::uword nc = c.n_cols;
	arma::mat x2 = arma::sum(arma::square(x), 0);
	arma::mat c2 = arma::sum(arma::square(c), 0);
	//x2.print("x2");
	//c2.print("c2");
	
	arma::mat distance2 = arma::repmat(c2, nx, 1)+arma::repmat(arma::trans(x2), 1, nc) - 2.0*(arma::trans(x)*c);
	//distance2.print("distance2");
	outX = arma::exp(-distance2/(2.0*sigma*sigma));
}

void computeKernel(const arma::mat& phi, const arma::mat& phiQ, double sigma, arma::mat& kernelMat)
{

	/* kernel parameter */

	/* read phi and convert to MatrixXD */	
	int D = phi.n_rows;
	int N = phi.n_cols;

	/* read phiq and convert to MatrixXD */	  
	int N2 = phiQ.n_rows;
	int S = phiQ.n_cols;

	/* compute kernel matrix */
	kernelMat.set_size(N, S); 

	for (int colK2 = 0; colK2 < S; ++colK2) {
		for (int colK1 = 0; colK1 < N; ++colK1) {
			kernelMat(colK1, colK2) = std::exp(-arma::norm((phi.col(colK1) - phiQ.col(colK2)), 2) / (2.0 * sigma * sigma));
		}
	}
}
void test_computeKernel()
{
	arma::mat x(2, 30);
	for(int i=0;i<x.n_rows;++i)
	{
		for(int k=0;k<x.n_cols;++k)
		{
			x(i, k) = 2*k+1+i;
		}
	}
	arma::mat c = x;
	x.print("x");
	double sigma = 0.25;
	arma::mat outX;
	compute_kernel(x, c, sigma, outX);
	outX.print("outX");
}

void init_params(ivm_params_t& params)
{
	params.Nadd = 150;
	params.output = 0;
	params.maxIter = std::numeric_limits<int>::max();
	params.epsilon = 0.001;
	params.tempInt = 0.95;
	//params.deselect = 0;
	params.CV = 5;
}

void check_input(ivm_params_t& params)
{
	if(params.maxIter)
	{
		params.maxIter = std::numeric_limits<int>::max();
	}
	if(params.Nadd)
	{
		params.Nadd = std::numeric_limits<unsigned int>::max();
	}
	
	params.C = 3;
}

// description: convert labels to target matrix
void labels2targets(const arma::icolvec& labels, int C, arma::imat& t)
{
// number of data
	int N = labels.n_rows;
	t.set_size(C, N);
	t.fill(0);
	for(arma::uword i=0;i<N;++i)
	{
		t((labels(i)-1), i) = 1;
	}
}

void field2Mat(const arma::field<arma::mat>& fld, arma::mat& outMat)
{
	arma::mat tmp;
	for(arma::uword i=0;i<fld.n_rows;++i)
	{
		arma::mat rtmp;
		for(arma::uword j=0;j<fld.n_cols;++j)
		{
			rtmp = arma::join_rows(rtmp, fld(i, j));
		}
		tmp = arma::join_cols(tmp, rtmp);
	}
	outMat = tmp;
}

void transposeField(const arma::field<arma::mat>& fld, arma::field<arma::mat>& out)
{
	out.set_size(fld.n_cols, fld.n_rows);
	for(arma::uword i=0;i<fld.n_rows;++i)
	{
		for(arma::uword j=0;j<fld.n_cols;++j)
		{
			out(j, i) = fld(i, j);
		}
	}
}

void getProbabilities(const arma::mat& alpha, const arma::mat& KS, double maxV, double minV, arma::mat& y)
{
	int C = alpha.n_cols;
	int N = KS.n_rows;
	arma::mat expAlphaKs = arma::zeros<arma::mat>(C, N);
	arma::mat maxVmat(C, N);
	maxVmat.fill(maxV);
	arma::mat minVmat(C, N);
	minVmat.fill(minV);
	for(int k=1;k<=C;++k)
	{
		arma::mat a = arma::trans(alpha.col(k-1));
		arma::mat b = arma::trans(KS);
		expAlphaKs.row(k-1) = (arma::exp(a*b));
	}
	arma::mat tmp1 = arma::repmat(arma::sum(expAlphaKs, 0), C, 1);
	//tmp1.print("tmp1");
	//arma::mat tmp2 = expAlphaKs / tmp1;
	//arma::mat tmp3 = arma::min(tmp2, maxVmat);
	//arma::mat tmp4 = arma::max(tmp3, minVmat);
	//tmp2.print("tmp2");
	//tmp3.print("tmp3");
	//tmp4.print("tmp4");
	y = arma::max(arma::min(expAlphaKs / tmp1, maxVmat), minVmat);
}

void ivm_learn(const arma::mat& phi, const arma::icolvec& c, const ivm_params_t& par, ivm_model_t& model)
{
	ivm_params_t params = par;
	params.tempInt = 1;
	params.epsilonBack = 10^-6;
	
	check_input(params);
	
	double maxV = 1 - 1e-8;
	double minV = 1e-8;

    model.indTrain = arma::regspace<arma::uvec>(1, 1, phi.n_cols);
    double lambda = *params.lambda;
	double sigma = *params.sigma;

	unsigned int N = phi.n_cols;
	
	int C = *params.C;
	
	params.Nadd = std::min(N, *params.Nadd);
	
// targets
	arma::imat t;
	labels2targets(c, C, t);
	
// initialization
	arma::uvec S = arma::zeros<arma::uvec>(0);        // empty subset S
	arma::mat KS = arma::zeros<arma::mat>(N, 0);       // kernel matrix of the subset S
	arma::mat Kreg = arma::zeros<arma::mat>(0,0);      // regularization matrix of the subset S
	arma::mat alpha = arma::zeros<arma::mat>(0, C);    // parameters
	arma::mat y = 1.0 / C * arma::ones<arma::mat>(C, N); // probabilities
	arma::field<arma::sp_mat> W(C, 1);
	arma::mat z = arma::zeros<arma::mat>(N, C);
	arma::field<arma::mat> invHess(1, C);
	arma::uvec all_delIV;
//arma::mat phi=arma::reshape(arma::regspace<arma::uvec>(1, 1, 60), 2, 30);
	arma::mat K;
	compute_kernel(phi, phi, sigma, K);

// %% pre-computations
// % weighting matrix
	for(int k=1;k<=C;++k)
	{
		arma::mat tt = arma::trans(y.row(k-1) % (1 - y.row(k-1)));
		arma::sp_mat sp(N, N);
		sp.diag(0) = tt.col(0);
		//sp.print("sp");
		W(k-1, 0)  = sp;
	}

//% training error
	arma::urowvec idx_train = arma::index_max(y, 0);
	arma::colvec val_train = y.elem(idx_train);
	
	//c.print("c");
	//arma::trans(idx_train).print("idx_train'");

	arma::uvec errIdx = arma::find(c != arma::trans(idx_train+1));
	//errIdx.print("errIdx");
	int err_train = errIdx.n_elem;
	
	double curr_reg;
	double lastL;
	if(S.n_elem>1)
	{
		curr_reg = lambda / 2.0 * arma::as_scalar((arma::sum(arma::sum(arma::trans(alpha) * Kreg % arma::trans(alpha), 0), 0)));
		lastL = -1.0 / N * arma::as_scalar(arma::sum(arma::sum(t % arma::log(y), 0), 0)) + curr_reg;
	}
	else
	{
		curr_reg = 0.0;
		lastL = std::numeric_limits<double>::infinity();
		
	}

	arma::mat y_store = y;
	arma::field<arma::sp_mat> W_store = W; 
	double fval = lastL;

	arma::wall_clock timer;
	timer.tic();

	for(int Q=1;Q<=*params.maxIter;++Q)
	{
		arma::mat alpha_store;
		arma::mat KS_store;
		arma::mat Kreg_store;
		// compute z from actual alpha
		for(int k=1;k<=C;++k)
		{
			//arma::mat wk_1 = arma::conv_to<arma::mat>::from(W(k-1,0));
			//W(k-1,0).print("W(k-1,0)");
			//wk_1.print("wk_1");
//			z.col(k-1) = (KS * alpha.col(k-1) + arma::inv(wk_1) * arma::trans(t.row(k-1) - y.row(k-1))) / N;
			arma::mat tmpB = arma::trans(t.row(k-1) - y.row(k-1));
			arma::vec tmpX = arma::spsolve(W(k-1,0), tmpB);
			z.col(k-1) = (KS * alpha.col(k-1) + tmpX ) / N;
			//z.col(k-1).print("z");
		}
		arma::uvec allIV = arma::regspace<arma::uvec>(1, 1, N);
		arma::uvec candidates = std_setdiff(allIV, S);
		candidates = std_setdiff(candidates, all_delIV);
		//candidates.print("candidates");
		
		arma::uvec points;
		if((*params.Nadd==std::numeric_limits<unsigned int>::max()) || (candidates.n_elem==1) )
		{
			points = candidates;
		}
		else
		{
			points = rand_sample(candidates, std::min(*params.Nadd, candidates.n_elem));
		}
		//arma::mat pints;
		//pints.load("points.dat", arma::raw_ascii);
		//pints.print("pints");
		//points=arma::conv_to<arma::uvec>::from(pints);
		//points.print("points");

				//arma::mat outMat;
				//arma::field<arma::mat> invHss(1, 3);
				//arma::mat a1("1,2;3,4");
				//arma::mat a2("11,12;13,14");
				//arma::mat a3("21,22;23,24");
				//invHss(0, 0) = a1;
				//invHss(0, 1) = a2;
				//invHss(0, 2) = a3;
				//invHss.print("invHss");
				//arma::field<arma::mat> transInvHss;
				//transposeField(invHss, transInvHss);
				//transInvHss.print("transInvHss");
				//field2Mat(transInvHss, outMat);
				//outMat.print("outMat");exit(0);
    // greedy selection
		int bestN = 0;
		if(points.n_elem>0)
		{
			arma::mat alpha_old = alpha;
			
			// select an IV if the objective function value can be
			// decreased
			//[bestN bestval] = test_importPoints(K, KS, Kreg, y', z, points, S, lambda, ...
			//			t', cell2mat(invHess'), [alpha_old; zeros(1, C)], params.tempInt, lastL);
			arma::field<arma::mat> transInvHss;
			transposeField(invHess, transInvHss);
			arma::mat outInvHessMat;
			field2Mat(transInvHss, outInvHessMat);
			arma::uvec infElem = find_nonfinite(outInvHessMat);
			if(infElem.n_elem>0)
			{
				std::cout << "warning:inf num:" << infElem.n_elem << std::endl;
			}
			outInvHessMat.elem( find_nonfinite(outInvHessMat) ).zeros();
			arma::mat alphaNew = arma::join_cols(alpha_old, arma::zeros<arma::mat>(1, C));

			double bestval;
			arma::vec dblPoints = arma::conv_to<arma::vec>::from(points);
			arma::vec dblS = arma::conv_to<arma::vec>::from(S);
			arma::mat dblT = arma::conv_to<arma::mat>::from(t);
			testImportPoints(K, KS, Kreg, arma::trans(y), z,
				dblPoints, dblS,
				lambda,	arma::trans(dblT), outInvHessMat, alphaNew, *params.tempInt, lastL,
				bestN, bestval);
				

	//bestN = 28;
			if(bestN!=0)
			{
				alpha_store = alpha;
				KS_store = KS;
				Kreg_store = Kreg;
				
				//% extend subset
				//S = [S, bestN];
				S.resize(S.n_elem+1);
				S(S.n_elem-1) = bestN;
				//std::cout << "S:" << S.n_rows << "x" << S.n_cols << std::endl;
				//S.print("S");
				
				//phi.print("phi");
				
				//% kernel matrices from new subset S
				//KS(:, end+1)   = compute_kernel(phi, phi(:, bestN), sigma);
				//Kreg(:, end+1) = compute_kernel(phi(:, S(1:end-1)), phi(:, S(end)), sigma);
				//Kreg(end+1, :) = compute_kernel(phi(:, S(end)), phi(:, S), sigma);
				
				arma::mat tmpK;
				compute_kernel(phi, phi.col(bestN-1), sigma, tmpK);
				KS.insert_cols(KS.n_cols, tmpK);
				
				if(S.n_elem>1)
				{
					compute_kernel(phi.cols(S.head(S.n_elem-1)-1), phi.cols(S.tail(1)-1), sigma, tmpK);
					Kreg.insert_cols(Kreg.n_cols, tmpK);
				}
				
				compute_kernel(phi.cols(S.tail(1)-1), phi.cols(S-1), sigma, tmpK);
				//S.print("insert rows.Kreg.S");
				//Kreg.print("insert rows.Kreg");
				//tmpK.print("insert rows.tmpK");
				Kreg.insert_rows(Kreg.n_rows, tmpK);
				
				arma::mat alpha_new = arma::zeros<arma::mat>(S.n_elem, C);
				//invHess = cell([1, C]);
				invHess.set_size(1, C);
				for(int k=1;k<=C;++k)
				{
					//% inverse Hessian         
					//invHess{k} = (1 / N * KS' * W{k} * KS + lambda * Kreg)^-1;
					
					//invHess(0, k-1) = arma::inv((1.0 / N * arma::trans(KS) * W(k-1, 0) * KS + lambda * Kreg));
					arma::mat tmpA = (1.0 / N * arma::trans(KS) * W(k-1, 0) * KS + lambda * Kreg);
					arma::mat tmpB = arma::trans(KS) * W(k-1, 0) * z.col(k-1);
					arma::vec tmpX = arma::solve(tmpA, tmpB);

					//% new parameter
					//alpha_new(:, k) = invHess{k} * KS' * W{k} * z(:, k);
					
					//alpha_new.col(k-1) = invHess(0, k-1) * arma::trans(KS) * W(k-1, 0) * z.col(k-1);
					alpha_new.col(k-1) = tmpX;
					
					//KS.print("KS");
					//W(k-1, 0).print(" W(k-1, 0)");
					//Kreg.print("Kreg");
					//invHess(0, k-1).print("invHess(0, k-1)");
					//alpha_new.col(k-1).print("alpha_new.col(k-1)");
					//z.print("z");
				}
	 
				alpha = alpha_new;
				
				//% probabilities
				getProbabilities(alpha, KS, maxV, minV, y);
				//alpha.print("alpha");
				//KS.print("KS");
				//y.print("y");
				
				//% weighting matrix
				for(int k=1;k<=C;++k)
				{
					arma::mat tt = arma::trans(y.row(k-1) % (1 - y.row(k-1)));
					arma::sp_mat sp(N, N);
					sp.diag(0) = tt.col(0);
					//sp.print("sp");
					W(k-1, 0)  = sp;
				}
				
				//% regularization parameter
				curr_reg = lambda / 2 * arma::sum(arma::sum(arma::trans(alpha) * Kreg % arma::trans(alpha)));
			}  

		}

		// negative log-likelihood 
//		fval = -1.0 / N * arma::as_scalar(arma::sum(arma::sum(t % arma::log(y), 0), 0) )+ curr_reg;
		fval = -1.0 / N * arma::as_scalar(arma::sum(arma::sum(t % arma::log(y))) )+ curr_reg;
		double fval_last = fval;

		// ratio of negative loglikelihood
		double rat = std::abs(fval - lastL) / std::abs(fval);

		// remove point in case of convergence
		if((rat<*params.epsilon) && (bestN!=0))
		{
			if(!points.empty())
			{
				//S(end) = [];
				S.resize(S.n_elem-1);
				
				// kernel matrices from subset S
				KS   = KS_store;
				Kreg = Kreg_store;
			}
			
			alpha = alpha_store;
			y = y_store;
			W  = W_store; 
			
			// training error
			//[val_train idx_train] = max(y); %#ok<*ASGLU>
			//err_train = sum(c ~= idx_train');
			arma::urowvec idx_train = arma::index_max(y, 0);
			arma::colvec val_train = y.elem(idx_train);
			arma::uvec errIdx = arma::find(c != arma::trans(idx_train+1));
			err_train = errIdx.n_elem;

			// negative log-likelihood
			curr_reg = lambda / 2 * arma::as_scalar(arma::sum(arma::sum(arma::trans(alpha) * Kreg % arma::trans(alpha))));
			fval = -1.0 / N * arma::as_scalar(arma::sum(arma::sum(t % arma::log(y)))) + curr_reg;
		}

		if(*params.output)
		{
			if(!(rat < *params.epsilon))
			{
				printf("forward: %3d: %d->%.2f %%, rat:%.4f,L:%f, #IV: %d\n",
					Q, err_train, 100.0*err_train / N, rat, fval, S.n_elem);
			}
			else
			{
				rat = std::abs(fval_last - lastL) / std::abs(fval_last);
				fval = fval_last;
				printf("forward: %3d: %d->%.2f %%, rat:%.4f,L:%f, #IV: %d\n",
					Q, err_train, 100.0*err_train / N, rat, fval_last, S.n_elem);
			}
		}
		
		if(!bestN)
		{
			if(*params.output)
			{
				std::cout << "No further import vector can be found." << std::endl;
				break;
			}
		}

		// check for convergence
		if(((rat < *params.epsilon) || (Q == *params.maxIter)) && (Q > 3))
		{
			break;
		}
		
		if((fval > (lastL * 1.3)) && (Q > 5))
		{
			break;
		}

		if(std::isnan(rat))
		{
			break;
		}

	   // check number of remaining points
		arma::uvec allNIV = arma::regspace<arma::uvec>(1, 1, N);
		arma::uvec diffNIV = std_setdiff(allIV, S);
		if(diffNIV.n_elem <= *params.Nadd)
		{
			params.Nadd = diffNIV.n_elem;
			if(*params.Nadd == 0)
			{
				// break if all training points are in the subset
				break;
			}
			if(points.n_elem == 0)
			{
				break;
			}
		}

   // negative log-likelihood
		curr_reg = lambda / 2 * arma::as_scalar(arma::sum(arma::sum(arma::trans(alpha) * Kreg % arma::trans(alpha))));
		fval = -1.0 / N * arma::as_scalar(arma::sum(arma::sum(t % arma::log(y)))) + curr_reg;

		// probabilities
		y_store = y;
		getProbabilities(alpha, KS, maxV, minV, y);

		// weighting matrix
		W_store = W;
		for(int k=1;k<=C;++k)
		{
			arma::mat tt = arma::trans(y.row(k-1) % (1 - y.row(k-1)));
			arma::sp_mat sp(N, N);
			sp.diag(0) = tt.col(0);
			//sp.print("sp");
			W(k-1, 0)  = sp;
		}

		// training error
		arma::urowvec idx_train = arma::index_max(y, 0);
		arma::colvec val_train = y.elem(idx_train);
		arma::uvec errIdx = arma::find(c != arma::trans(idx_train+1));
		err_train = errIdx.n_elem;

		invHess.set_size(1, C);
		for(int k=1;k<=C;++k)
		{
			//% inverse Hessian         
			//invHess{k} = (1 / N * KS' * W{k} * KS + lambda * Kreg)^-1;
			invHess(0, k-1) = arma::inv((1.0 / N * arma::trans(KS) * W(k-1, 0) * KS + lambda * Kreg));
		}
		lastL = fval;

//		break;
	}
	double nsec = timer.toc();

	std::cout << "number of seconds: " << nsec << std::endl;
	// return results
	model.P = y;
	model.params = params;
	model.trainError = err_train;
	model.trainTime = nsec;
	model.IV = phi.cols(S-1);
	model.kernelSigma = *params.sigma;
	model.lambda = lambda;
	model.C = C;
	model.c = c.elem(S-1);
	model.S = S;
	model.nIV = S.n_elem;
	model.alpha = alpha;
	model.fval = fval;
	
}

// IVM_PREDICT classify data with the trained Import Vector Machine model
//
//   IVM_PREDICT classify PHIT and evaluate regarding CT based on the model
//   MODEL; the result is stored in RESULT
//
//   Authors:
//     Ribana Roscher(ribana.roscher@fu-berlin.de)
//   Date: 
//     November 2014 (last modified)
void ivm_predict(const ivm_model_t& model, arma::mat& phit, arma::icolvec& ctest, ivm_result_t& result)
{
// kernel
	double maxV = 1 - 1.0e-8;
	double minV = 1.0e-8;

    arma::mat K_test;
	compute_kernel(phit, model.IV, model.kernelSigma, K_test);

	arma::mat yp;
	getProbabilities(model.alpha, K_test, maxV, minV, yp);
	result.P = yp;
	
	//phit.print("phit");
	//K_test.print("K_test");
	//result.P.print("result.P");

	if(!arma::is_finite(result.P))
	{
		std::cout << "Warning: There are nan or inf in the probabilities." << std::endl;
	}

	// store results
	result.model = model;
	result.nIV = model.IV.n_cols;
	result.Ntest = 0;
	result.oa = 0;
	result.aa = 0;
	result.aa_c.reset();
	result.kappa = 0;

// accuracy (if test labels are given)
//[val, idx] = max(result.P);
	arma::urowvec pidx = arma::index_max(result.P, 0);
	arma::colvec pval = result.P.elem(pidx);
	//pidx.print("pidx");

    // find nonzero entries
    arma::uvec ind = arma::find(ctest!=0);
    //ind.print("ind");
    arma::uvec idx = pidx.elem(ind);
    arma::icolvec ct = ctest.elem(ind);
    //ct.print("ct");
    //idx.print("idx");
    
    // confusion matrix (rows = true labels, columns = estimated labels)
    arma::mat confMat = arma::zeros<arma::mat>(model.C, model.C);
    for(int i=0;i<ct.n_elem;++i)
    {
        confMat(ct(i)-1, idx(i)) = confMat(ct(i)-1, idx(i)) + 1;
    }
    confMat.print("confMat");

    // overall accuracy
    double oa = arma::trace(confMat) / arma::accu(confMat);
    
    // class accuracy
    arma::vec aa_c = arma::diagvec(confMat) / arma::sum(confMat, 1);
    aa_c.print("aa_c");
    
    // average class accuracy
    double aa = arma::sum(aa_c) / aa_c.n_elem;

    // kappa coefficient (http://kappa.chez-alice.fr/kappa_intro.htm)
    double Po = oa;
    double accu = arma::accu(confMat);
    double Pe = arma::as_scalar(arma::sum(confMat, 0) * arma::sum(confMat, 1)) / std::pow(accu, 2.0);
    double K = (Po - Pe) / (1 - Pe);
    
    // store results
    result.confMat = confMat;
    result.confMatNorm = confMat / arma::repmat(arma::sum(confMat, 1), 1, model.C);
    result.oa = oa;
    result.aa = aa;
    result.aa_c = aa_c;
    result.kappa = K;
    result.Ntest = arma::accu(confMat);

}


