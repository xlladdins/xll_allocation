// fms_allocation.h - Efficient portfolio allocation
// https://arxiv.org/pdf/2009.10852.pdf
#pragma once
#include <valarray>
#include "fms_blas/fms_blas.h"

namespace fms::allocation {

	class portfolio {
		int n; // dimension
		std::valarray<double> x; // initial prices
		std::valarray<double> EX; // expected future prices
		std::valarray<double> Sigma; // volatilities
		std::valarray<double> rho; // upper Cholesky correlation factor
		std::valarray<double> V_; // inverse of covariance matrix
		std::valarray<double> xi; // optimal portfolio
	public:
		portfolio(int n, const double* x, const double* EX, const double* Sigma, const double* rho)
			: n(n), x(x, n), EX(EX, n), Sigma(Sigma, n), rho(rho, n*n), V_(n * n), xi(n)
		{
			// V_ = inv(Sigma rho rho' Sigma')
			//mkl::blas::gemm(mkl::matrix(n, n, &rho[0]), mkl::matrix(n, 1, &Sigma[0]), &V_[0]);
		}
		// minimize variance given target return
		double* minimize(double R, double* X)
		{
			
			R = *X;

			return X;
		}
	};


} // namespace fms::allocation