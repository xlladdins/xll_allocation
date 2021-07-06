// fms_allocation.h - Efficient portfolio allocation
// https://arxiv.org/pdf/2009.10852.pdf
#pragma once
#include <valarray>
#include "fms_blas/fms_blas.h"

namespace fms::allocation {

	template<class X = double>
	class covariance {
		blas::matrix<X> C;
		std::valarray<X> a;
	public:
		covariance(int n, const X* rho, const X* sigma = nullptr)
			: C(n, n), a(n * n)
		{
			C = blas::matrix<X>(n, n, &a[0]);

			// correlations
			C(0, 0) = 1;
			for (int i = 1; i < n; ++i) {
				C(i, i) = 1;
				for (int j = 0; j < i; ++j) {
					C(i, j) = rho[j + (i * (i + 1)) / 2];
				}
			}

			C.lower();
			lapack::potrf(C); // rows are unit vector correlations

			if (sigma) {
				//blas::trmm(C, blas::matrix<X>(n, 1, sigma));
			}

		}
	};

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