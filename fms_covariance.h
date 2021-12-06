// fms_covariance.h - covariance matrix
#pragma once
#include "fms_blas/fms_blas.h"

namespace fms {

	// mean of columns of X
	// X is n x m, EX has size m
	inline blas::vector<double> mean(int n, int m, const double* X, double* EX)
	{
		double _n = 1. / n;
		blas::vector<const double> n_(n, &_n, 0); // n_ = {1/n,1/n, ...}
		for (int j = 0; j < m; ++j) {
			EX[j] = blas::dot(blas::vector(n, X + j, m), n_);
		}
		
		return blas::vector(m, EX);
	}

	// Cov(X,X) = E[X'X] - E[X]'E[X]
	// X is n x m, Cov(X, X) is m x m
	inline blas::matrix<double> covariance(int n, int m, const double* X, double* CovX, double* EX = nullptr)
	{
		blas::matrix<double> x(n, m, const_cast<double*>(X));
		blas::matrix<double> XX = blas::gemm(x.transpose(), x, CovX, 1./n);
		if (EX) {
			// CovX -= EX EX'
			//blas::syr<double>(CblasLower, -1, mean(n, m, X, EX), XX);
			mean(n, m, X, EX);
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					XX(i, j) -= EX[i] * EX[j];
				}
			}
		}

		return XX;
	}
	/*
	template<class X = double>
	class covariance {
		blas::matrix<X> C;
		std::valarray<X> a;
	public:
		// packed correlation [rho_11 | rho_21, rho v_22 | ... ]
		covariance(int n, const X* rho, const X* sigma = nullptr)
			: C(n, n), a(n* n)
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
	*/


} // namespace fms
