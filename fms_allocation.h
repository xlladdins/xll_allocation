// fms_allocation.h - Efficient portfolio allocation
// https://arxiv.org/pdf/2009.10852.pdf
#pragma once
#include <valarray>
#include "fms_blas/fms_lapack.h"

namespace fms::allocation {

	static inline const char fms_allocation_correlation_doc[] = R"xyzyx(
Unit vectors \((e_i)\) determine a correlation matrix \([\rho_{ij}]\)
where \(\rho_{ij} = e_i\cdot e_j\) is the inner product of the vectors.
For every correlation matrix we can find unit vectors that satisfy this
using the Cholesky decomposition of the correlation matrix. The rows
of the lower decompsition are the \((e_i\).
)xyzyx";
	template<class X = double>
	class correlation : public blas::matrix<X,CblasNoTrans,CblasLower> {
		std::valarray<X> _e; // lower Cholesky factor
	public:
		static inline const char fms_allocation_correlation_correlation_doc[] = R"xyzyx(
		Packed rows of the correlation matrix without the unit diagonal.
)xyzyx";
		static inline const char* fms_allocation_correlation_correlation_arguments[] = {
			"is the dimension of the correlation matrix",
			"is a pointer to an array of n*(n-1)/2 packed correlations",
		};
		correlation(size_t n, const X* rho)
			: blas::matrix<X>(n, n), _e(n * n)
		{ 
			using blas::matrix<X>::operator();

			blas::matrix<X>::data() = &_e[0];

			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < i; ++j) {
					operator()(i, j) = rho[j + (i * (i + 1)) / 2];
				}
				operator()(i, i) = X(1);
			}

			lapack::potrf(*this);
		}
		// i-th unit correlation vector
		blas::vector<X> row(int i) const
		{
			return blas::matrix<X>::row(i);
		}
		// get i, j correlation
		X rho(int i, int j) const
		{
			return blas::dot(row(i), row(j));
		}
		// set i, j correlation to r_ij by rotating e_i
		correlation& rho(int i, int j, X rij)
		{
			static constexpr X eps = sqrt(std::numeric_limits<X>::epsilon());

			// so we can rotate back from 0
			if (fabs(rij) < eps) {
				rij = eps; // ??? sign, jitter
			}

			// e_i' = c e_i + s e_j
			// rij = e_i' . e_j = c rho + sqrt(1 - c^2)
			// (rij - c rho)^2 = 1 - c^2
			// (1 - rho^2)c^2 + 2 rij rho + 1 - rij^2 = 0
			X rho = rho(i, j);
			X A = 1 - rho * rho;
			X B = rij * rho;
			X C = 1 - rij * rij;
			X D = B*B - A*C;
			X c = (-B + sqrt(D))/A;

			blas::scal<X>(c, row(i));
			blas::axpy<X>(sqrt(1 - c * c), row(j), row(i));

			return *this;
		}
	};

	template<class X = double>
	class covariance {
		blas::matrix<X> C;
		std::valarray<X> a;
	public:
		// packed correlation [rho_11 | rho_21, rho v_22 | ... ]
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