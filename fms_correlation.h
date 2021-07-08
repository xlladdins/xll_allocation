// fms_correlation.h - correlation matrix
#pragma once
#include <valarray>
#include "fms_blas/fms_lapack.h"

namespace fms {

	static inline const char correlation_doc[] = R"xyzyx(
Unit vectors \((e_i)\) determine a correlation matrix \([\rho_{ij}]\)
where \(\rho_{ij} = e_i\cdot e_j\) is the inner product of the vectors.
For every correlation matrix we can find unit vectors that satisfy this
using the Cholesky decomposition of the correlation matrix. The rows
of the lower decompsition are the \((e_i\).
)xyzyx";
	template<class X = double>
	class correlation : public blas::matrix<X> {
		std::valarray<X> _e; // lower Cholesky factor
	public:
		using blas::matrix<X>::operator();
		using blas::matrix<X>::resize;
		using blas::matrix<X>::row;
		using blas::matrix<X>::rows;
		using blas::matrix<X>::transpose;

		static inline const char doc[] = R"xyzyx(
		Packed rows of the correlation matrix without the unit diagonal.
)xyzyx";
		static inline const char* arguments[] = {
			"is the dimension of the correlation matrix",
			"is a pointer to an array of n*(n-1)/2 packed correlations",
		};
		correlation(int n, const X* rho)
			: blas::matrix<X>(n, n), _e(n * n)
		{
			resize(n, n, &_e[0]);

			operator()(0, 0) = X(1);
			for (int i = 1; i < n; ++i) {
				operator()(i, i) = X(1);
				for (int j = 0; j < i; ++j) {
					operator()(i, j) = rho[(i*(i - 1))/2 + j];
				}
			}

			lapack::potrf<X>(*this); // lower
		}
		correlation(const correlation& rho)
			: blas::matrix<X>(rho.rows(), rho.columns()), _e(rho.size())
		{ 
			_e = rho._e;
			resize(rho.rows(), rho.columns(), &_e[0]);
		}
		correlation& operator=(const correlation& rho)
		{
			if (this != &rho) {
				_e = rho._e;
				resize(rho.rows(), rho.columns(), &_e[0]);
			}

			return *this;
		}
		~correlation()
		{ }

		static inline const char dimension_doc[] = R"xyzyx(
		Return the dimension of the correlation matrix.
)xyzyx";
		int dimension() const
		{
			return rows();
		}

		static inline const char get_doc[] = R"xyzyx(
		Return the underlying correlation matrix in preallocated array.
)xyzyx";
		// 
		void get(X* cor) const
		{
			blas::gemm(*this, transpose(), cor);
		}

		static inline const char rho_doc[] = R"xyzyx(
		Return the i, j correlation.
)xyzyx";
		X rho(int i, int j) const
		{
			return blas::dot(row(i), row(j));
		}
		 
		static inline const char rho_set_doc[] = R"xyzyx(
		Set i, j correlation to r_ij by rotating e_i along e_j.
)xyzyx";
		correlation& rho(int i, int j, X r_ij)
		{
			static X eps = sqrt(std::numeric_limits<X>::epsilon());

			if (i == j) {
				return *this;
			}
			// so we can rotate back from rho = 0 or rho = 1
			if (fabs(r_ij) < eps) {
				r_ij = eps; // ??? sign, jitter
			}
			else if (fabs(1 - r_ij) < eps) {
				r_ij = 1 - eps;
			}

			// e_i' = c e_i + s e_j
			// r_ij = e_i' . e_j = c rho + sqrt(1 - c^2)
			// (rij - c rho)^2 = 1 - c^2
			// (1 + rho^2)c^2 - 2 rij rho + rij^2 - 1 = 0
			X rho_ij = rho(i, j);
			X A = 1 + rho_ij * rho_ij;
			X B = - r_ij * rho_ij;
			X C = r_ij * r_ij - 1;
			X D = B * B - A * C;
			// largest cos gives smallest theta
			X c = fabs(A) > eps ? (-B + copysign(sqrt(D), A)) / A : C / (B + copysign(sqrt(D), B));
			X s = sqrt(1 - c * c);

			blas::scal<X>(c, row(i));
			blas::axpy<X>(s, row(j), row(i));

			return *this;
		}
	};

} // namespace fms
