// fms_correlation.h - correlation matrix
#if 0
#pragma once
#include <valarray>
#include "fms_blas/fms_lapack.h"

namespace fms {

	static inline const char correlation_doc[] = R"xyzyx(
Unit vectors \((e_i)\) determine a correlation matrix \([\rho_{ij}]\)
where \(\rho_{ij} = e_i\cdot e_j\) is the inner product of the vectors.
For every correlation matrix we can find unit vectors that satisfy this
using the Cholesky decomposition of the correlation matrix. The rows
of the lower decomposition are the \((e_i\).
)xyzyx";
	template<class X = double>
	class correlation {
		blas::vector<X> _e; // lower Cholesky factor
		blas::tp<X> L;
	public:
		using blas::matrix<X>::operator();
		using blas::matrix<X>::row;
		using blas::matrix<X>::rows;

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
			blas::matrix<X>::a = &_e[0];

			operator()(0, 0) = X(1);
			for (int i = 1; i < n; ++i) {
				operator()(i, i) = X(1);
				for (int j = 0; j < i; ++j) {
					operator()(i, j) = rho[(i*(i - 1))/2 + j];
				}
			}

			lapack::potrf<X>(CblasLower, *this);
		}
		correlation(const correlation& rho)
			: blas::matrix<X>(rho.rows(), rho.columns()), _e(rho.size())
		{ 
			_e = rho._e;
			blas::matrix<X>::a = &_e[0];
		}
		correlation& operator=(const correlation& rho)
		{
			if (this != &rho) {
				_e = rho._e;
				blas::matrix<X>::a = &_e[0];
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
		Return the underlying correlation matrix in preallocated n x n array.
)xyzyx";
		// fill cor with correlations
		blas::matrix<X> get(X* cor) const
		{
			return blas::gemm(transpose(), *this, cor);
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
			ensure(-1 <= r_ij && r_ij <= 1);
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

			// row(i) = c row(i) + s row(j)
			blas::scal<X>(c, row(i));
			blas::axpy<X>(s, row(j), row(i));

			return *this;
		}
	};

} // namespace fms
#endif // 0