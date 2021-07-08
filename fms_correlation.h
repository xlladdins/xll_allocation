// fms_correlation.h - correlation matrix
#pragma once
#include "fms_blas/fms_lapack.h"

namespace fms {

	static inline const char fms_correlation_doc[] = R"xyzyx(
Unit vectors \((e_i)\) determine a correlation matrix \([\rho_{ij}]\)
where \(\rho_{ij} = e_i\cdot e_j\) is the inner product of the vectors.
For every correlation matrix we can find unit vectors that satisfy this
using the Cholesky decomposition of the correlation matrix. The rows
of the lower decompsition are the \((e_i\).
)xyzyx";
	template<class X = double>
	class correlation : public blas::matrix<X, CblasNoTrans, CblasLower> {
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
			: blas::matrix<X>(n, n), _e(n* n)
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

			// so we can rotate back from rho = 0 or rho = 1
			if (fabs(rij) < eps) {
				rij = eps; // ??? sign, jitter
			}
			else if (fabs(1 - rij) < eps) {
				rij = 1 - eps;
			}

			// e_i' = c e_i + s e_j
			// rij = e_i' . e_j = c rho + sqrt(1 - c^2)
			// (rij - c rho)^2 = 1 - c^2
			// (1 - rho^2)c^2 + 2 rij rho + 1 - rij^2 = 0
			X rho = rho(i, j);
			X A = 1 - rho * rho;
			X B = rij * rho;
			X C = 1 - rij * rij;
			X D = B * B - A * C;
			// largest c gives smallest theta
			X c = fabs(A) > eps ? (-B + copysigne(sqrt(D), A)) / A : C / (B + copysign(sqrt(D), B));

			blas::scal<X>(c, row(i));
			blas::axpy<X>(sqrt(1 - c * c), row(j), row(i));

			return *this;
		}
	};

} // namespace fms
