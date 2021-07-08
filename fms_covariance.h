// fms_covariance.h - covariance matrix
#pragma once
#include "fms_correlation.h"

namespace fms {

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



} // namespace fms
