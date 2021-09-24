// fms_allocation.h - Efficient portfolio allocation
// https://arxiv.org/pdf/2009.10852.pdf
#pragma once
#include <valarray>
#include "fms_blas/fms_lapack.h"
#include "fms_correlation.h"

namespace fms::allocation {

static inline const char fms_allocation_doc[] = R"xyzyx(
Given a vector of realized returns \(R\) with covariance \(\Sigma = \operatorname{Cov}(R,R)\)
and a target expected realized return \(\rho\), find a portfolio having miniumum variance.
)xyzyx";

	template<class X>
	class portfolio {
		std::valarray<X> ER; // expected realized return
		std::valarray<X> Sigma; // volatilities
		fms::correlation<X> rho; // lower Cholesky correlation unit vectors

		blas::vector_array<X> V_x, V_EX; // V^-1 x, V^-1 E[R]
		X A, B, C, D;
	public:
		portfolio(int n, const X* x, const X* ER, const X* Sigma, const correlation<X>& rho)
			: ER(ER, n), Sigma(Sigma, n), rho(rho)
		{
			// calculate V_x, V_ER
			A = 0; // x V_x, x = {1,1, ...}
			B = 0;  // x V_EX
			C = 0; // E[x] V_EX
			D = B * B - A * C;
		}
		// minimize variance given target return
		// optimal porfolio is put in xi
		// minimum variance is returned
		// xi = V^-1(lambda x + mu E[X])
		X minimize(X R, X* _xi)
		{
			X lambda = (C - R * B) / D;
			X mu = (R * A - B) / D;

			// xi = lambda V_x + mu V_EX
			auto xi = blas::vector<X>(rho.dimension(), _xi);
			xi.copy(V_x);
			blas::scal(mu, xi);
			blas::axpy(lambda, V_EX, xi);
			
			return (C - 2*B*R + A*R*R)/D;
		}
	};


} // namespace fms::allocation