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

	template<class X = double>
	class portfolio {
		blas::vector_array<X> V_x, V_EX; // V^-1 x, V^-1 E[R]
		X A, B, C, D;
	public:
		portfolio(int n, const X* ER, const X* Sigma, const correlation<X>& rho)
			: V_x(n), V_EX(n)
		{
			// calculate V_x, V_EX
			// V_x = V^-1 x, V = sigma rho rho' sigma 
			// V_x = sigma^-1 rho'^-1 rho^-1 sigma^-1 x
			
			// x1 = sigma^-1 x, x = (1,1,...)
			for (int i = 0; i < n; ++i) {
				ensure(Sigma[i] > 0);
				V_x[i] = 1 / Sigma[i];
				V_EX[i] = ER[i] / Sigma[i];
			}

			// x2 = rho^-1 x1
			blas::trsv(CblasLower, rho, V_x.data());
			blas::trsv(CblasLower, rho, V_EX.data());
			
			// x3 = rho'^-1 x2
			blas::trsv(CblasLower, rho.transpose(), V_x.data());
			blas::trsv(CblasLower, rho.transpose(), V_EX.data());
			
			// x4 = sigma^-1 x3
			for (int i = 0; i < n; ++i) {
				V_x[i] = V_x[i] / Sigma[i];
				V_EX[i] = V_EX[i] / Sigma[i];
			}
			
			double _1 = 1;
			blas::vector x(n, &_1, 0); // x = {1,1, ...}
			blas::vector EX(n, const_cast<double*>(ER));

			A = blas::dot(x, V_x); // x . V_x
			B = blas::dot(x, V_EX);  // x . V_EX
			C = blas::dot(EX, V_EX); // E[X] . V_EX
			D = A * C - B * B;
		}

		int size() const
		{
			return V_x.size();
		}

		// minimize variance given target return
		// optimal porfolio is put in xi
		// minimum variance is returned
		// xi = V^-1(lambda x + mu E[X])
		X minimize(X r, X* _xi)
		{
			X lambda = (C - r * B) / D;
			X mu = (r * A - B) / D;

			if (_xi) {
				// xi = lambda V_x + mu V_EX
				auto xi = blas::vector<X>(V_x.size(), _xi);
				xi.copy(V_x);
				blas::scal(lambda, xi);
				blas::axpy(mu, V_EX, xi);
			}
			
			return (C - 2*B*r + A*r*r)/D;
		}
		// maximize return given target variance
		// optimal porfolio is put in xi
		// maximum return is returned
		// xi = V^-1(lambda x + mu E[X])
		// max xi.ER - lambda(xi.x - 1) - mu/2 (xi' V xi - sigma^2)
		// 0 = ER - lambda x - mu V xi
		// xi = V^{-1}(ER - lambda x)
		//X maximize(X s, X* _xi);
	};


} // namespace fms::allocation