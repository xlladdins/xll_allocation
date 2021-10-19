// fms_allocation.h - Efficient portfolio allocation
// https://arxiv.org/pdf/2009.10852.pdf
#pragma once
#include <valarray>
#include "fms_blas/fms_lapack.h"
#include "fms_blas/fms_trnlsp.h"
#include "fms_correlation.h"

namespace fms::allocation {

static inline const char fms_allocation_doc[] = R"xyzyx(
Given a vector of realized returns \(R\) with covariance \(\Sigma = \operatorname{Cov}(R,R)\)
and a target expected realized return \(\rho\), find a portfolio having miniumum variance.
)xyzyx";

	template<class X = double>
	class portfolio {
		blas::vector_alloc<X> V_x, V_EX; // V^-1 x, V^-1 E[R]
		X A, B, C, D;


		void ABCD(const blas::vector<X>& EX)
		{
			A = sum(V_x);   // x . V_x
			B = sum(V_EX);  // x . V_EX
			C = blas::dot(EX, V_EX); // E[X]' V^{-1} EX
			D = A * C - B * B;
		}
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

			ABCD(blas::vector(n, const_cast<X*>(ER)));
		}
		// V is lower triangular covariance matrix
		portfolio(int n, const X* ER, /*const*/ X* V)
			: V_x(n), V_EX(n)
		{
			// calculate V_x, V_EX
			blas::trsv(CblasLower, blas::matrix(n, n, V), V_x.data());
			blas::trsv(CblasLower, blas::matrix(n, n, V), V_EX.data());

			ABCD(blas::vector(n, const_cast<X*>(ER)));
		}

		int size() const
		{
			return V_x.size();
		}

		X lambda(double r) const
		{
			return (C - r * B) / D;
		}
		X mu(double r) const
		{
			return (r * A - B) / D;
		}

		// minimize variance given target return
		// optimal porfolio is put in xi
		// minimum variance is returned
		// xi = V^-1(lambda x + mu E[X])
		X minimize(X r, X* _xi, double* _lambda = nullptr, double* _mu = nullptr)
		{
			X lambda = (C - r * B) / D;
			X mu = (r * A - B) / D;
			if (_lambda) {
				*_lambda = lambda;
			}
			if (_mu) {
				*_mu = mu;
			}

			if (_xi) {
				// xi = lambda V_x + mu V_EX
				auto xi = blas::vector<X>(V_x.size(), _xi);
				xi.copy(V_x);
				blas::scal(lambda, xi);
				blas::axpy(mu, V_EX, xi);
			}
			
			return sqrt((C - 2*B*r + A*r*r)/D);
		}
		// maximize return given target variance
		// optimal porfolio is put in xi
		// maximum return is returned
		// xi = V^-1(lambda x + mu E[X])
		// max xi.EX - lambda(xi.x - 1) - mu/2 (xi' V xi - sigma^2)
		// 0 = EX - lambda x - mu V xi
		// xi = V^{-1}(EX - lambda x)/mu
		// 1 = xi.x = (B - lambda A)/mu
		// sigma^2 = xi.V xi = (C - 2B^2 lambda + A lambda^2)/mu^2
		// 0 = (C - 2B lambda + A lambda^2) - sigma^2(B^2 - 2 AB lambda + A^2 lambda^2)
		// 0 = (C - sigma^2 B^2) - 2(B - sigma^2 AB) lambda + (A - sigma^2 A^2) lambda^2
		// mu = B - lambda A
		// xi.EX = (C - lambda B)/mu 
		X maximize(X sigma, X* _xi, double* _lambda = nullptr, double* _mu = nullptr)
		{
			double c = C - sigma * sigma * B * B;
			double b = B - sigma * sigma * A * B;
			double a = A - sigma * sigma * A * A;
			double d = sqrt(b * b - a * c);

			double lambda = (-b + d) / a; // +- d ???
			double mu = B - lambda * A;
			if (_lambda) {
				*_lambda = lambda;
			}
			if (_mu) {
				*_mu = mu;
			}

			if (_xi) {
				// xi = (V_EX - lambda V_X)/mu
				auto xi = blas::vector<X>(V_x.size(), _xi);
				xi.copy(V_EX);
				blas::axpy(-lambda, V_x, xi);
				blas::scal(1/mu, xi);
			}

			return (C - lambda * B) / mu;
		}

	};

	// x = (lambda, um, xi)
	template<class X>
	X maximize(X sigma, int n, const double* ER, const double* V, double* x)
	{
		portfolio<X> p(n, ER, V);
		
		return p.maximize(sigma, x + 2, x + 0, x + 1);
	}
	// F(lambda, mu, xi) = -(xi.EX - lambda(xi.x - 1) - mu/2 (xi' V xi - sigma^2))
	// subject to l <= xi <= u
	// D_lambda = xi.x - 1, x = (1, 1, ...)
	// D_mu = V xi - sigma*sigma
	// D_xi F(lambda, mu, xi) = -EX + lambda x + mu V xi
	// x = (lambda, mu, xi)
	// On entry x contains initial guess
	template<class X>
	X maximize(X sigma, int n, const double* ER, const double* V, double* xi, const double* l, const double* u)
	{
		// initial xi
		maximize(sigma, n - 2, ER, V, xi);
		fms::trnslpbc p(n, 1, xi, l, u);
		p.f = [&n,sigma,ER,V](int n, int, int, const double* x, double* fx) {
			blas::matrix v(n, n, V);
			double lambda = x[0];
			double mu = x[1];
			blas::vector xi(n, x + 2);
			*fx = -blas::dot(xi, blas::vector(n, ER));
			*fx += lambda * (blas::sum(xi) - 1);
			*fx += (mu / 2) * (blas::quad(CblasLower, v, xi) - sigma & sigma);
		};
		p.df = [&n, sigma, ER, V](int n, int, int, const double* x, double* df) {
			blas::matrix v(n, n, V);
			double lambda = x[0];
			double mu = x[1];
			blas::vector xi(n, x + 2);
			df[0] = blas::sum(xi) - 1;
			df[1] = (blas::quad(CblasLower, v, xi) - sigma * sigma) / 2;
			blas::vector dF(n, df + 2);
			// mu V xi
			blas::gemv(v, xi, dF.data(), dF.incr(), mu);
			// + lambda x
			blas::axpy(1, blas::vector(n, &lambda, 0), dF);
			// -EX
			blas::axpy(-1, blas::vector(n, ER), dF);
		};
		ensure(TR_SUCCESS == p.init());

		blas::vector<X> x(2 + n, xi);
		blas::vector_alloc<X> df(2 + n);
		ensure(TR_SUCCESS = p.check(x.data(), df.data()));
		
		int rci = 0;
		ensure(TR_SUCCESS == p.solver(x.data(), df.data(), rci));

		int iter = 0, st_cr = 0;
		double r1 = 0, r2 = 0;
		ensure(TR_SUCCESS == p.get(iter, st_cr, r1, r2));

		double result;
		p.f(n, 1, x, &result);

		return result;
	}

} // namespace fms::allocation