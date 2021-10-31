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
		template<class T>
		void ABCD(const blas::vector<T>& EX)
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
			blas::trsv(CblasLower, rho, V_x);
			blas::trsv(CblasLower, rho, V_EX);
			
			// x3 = rho'^-1 x2
			blas::trsv(CblasLower, rho.transpose(), V_x);
			blas::trsv(CblasLower, rho.transpose(), V_EX);
			
			// x4 = sigma^-1 x3
			for (int i = 0; i < n; ++i) {
				V_x[i] = V_x[i] / Sigma[i];
				V_EX[i] = V_EX[i] / Sigma[i];
			}

			ABCD(blas::vector(n, ER));
		}
		// V is symmetric covariance matrix
		portfolio(int n, const X* ER, const X* Cov, CBLAS_UPLO uplo = CblasLower)
			: V_x(n), V_EX(n)
		{
			// calculate V_x, V_EX
			blas::matrix_alloc<X> L(n, n); // lower Cholesky factor
			L.copy(n * n, Cov);
			lapack::potrf(uplo, L);
			
			V_x.fill(1);
			blas::trsv(uplo, L, V_x);
			blas::trsv(uplo, L.transpose(), V_x);

			V_EX.copy(n, ER);
			blas::trsv(uplo, L, V_EX);
			blas::trsv(uplo, L.transpose(), V_EX);

			ABCD(blas::vector(n, ER));
		}

		int size() const
		{
			return V_x.size();
		}

		// Cov^-1 x
		const blas::vector<X>& Cov_x() const
		{
			return V_x;
		}
		// Cov^-1 E[X}
		const blas::vector<X>& Cov_EX() const
		{
			return V_EX;
		}

		X lambda(double r) const
		{
			return (C - r * B) / D;
		}
		X mu(double r) const
		{
			return (r * A - B) / D;
		}

		// minimize variance given target realized return
		// optimal porfolio is put in xi
		// minimum variance is returned
		// min (1/2) xi' V xi - lambda(xi . x - 1) - mu(xi . EX - r)
		// 0 = V xi - lambda x - mu EX
		// xi = V^-1(lambda x + mu E[X])
		// 1 = x' xi = x' V^-1 x lambda + x' V^-1 EX mu = [ A B ] [ lambda ]
		// r = EX' xi = EX V^-1 x lamdda + EX V^-1 EX mu  [ B C ] [ mu     ]
		// [lambda] = 1/D [ C  -B ] [ 1 ]
		// [mu    ]       [ -B  A ] [ r ]
		X minimize(X r, X* _xi, X* _lambda = nullptr, X* _mu = nullptr)
		{
			X lambda = (C - r * B) / D;
			X mu = (-B + r * A) / D;
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
		// max xi.EX - lambda(xi.x - 1) - mu/2 (xi' V xi - sigma^2)
		// 0 = EX - lambda x - mu V xi
		// xi = V^{-1}(EX - lambda x)/mu
		// 1 = xi' x = (B - lambda A)/mu so mu = B - lambda A
		// sigma^2 = xi' V xi 
		//         = (C - 2B^2 lambda + A lambda^2)/mu^2
		//         = (C - 2B^2 lambda + A lambda^2)/(B - lambda A)^2 
		// 0 = (C - 2B lambda + A lambda^2) - sigma^2(B^2 - 2 AB lambda + A^2 lambda^2)
		// 0 = (C - sigma^2 B^2) - 2(B - sigma^2 AB) lambda + (A - sigma^2 A^2) lambda^2
		// return xi' EX = (C - lambda B)/mu
		X maximize(X sigma, X* _xi, double* _lambda = nullptr, double* _mu = nullptr)
		{
			double c = C - sigma * sigma * B * B;
			double b = B - sigma * sigma * A * B;
			double a = A - sigma * sigma * A * A;
			double d = sqrt(b * b - a * c);

			double lambda = (b + d) / a; // +- d ???
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

	// x = (xi[n], lambda, mu)
	template<class X>
	X maximize(X sigma, int n, const double* ER, const double* V, double* x)
	{
		portfolio<X> p(n, ER, V);
		
		return p.maximize(sigma, x, x + n, x + n + 1);
	}
	// F(x) = F(xi[n], lambda, mu) = -(xi.EX - lambda(xi.x - 1) - mu/2 (xi' V xi - sigma^2))
	// subject to l <= x <= u
	// D_xi F(xi[n], lambda, mu) = -EX + lambda x + mu V xi
	// D_lambda = xi.x - 1, x = (1, 1, ...)
	// D_mu = V xi - sigma*sigma
	// On entry x contains initial guess
	template<class X>
	X maximize(X sigma, int n, const double* ER, const double* V, double* x, const double* l, const double* u)
	{
		
		{
			int ret;
			
			double x_[2];// = { -1, 1 };
			//double l_[2];// = { 0, 0 };
			//double u_[2];// = { 10, 10 };
			double* _x = x_;
			
			trnslpbc q(2, 2, _x, l, u);
			ret = q.init();
			ensure(TR_SUCCESS == ret);			
		}
		
		trnslpbc p(n + 2, 2, x, l, u);
		ensure(TR_SUCCESS == p.init());
		x = x; u = u; l = l;

		struct data {
			double sigma;
			const double* ER;
			const double* V;
		};

		// move inside bounds
		/*
		for (int i = 0; i < n + 2; ++i) {
			if (x[i] <= l[i]) {
				x[i] = l[i] + 0.1;// p.rs;
			}
			if (x[i] >= u[i]) {
				x[i] = u[i] - 0.1;// p.rs;
			}
		}
		*/

		p.f = [](int N, int, double* x, double* fx, void* pdata) {
			int n = N - 2;
			data* p = (data*)pdata;

			blas::matrix v(n, n, p->V);
			blas::vector xi(n, x);
			double lambda = x[n];
			double mu = x[n + 1];

			*fx = -blas::dot(xi, blas::vector(n, p->ER));
			*fx += lambda * (blas::sum(xi) - 1);
			*fx += (mu / 2) * (blas::quad(CblasLower, v, xi) - p->sigma * p->sigma);
		};
	
		p.df = [](int N, int, double* x, double* df, void* pdata) {
			int n = N - 2;
			data* p = (data*)pdata;

			blas::matrix v(n, n, p->V);
			blas::vector xi(n, x);
			blas::vector dF(n, df);
			double lambda = x[n];
			double mu = x[n + 1];
			
			// dF = mu V xi
			blas::gemv(v, xi, dF, mu);
			// + lambda x
			blas::axpy(1., blas::vector(n, &lambda, 0), dF);
			// -EX
			blas::axpy(-1., blas::vector(n, p->ER), dF);
			df[n] = blas::sum(xi) - 1;
			df[n + 1] = (blas::quad(CblasLower, v, xi) - p->sigma * p->sigma) / 2;
		};

		//blas::vector x_(n + 2, x);
		blas::vector_alloc<double> df(n + 2);
		ensure(TR_SUCCESS == p.check(x, df.data()));
		
		int rci = 0;
		data data = { sigma, ER, V };

		ensure(TR_SUCCESS == p.solver(x, df.data(), rci, &data));

		int iter = 0, st_cr = 0;
		double r1 = 0, r2 = 0;
		ensure(TR_SUCCESS == p.get(iter, st_cr, r1, r2));

		double result;
		p.f(n, 1, x, &result, &data); // already computed???

		return result;
	}

} // namespace fms::allocation