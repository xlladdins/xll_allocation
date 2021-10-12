// xll_allocation.cpp - optimum asset allocation
#include "xll_allocation.h"

using namespace xll;

#ifdef _DEBUG
int xll_allocation_test = []() {
	//_crtBreakAlloc = 210;
	try {
		//mkl_disable_fast_mm();
		double r = 0.6;
		fms::correlation c(2, &r); // [1   0] [1 .6] = [1 .6]
								   // [.6 .8] [0 .8]   [.6 1]
		{
			ensure(1 == c(0, 0));
			ensure(0 == c(0, 1));
			ensure(.6 == c(1, 0));
			ensure(.8 == c(1, 1));

			ensure(1 == c.rho(0, 0));
			ensure(0.6 == c.rho(0, 1));
			ensure(0.6 == c.rho(1, 0));
			ensure(1 == c.rho(1, 1));
		}
		{
			double ER[] = { 2, 3};
			double Sigma[] = { .1, .2 };

			fms::allocation::portfolio ap(2, ER, Sigma, c);
			
			double sigma2, xi[2];

			auto eq = [](double x, double y, double tol) {
				return fabs(x - y) <= tol;
			};
			constexpr double eps = std::numeric_limits<double>::epsilon();

			sigma2 = ap.minimize(ER[0], xi);
			ensure(Sigma[0] == sigma2);
			ensure(eq(1, xi[0], eps));
			ensure(eq(0, xi[1], eps));

			sigma2 = ap.minimize(ER[1], xi);
			ensure(eq(Sigma[1], sigma2, 100*eps));
			ensure(eq(0, xi[0], 100*eps));
			ensure(eq(1, xi[1], 100*eps));

			double r_ = ap.maximize(sigma2, xi);
			ensure(r_ == r);
			//ensure(eq(0, xi[0], 2 * eps));
			//ensure(eq(1, xi[1], 2 * eps));
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 1;
	}

	return 0;
}();
#endif // _DEBUG

AddIn xai_allocation(
	Function(XLL_HANDLE, "xll_allocation", "\\" CATEGORY ".ALLOCATION")
	.Arguments({
		Arg(XLL_FP, "ER", "is a vector of expected realized returns."),
		Arg(XLL_FP, "Cov", "is lower triangular covariance matrix."),
		})
	.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp("Return handle to portfolio.")
	.Documentation(R"xyzyx()xyzyx")
);
HANDLEX WINAPI xll_allocation(const _FPX* pR, const _FPX* pCov)
{
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;

	try {
		int n = size(*pR);

		ensure(n == pCov->rows);
		ensure(n == pCov->columns);

		handle<fms::allocation::portfolio<>> h_(new fms::allocation::portfolio<>(n, pR->array, (double*)pCov->array));
		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return h;
}

AddIn xai_allocation_minimize(
	Function(XLL_FPX, "xll_allocation_minimize", CATEGORY ".ALLOCATION.MINIMIZE")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle returned by ALLOCATION."),
		Arg(XLL_DOUBLE, "R", "is the target expected realized return."),
		})
		.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp("Return a portfolio with given return and minimum volatility.")
	.Documentation(R"xyzyx()xyzyx")
);

_FPX* WINAPI xll_allocation_minimize(HANDLEX h, double r)
{
#pragma XLLEXPORT
	static FPX xi;

	try {
		xi[0] = XLL_NAN;
		handle<fms::allocation::portfolio<>> h_(h);
		ensure(h_);
		int n = h_->size();
		xi.resize(n, 1);
		h_->minimize(r, xi.array());
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return xi.get();
}

AddIn xai_allocation_minimum(
	Function(XLL_DOUBLE, "xll_allocation_minimum", CATEGORY ".ALLOCATION.MINIMUM")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle returned by ALLOCATION."),
		Arg(XLL_DOUBLE, "R", "is the target expected realized return."),
		})
		.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp("Return the minimum volatility.")
	.Documentation(R"xyzyx()xyzyx")
);
double WINAPI xll_allocation_minimum(HANDLEX h, double r)
{
#pragma XLLEXPORT
	double sigma = XLL_NAN;

	try {
		handle<fms::allocation::portfolio<>> h_(h);
		ensure(h_);
		sigma = h_->minimize(r, nullptr);
		ensure(sigma >= 0);
		sigma = sqrt(sigma);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return sigma;
}