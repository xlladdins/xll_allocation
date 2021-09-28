// xll_allocation.cpp - optimum asset allocation
#include "xll_allocation.h"

using namespace xll;

AddIn xai_allocation(
	Function(XLL_HANDLE, "xll_allocation", CATEGORY ".ALLOCATION")
	.Arguments({
		Arg(XLL_FP, "R", "is a vector of expected realized returns."),
		Arg(XLL_FP, "Sigma", "is vector of volatilities of returns."),
		Arg(XLL_FP, "Rho", "is the lower Cholesky factor of return correlations."),
		})
	.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp("Return handle to portfolio.")
	.Documentation(R"xyzyx()xyzyx")
);
HANDLEX WINAPI xll_allocation(const _FPX* pR, const _FPX* pSigma, const _FPX* pRho)
{
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;

	try {
		int n = size(*pR);

		ensure(n == (int)size(*pSigma));
		ensure(n == pRho->rows);
		ensure(n == pRho->columns);

		fms::correlation rho(n, pRho->array);

		handle<fms::allocation::portfolio<>> h_(new fms::allocation::portfolio<>(n, pR->array, pSigma->array, rho));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return h;
}