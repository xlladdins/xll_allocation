// xll_allocation.cpp - optimum asset allocation
#include "xll_allocation.h"

using namespace xll;

XLL_CONST(DOUBLE, DBL_MAX, DBL_MAX, "Maximum double value.", "CATEGORY", "https://docs.microsoft.com/en-us/cpp/cpp/floating-limits?view=msvc-160");
XLL_CONST(DOUBLE, DBL_EPSILON, DBL_EPSILON, "Smalles double value for which 1 + epsilon != 1.", "CATEGORY", "https://docs.microsoft.com/en-us/cpp/cpp/floating-limits?view=msvc-160");

//Auto<Close> xac_free_buffers([]() { mkl_free_buffers(); return TRUE; });

AddIn xai_allocation(
	Function(XLL_HANDLE, "xll_allocation", "\\" CATEGORY ".ALLOCATION")
	.Arguments({
		Arg(XLL_FP, "ER", "is a vector of expected realized returns."),
		Arg(XLL_FP, "Cov", "is lower triangular covariance matrix."),
		// Arg(XLL_BOOL, "_uplo", "is and optional boolean indicating the covariance matrix is upper."),
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

		handle<fms::allocation> h_(new fms::allocation(n, pR->array, pCov->array));
		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return h;
}

#ifdef _DEBUG

AddIn xai_allocation_V_x(
	Function(XLL_FPX, "xll_allocation_Cov_x", CATEGORY ".ALLOCATION.Cov_x")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle."),
		})
	.Category(CATEGORY)
	.FunctionHelp("Cov inverse x.")
);
_FPX* WINAPI xll_allocation_Cov_x(HANDLEX h)
{
#pragma XLLEXPORT
	static FPX x;
	
	handle<fms::allocation::portfolio<>> h_(h);
	if (h_) {		
		int n = h_->size();
		x.resize(1, n);
		blas::vector(n, x.array()).copy(h_->Cov_x());
	}

	return x.get();
}

AddIn xai_allocation_V_EX(
	Function(XLL_FPX, "xll_allocation_Cov_EX", CATEGORY ".ALLOCATION.Cov_EX")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Cov inverse E[X].")
);
_FPX* WINAPI xll_allocation_Cov_EX(HANDLEX h)
{
#pragma XLLEXPORT
	static FPX x;

	handle<fms::allocation::portfolio<>> h_(h);
	if (h_) {
		int n = h_->size();
		x.resize(1, n);
		blas::vector(n, x.array()).copy(h_->Cov_EX());
	}

	return x.get();
}

#endif // _DEUBG

AddIn xai_allocation_minimize(
	Function(XLL_FPX, "xll_allocation_minimize", CATEGORY ".ALLOCATION.MINIMIZE")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle returned by ALLOCATION."),
		Arg(XLL_DOUBLE, "r", "is the target realized return."),
		Arg(XLL_FP, "_bounds", "is an optional array of bounds."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Return the minimum volatility with given return.")
	.Documentation(R"xyzyx(
The last argument can be a two row array of lower and upper bounds.
)xyzyx")
);
_FPX* WINAPI xll_allocation_minimize(HANDLEX h, double r, const _FPX* plu)
{
#pragma XLLEXPORT
	static FPX x;

	try {
		handle<fms::allocation::portfolio<>> h_(h);
		ensure(h_);
		int n = h_->size();
		x.resize(1, n + 2);
		h_->minimize(r, x.array(), &x[n], &x[n + 1]);
		if (size(*plu) > 1) {
			ensure(2 == plu->rows);
			ensure(n + 2 == plu->columns);
		}
		else if (plu->array[0]) {

		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return x.get();
}

AddIn xai_allocation_minimum(
	Function(XLL_DOUBLE, "xll_allocation_minimum", CATEGORY ".ALLOCATION.MINIMUM")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle returned by ALLOCATION."),
		Arg(XLL_DOUBLE, "R", "is the target expected realized return."),
		})
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
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return sigma;
}

AddIn xai_allocation_maximize(
	Function(XLL_FPX, "xll_allocation_maximize", CATEGORY ".ALLOCATION.MAXIMIZE")
	.Arguments({
		Arg(XLL_DOUBLE, "sigma", "is the target volatility."),
		Arg(XLL_FP, "ER", "is a vector of expected realized returns."),
		Arg(XLL_FP, "Cov", "is lower triangular covariance matrix."),
		Arg(XLL_FP, "_lower", "is an optional array of lower bounds."),
		Arg(XLL_FP, "_upper", "is an optional array of upper bounds."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Return the optimal portfolio with given volatility.")
	.Documentation(R"xyzyx(
The last argument can be a two row array of lower and upper bounds.
)xyzyx")
);
_FPX* WINAPI xll_allocation_maximize(double sigma, const _FPX* pER, const _FPX* pV, const _FPX* pl, const _FPX* pu)
{
#pragma XLLEXPORT
	static FPX x;

	try {
		unsigned n = size(*pER);
		x.resize(1, n + 2);
		fms::allocation::portfolio<> p(n, pER->array, pV->array);
		p.maximize(sigma, x.array(), &x[n], &x[n + 1]);
		if (size(*pl) > 1) {
			ensure(n + 2 == size(*pl));
			ensure(n + 2 == size(*pu));

			fms::allocation::maximize(sigma, n, pER->array, pV->array, x.array(), pl->array, pu->array);
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return x.get();
}

AddIn xai_allocation_maximum(
	Function(XLL_DOUBLE, "xll_allocation_maximum", CATEGORY ".ALLOCATION.MAXIMUM")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle returned by ALLOCATION."),
		Arg(XLL_DOUBLE, "sigma", "is the target volatility."),
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the maximum realized return.")
	.Documentation(R"xyzyx()xyzyx")
);
double WINAPI xll_allocation_maximum(HANDLEX h, double r)
{
#pragma XLLEXPORT
	double sigma = XLL_NAN;

	try {
		handle<fms::allocation::portfolio<>> h_(h);
		ensure(h_);
		sigma = h_->maximize(r, nullptr);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return sigma;
}
