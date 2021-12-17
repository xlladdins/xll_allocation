// xll_allocation.cpp - optimum asset allocation
#include "xll_allocation.h"

using namespace xll;
using namespace fms;

XLL_CONST(DOUBLE, DBL_MAX, DBL_MAX, "Maximum double value.", "CATEGORY", "https://docs.microsoft.com/en-us/cpp/cpp/floating-limits?view=msvc-160");
XLL_CONST(DOUBLE, DBL_EPSILON, DBL_EPSILON, "Smalles double value for which 1 + epsilon != 1.", "CATEGORY", "https://docs.microsoft.com/en-us/cpp/cpp/floating-limits?view=msvc-160");

//Auto<Close> xac_free_buffers([]() { mkl_free_buffers(); return TRUE; });

AddIn xai_allocation(
	Function(XLL_HANDLE, "xll_allocation", "\\" CATEGORY ".ALLOCATION")
	.Arguments({
		Arg(XLL_FP, "L", "is lower triangular packed covariance matrix."),
		Arg(XLL_FP, "ER", "is a vector of expected realized returns."),
		// Arg(XLL_BOOL, "_uplo", "is and optional boolean indicating the covariance matrix is upper."),
		})
	.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp("Return handle to portfolio.")
	.Documentation(R"xyzyx()xyzyx")
);
HANDLEX WINAPI xll_allocation(const _FPX* pL, const _FPX* pR)
{
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;

	try {
		auto n = size(*pR);
		ensure((n*(n+1)/2) == size(*pL));

		handle<fms::allocation> h_(new fms::allocation(n, pL->array, pR->array));
		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return h;
}
#ifdef _DEBUG

// non-owning vector
inline blas::vector<double> fpvector(_FPX* pa)
{
	return blas::vector<double>(size(*pa), pa->array, 1);
}
// non-owning matrix
inline blas::matrix<double> fpmatrix(_FPX* pa)
{
	return blas::matrix<double>(pa->rows, pa->columns, pa->array);
}

AddIn xai_project(
	Function(XLL_FPX, "xll_project", CATEGORY ".PROJECT")
	.Arguments({
		Arg(XLL_FPX, "z", "is a vector."),
		Arg(XLL_FPX, "x", "is a vector."),
		Arg(XLL_FPX, "y", "is a vector."),
		Arg(XLL_DOUBLE, "a", "is a number."),
		Arg(XLL_DOUBLE, "b", "is a number."),
		})
	.Category(CATEGORY)
	.FunctionHelp("Project z onto z'x = a, z'y = b.")
);
_FPX* WINAPI xll_project(_FPX* pz, _FPX* px, _FPX* py, double a, double b)
{
#pragma XLLEXPORT
	static FPX xi;

	try {
		ensure(size(*pz) == size(*px));
		ensure(size(*px) == size(*py));
		auto z = fpvector(pz);
		project(fpvector(px), fpvector(py), a, b, z);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pz;
}

AddIn xai_allocation_fmin(
	Function(XLL_DOUBLE, "xll_allocation_fmin", CATEGORY ".ALLOCATION.FMIN")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle returned by ALLOCATION."),
		Arg(XLL_DOUBLE, "R", "is the target expected realized return."),
		Arg(XLL_FPX, "x", "is a vector of xi, lambda, mu.")
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the fmin volatility.")
	.Documentation(R"xyzyx()xyzyx")
);
double WINAPI xll_allocation_fmin(HANDLEX h, double R, _FPX* px)
{
#pragma XLLEXPORT
	double sigma = XLL_NAN;

	try {
		auto N = size(*px);
		handle<fms::allocation> h_(h);
		ensure(h_);
		ensure(N == h_->size() + 2u);
		sigma = fms::fmin(R, vec(N, px->array), h_->L, h_->ER);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return sigma;
}

AddIn xai_allocation_gmin(
	Function(XLL_FPX, "xll_allocation_gmin", CATEGORY ".ALLOCATION.GMIN")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle returned by ALLOCATION."),
		Arg(XLL_DOUBLE, "R", "is the target expected realized return."),
		Arg(XLL_FPX, "x", "is a vector of xi, lambda, mu.")
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the gmin gradient.")
	.Documentation(R"xyzyx()xyzyx")
);
_FPX* WINAPI xll_allocation_gmin(HANDLEX h, double R, _FPX* px)
{
#pragma XLLEXPORT
	static FPX g;

	try {
		auto N = size(*px);
		handle<fms::allocation> h_(h);
		ensure(h_);
		ensure(N == h_->size() + 2u);
		g.resize(1, N);
		vec g_(N, g.array());
		fms::gmin(R, vec(N, px->array), h_->L, h_->ER, g_);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return g.get();
}

AddIn xai_allocation_V_1(
	Function(XLL_FPX, "xll_allocation_V_1", CATEGORY ".ALLOCATION.V_1")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle."),
		})
	.Category(CATEGORY)
	.FunctionHelp("Cov inverse x.")
);
_FPX* WINAPI xll_allocation_V_1(HANDLEX h)
{
#pragma XLLEXPORT
	static FPX x;
	
	handle<fms::allocation> h_(h);
	if (h_) {		
		int n = h_->size();
		x.resize(1, n);
		blas::vector(n, x.array()).copy(h_->V_1);
	}

	return x.get();
}

AddIn xai_allocation_V_ER(
	Function(XLL_FPX, "xll_allocation_V_ER", CATEGORY ".ALLOCATION.V_ER")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Cov inverse E[X].")
);
_FPX* WINAPI xll_allocation_V_ER(HANDLEX h)
{
#pragma XLLEXPORT
	static FPX x;

	handle<fms::allocation> h_(h);
	if (h_) {
		int n = h_->size();
		x.resize(1, n);
		blas::vector(n, x.array()).copy(h_->V_ER);
	}

	return x.get();
}

AddIn xai_allocation_ABCD(
	Function(XLL_FPX, "xll_allocation_ABCD", CATEGORY ".ALLOCATION.ABCD")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Return A, B, C, D.")
);
_FPX* WINAPI xll_allocation_ABCD(HANDLEX h)
{
#pragma XLLEXPORT
	static FPX x(1,4);

	handle<fms::allocation> h_(h);
	if (h_) {
		x[0] = h_->A;
		x[1] = h_->B;
		x[2] = h_->C;
		x[3] = h_->D;
	}
	else {
		return 0;
	}

	return x.get();
}
#endif // _DEUBG


AddIn xai_allocation_minimize(
	Function(XLL_FPX, "xll_allocation_minimize", CATEGORY ".ALLOCATION.MINIMIZE")
	.Arguments({
		Arg(XLL_HANDLEX, "h", "is a handle returned by ALLOCATION."),
		Arg(XLL_DOUBLE, "R", "is the target realized return."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Return the minimum volatility with given return.")
	.Documentation(R"xyzyx(
The last argument can be a two row array of lower and upper bounds.
)xyzyx")
);
_FPX* WINAPI xll_allocation_minimize(HANDLEX h, double R)
{
#pragma XLLEXPORT
	static FPX x;

	try {
		handle<fms::allocation> h_(h);
		ensure(h_);
		int n = h_->size();
		x.resize(1, n + 2);
		auto x_ = vec(n + 2, x.array());
		h_->minimum(R, x_);
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
double WINAPI xll_allocation_minimum(HANDLEX h, double R)
{
#pragma XLLEXPORT
	double sigma = XLL_NAN;

	try {
		handle<fms::allocation> h_(h);
		ensure(h_);
		auto v = vec{};
		sigma = h_->minimum(R, v);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return sigma;
}



#if 0
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
		fms::allocation> p(n, pER->array, pV->array);
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
#endif // 0