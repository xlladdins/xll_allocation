// xll_correlation.cpp - matrix correlation
#if 0
#include "fms_correlation.h"
#include "xll/xll/xll.h"

#define CATEGORY "XLL"

using namespace fms;
using namespace xll;

AddIn xai_correlation_(
	Function(XLL_HANDLEX, "xll_correlation_", "\\XLL.CORRELATION")
	.Arguments({
		Arg(XLL_LONG, "n", correlation<>::arguments[0]),
		Arg(XLL_FPX, "rho", correlation<>::arguments[1]),
		})
	.Uncalced()
	.FunctionHelp(correlation<>::doc)
	.Category(CATEGORY)
	.Documentation(correlation_doc)
);
HANDLEX WINAPI xll_correlation_(LONG n, const _FPX* prho)
{
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;

	try {
		handle<correlation<>> h_(new correlation(n, prho->array));
		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return h;
}

AddIn xai_correlation(
	Function(XLL_FPX, "xll_correlation", "XLL.CORRELATION")
	.Arguments({
		Arg(XLL_HANDLEX, "cor", "is a handle returned by \\XLL.CORRELATION."),
		})
	.FunctionHelp("Return the correlation matrix.")
	.Category(CATEGORY)
	.Documentation(R"()")
);
_FPX* WINAPI xll_correlation(HANDLEX hrho)
{
#pragma XLLEXPORT
	static FPX cor;

	try {
		handle<correlation<>> h(hrho);
		ensure(h);
		cor.resize(h->rows(), h->columns());
		h->get(cor.array());
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return cor.get();
}

AddIn xai_correlation_rho(
	Function(XLL_HANDLEX, "xll_correlation_rho", "XLL.CORRELATION.RHO")
	.Arguments({
		Arg(XLL_HANDLEX, "cor", "is a handle returned by \\XLL.CORRELATION."),
		Arg(XLL_LONG, "i", "is the row index."),
		Arg(XLL_LONG, "j", "is the column index."),
		Arg(XLL_LPOPER, "_rho", "is a optional correlation to set."),
		})
		.FunctionHelp("Return the correlation or matrix handle if _rho is not missing.")
	.Category(CATEGORY)
	.Documentation(R"()")
);
HANDLEX WINAPI xll_correlation_rho(HANDLEX hrho, LONG i, LONG j, LPOPER prho)
{
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;

	try {
		handle<correlation<>> h_(hrho);
		ensure(h_);
		if (prho->is_missing()) {
			h = h_->rho(i, j);
		}
		else {
			ensure(prho->is_num());
			h_->rho(i, j, prho->as_num());
			h = hrho;
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return h;
}
#endif // 0