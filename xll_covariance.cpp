// xll_covariance.cpp - Covariance matrix
#include "fms_covariance.h"
#include "xll_allocation.h"

using namespace xll;

AddIn xai_fms_mean(
	Function(XLL_FP, "xll_fms_mean", CATEGORY ".MEAN")
	.Arguments({
		Arg(XLL_FP, "array", "is an array."),
		})
	.Category(CATEGORY)
	.FunctionHelp("Return means of column vectors.")
);
_FPX* WINAPI xll_fms_mean(const _FPX* pa)
{
#pragma XLLEXPORT
	static FPX a;

	try {
		a.resize(1, pa->columns);
		std::fill(a.begin(), a.end(), 0);
		fms::mean(pa->rows, pa->columns, pa->array, a.array());
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return a.get();
}