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
	.Documentation(R"(
Return a one row array of the mean of each column.
)")
);
_FPX* WINAPI xll_fms_mean(const _FPX* pa)
{
#pragma XLLEXPORT
	static FPX a;

	try {
		a.resize(1, pa->columns);
		fms::mean(pa->rows, pa->columns, pa->array, a.array());
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return a.get();
}

AddIn xai_fms_cov(
	Function(XLL_FP, "xll_fms_cov", CATEGORY ".COV")
	.Arguments({
		Arg(XLL_FP, "array", "is an array."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Return mean of columns in the first row and covariance of column vectors below.")
	.Documentation(R"(
Return a \((n + 1)\times n\) matrix where the first has the column means and the remaining
\(n\times n\) matrix is the sample covariance of the columns.
)")
);
_FPX* WINAPI xll_fms_cov(const _FPX* pa)
{
#pragma XLLEXPORT
	static FPX a;

	try {
		a.resize(pa->columns + 1, pa->columns);
		fms::covariance(pa->rows, pa->columns, pa->array, a.array() + pa->columns, a.array());
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return a.get();
}