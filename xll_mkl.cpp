// xll_mkl.cpp - Intel MKL routines
#include "fms_mkl.h"
#include "xll/xll/xll.h"

using namespace mkl;
using namespace xll;

#define CATEGORY "MKL"

inline mkl::matrix<double> fpmatrix(_FPX* pa)
{
	return mkl::matrix<double>(pa->rows, pa->columns, pa->array);
}
inline mkl::matrix<const double> fpmatrix(const _FPX* pa)
{
	return mkl::matrix<const double>(pa->rows, pa->columns, pa->array);
}

AddIn xai_mkl_blas_gemm(
	Function(XLL_FPX, "xll_mkl_blas_gemm", "BLAS.GEMM")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_FPX, "b", "is a matrix."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Return the matrix product of a and b.")
);
_FPX* WINAPI xll_mkl_blas_gemm(const _FPX* pa, const _FPX* pb)
{
#pragma XLLEXPORT
	static FPX c;

	try {
		ensure(pa->columns == pb->rows);

		c.resize(pa->rows, pb->columns);
		blas::mm(fpmatrix(pa), fpmatrix(pb), c.array());
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return c.get();
}

AddIn xai_mkl_lapack_potrf(
	Function(XLL_FPX, "xll_mkl_lapack_potrf", "LAPACK.POTRF")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Return the Cholesky decomposition.")
);
_FPX* WINAPI xll_mkl_lapack_potrf(_FPX* pa)
{
#pragma XLLEXPORT
	try {
		ensure(pa->columns == pa->rows);

		if (0 != lapack::potrf(pa->rows, pa->array))
			return nullptr;
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}

#ifdef _DEBUG
Auto<OpenAfter> xaoa_mkl_test(mkl::test);
#endif // _DEBUG