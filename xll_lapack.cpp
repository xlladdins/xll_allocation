// xll_lapack.cpp - Intel MKL routines
#include "fms_blas/fms_blas.h"
#include "xll/xll/xll.h"

using namespace blas;
using namespace lapack;
using namespace xll;

#define CATEGORY "MKL"

inline auto fpmatrix(_FPX* pa)
{
	return blas::matrix(pa->rows, pa->columns, pa->array);
}
inline auto fpmatrix(const _FPX* pa)
{
	return blas::matrix(pa->rows, pa->columns, pa->array);
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
_FPX* WINAPI xll_mkl_blas_gemm(_FPX* pa, _FPX* pb)
{
#pragma XLLEXPORT
	static FPX c;

	try {
		ensure(pa->columns == pb->rows);

		c.resize(pa->rows, pb->columns);
		blas::gemm(fpmatrix(pa), fpmatrix(pb), c.array());
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
		Arg(XLL_BOOL, "_upper", "is an optional boolean indicatine upper decomposition. Default is false.")
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the Cholesky decomposition.")
	.Documentation(R"()")
);
_FPX* WINAPI xll_mkl_lapack_potrf(_FPX* pa, BOOL upper)
{
#pragma XLLEXPORT
	try {
		ensure(pa->columns == pa->rows);

		auto a = fpmatrix(pa);
		if (upper)
			a.upper();
		else
			a.lower();

		lapack::potrf(a);

		if (a.uplo() == CblasUpper) {
			for (int i = 1; i < a.rows(); ++i)
				for (int j = 0; j < i; ++j)
					a(i, j) = 0;
		}
		else { // lower
			for (int i = 0; i < a.rows(); ++i)
				for (int j = i + 1; j < a.columns(); ++j)
					a(i, j) = 0;
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}

AddIn xai_mkl_lapack_potri(
	Function(XLL_FPX, "xll_mkl_lapack_potri", "LAPACK.POTRI")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_BOOL, "_upper", "is an optional boolean indicatine upper decomposition. Default is false.")
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the inverse of the Cholesky decomposition.")
	.Documentation(R"()")
);
_FPX* WINAPI xll_mkl_lapack_potri(_FPX* pa, BOOL upper)
{
#pragma XLLEXPORT
	try {
		ensure(pa->columns == pa->rows);

		auto a = fpmatrix(pa);
		if (upper)
			a.upper();
		else
			a.lower();

		lapack::potri(a);

		if (a.uplo() == CblasUpper) {
			for (int i = 1; i < a.rows(); ++i)
				for (int j = 0; j < i; ++j)
					a(i, j) = 0;
		}
		else { // lower
			for (int i = 0; i < a.rows(); ++i)
				for (int j = i + 1; j < a.columns(); ++j)
					a(i, j) = 0;
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}