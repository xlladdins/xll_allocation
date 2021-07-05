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

#define POTRF_TOPIC "https://software.intel.com/content/www/us/en/develop/documentation/" \
	"onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/" \
	"lapack-linear-equation-computational-routines/matrix-factorization-lapack-computational-routines/potrf.html"

AddIn xai_mkl_lapack_potrf(
	Function(XLL_FPX, "xll_mkl_lapack_potrf", "LAPACK.POTRF")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_BOOL, "_lower", "is an optional boolean indicating lower decomposition. Default is upper."),
		Arg(XLL_BOOL, "_nofill", "is an optional boolean indicated unused values are not set to 0. Default is false.")
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the upper or lower Cholesky decomposition of a.")
	.HelpTopic(POTRF_TOPIC)
	.Documentation(R"(
This function calculates the Cholesky decomposition of a symmetric positive definite matrix \(A\).
The upper decomposition statisfies \(A = U' U\) and the lower satisifes \(A = L L'\) where
prime indicates matrix transpose. Only the upper or lower entries of \(A\) are used.
)")
);
_FPX* WINAPI xll_mkl_lapack_potrf(_FPX* pa, BOOL lower, BOOL nofill)
{
#pragma XLLEXPORT
	try {
		ensure(pa->columns == pa->rows);

		auto a = fpmatrix(pa);
		if (lower)
			a.lower();
		else
			a.upper();

		int ret = lapack::potrf(a);
		if (ret != 0) {
			char buf[64];
			if (ret < 0) {
				sprintf_s(buf, sizeof(buf), "dpotrf: parameter %d had an illegal value", -ret);
			}
			else {
				sprintf_s(buf, sizeof(buf), "dpotrf: the leading minor of order %d is not positive definite", ret);
			}
			XLL_ERROR(buf);
		}

		if (nofill == FALSE) {
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
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}

#define POTRI_TOPIC "https://software.intel.com/content/www/us/en/develop/documentation/" \
	"onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/" \
	"lapack-linear-equation-computational-routines/matrix-inversion-lapack-computational-routines/potri.html"

AddIn xai_mkl_lapack_potri(
	Function(XLL_FPX, "xll_mkl_lapack_potri", "LAPACK.POTRI")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_BOOL, "_lower", "is an optional boolean indicating lower triangular data. Default is upper.")
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the inverse of a symmetric positive definite matrix a.")
	.HelpTopic(POTRI_TOPIC)
	.Documentation(R"(
Return the inverse of the symmetric positive definite matrix <code>A/code>
using the Cholesky decomposition. Only the upper or lower entries of \(A\) are used.
)")
);
_FPX* WINAPI xll_mkl_lapack_potri(_FPX* pa, BOOL lower)
{
#pragma XLLEXPORT
	try {
		ensure(pa->columns == pa->rows);

		auto a = fpmatrix(pa);
		if (lower)
			a.lower();
		else
			a.upper();

		xll_mkl_lapack_potrf(pa, lower, TRUE);

		int ret = lapack::potri(a);
		if (ret != 0) {
			char buf[64];
			if (ret < 0) {
				sprintf_s(buf, sizeof(buf), "dpotri: parameter %d had an illegal value", -ret);
			}
			else {
				sprintf_s(buf, sizeof(buf), "dpotri: the %d-th diagonal element of the Cholesky factor is zero, and the inversion could not be completed", ret);
			}
			XLL_ERROR(buf);
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}