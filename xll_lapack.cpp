// xll_lapack.cpp - Intel MKL routines
#include "fms_blas/fms_lapack.h"
#include "xll/xll/xll.h"

using namespace blas;
using namespace lapack;
using namespace xll;

#define CATEGORY "LAPACK"

void xerbla(const char* srname, const int* info, const int)
{
	char buf[256];

	if (*info < 0) {
		sprintf_s(buf, "%s: parameter %d had an illegal value", srname, -*info);
	}
	else if (*info == 1001) {
		sprintf_s(buf, "%s: incompatible optional parameters", srname);
	}
	else if (*info == 1000 or *info == 1089) {
		sprintf_s(buf, "%s: insufficient workspace available", srname);
	}
	else if (info > 0) {
		sprintf_s(buf, "%s: returned error code %d", srname, *info);
	}

	XLL_ERROR(buf);
}

void LAPACKE_xerbla(const char* name, lapack_int info)
{
	char buf[256];

	if (info < 0) {
		sprintf_s(buf, "%s: parameter %d had an illegal value", name, -info);
	}
	else if (info == LAPACK_WORK_MEMORY_ERROR) {
		sprintf_s(buf, "%s: not enough memory to allocate work array", name);
	}
	else if (info == LAPACK_TRANSPOSE_MEMORY_ERROR) {
		sprintf_s(buf, "%s: not enough memory to transpose matrix", name);
	}
	else if (info > 0) {
		sprintf_s(buf, "%s: returned error code %d", name, info);
	}

	XLL_ERROR(buf);
}

inline blas::matrix<double> fpmatrix(_FPX* pa)
{
	return blas::matrix<double>(pa->rows, pa->columns, pa->array);
}

inline const blas::matrix<const double> fpmatrix(const _FPX* pa)
{
	return blas::matrix<const double>(pa->rows, pa->columns, pa->array);
}

AddIn xai_blas_gemm(
	Function(XLL_FPX, "xll_blas_gemm", "BLAS.GEMM")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_FPX, "b", "is a matrix."),
		})
	.Category("BLAS")
	.FunctionHelp("Return the matrix product of a and b.")
);
_FPX* WINAPI xll_blas_gemm(const _FPX* pa, const _FPX* pb)
{
#pragma XLLEXPORT
	static FPX c;

	try {
		ensure(pa->columns == pb->rows);

		c.resize(pa->rows, pb->columns);
		//blas::gemm(fpmatrix(pa), fpmatrix(pb), c.array());
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

AddIn xai_lapack_potrf(
	Function(XLL_FPX, "xll_lapack_potrf", "LAPACK.POTRF")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_BOOL, "_lower", "is an optional boolean indicating lower decomposition. Default is upper."),
		Arg(XLL_BOOL, "_nofill", "is an optional boolean indicated unused values are not set to 0. Default is false."),
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the upper or lower Cholesky decomposition of a.")
	//.HelpTopic(POTRF_TOPIC)
	.Documentation(R"(
This function calculates the Cholesky decomposition of a symmetric positive definite matrix \(A\).
The upper decomposition statisfies \(A = U' U\) and the lower satisifes \(A = L L'\) where
prime indicates matrix transpose. Only the upper or lower entries of \(A\) are used.
)")
);
_FPX* WINAPI xll_lapack_potrf(_FPX* pa, BOOL lower, BOOL nofill)
{
#pragma XLLEXPORT
	try {
		ensure(pa->columns == pa->rows); // fix up ld???

		auto a = fpmatrix(pa);
		lapack::potrf(a, lower ? CblasLower : CblasUpper);

		if (nofill == FALSE) {
			if (lower) {
				for (int i = 0; i < a.rows(); ++i)
					for (int j = i + 1; j < a.columns(); ++j)
						a(i, j) = 0;
			}
			else { // upper
				for (int i = 1; i < a.rows(); ++i)
					for (int j = 0; j < i; ++j)
						a(i, j) = 0;
			}
		}
		// a[upper] = 0 ???
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

AddIn xai_lapack_potri(
	Function(XLL_FPX, "xll_lapack_potri", "LAPACK.POTRI")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_BOOL, "_lower", "is an optional boolean indicating lower triangular data. Default is upper.")
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the inverse of a symmetric positive definite matrix a.")
	//.HelpTopic(POTRI_TOPIC)
	.Documentation(R"(
Return the inverse of the symmetric positive definite matrix <code>A</code>
using the Cholesky decomposition. Only the upper or lower entries of \(A\) are used.
The <code>POTRF</code> function must be called on <code>a</code> prior to calling
<code>POTRI</code>.
)")
);
_FPX* WINAPI xll_lapack_potri(_FPX* pa, BOOL lower)
{
#pragma XLLEXPORT
	try {
		ensure(pa->columns == pa->rows);

		auto a = fpmatrix(pa);
		lapack::potri(a, lower ? CblasLower : CblasUpper);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}