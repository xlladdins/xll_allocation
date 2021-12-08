// xll_lapack.cpp - Intel MKL routines
#include "fms_blas/fms_blas.h"
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

AddIn xai_array_(
	Function(XLL_HANDLE, "xll_array_", "\\ARRAY")
	.Arguments({
		Arg(XLL_FP, "array", "is an array or handle to an array of numbers."),
		})
	.Uncalced()
	.FunctionHelp("Return a handle to the in-memory array.")
	.Category(CATEGORY)
	.Documentation(R"(
Create an in-memory two-dimensional array of numbers to be used by array functions.
)")
);
HANDLEX WINAPI xll_array_(const _FPX* pa)
{
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;

	try {
		handle<FPX> h_(new FPX(*pa));
		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCTION__ ": unknown exception");
	}

	return h;
}

AddIn xai_array_get(
	Function(XLL_FP, "xll_array_get", "ARRAY")
	.Arguments({
		Arg(XLL_HANDLE, "handle", "is a handle to an array of numbers."),
		})
	.FunctionHelp("Return an array associated with handle.")
	.Category(CATEGORY)
	.Documentation(R"(
Retrieve an in-memory array created by
<code>\ARRAY</code>. By default the handle is checked to
ensure the array was created by a previous call to <code>\ARRAY</code>.
)")
.SeeAlso({ "\\ARRAY" })
);
_FPX* WINAPI xll_array_get(HANDLEX h)
{
#pragma XLLEXPORT
	_FPX* pa = nullptr;

	try {
		handle<FPX> h_(h);
		if (h_) {
			pa = h_->get();
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCTION__ ": unknown exception");
	}

	return pa;
}

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

AddIn xai_blas_vector_(
	Function(XLL_HANDLEX, "xll_blas_vector_", "\\BLAS.VECTOR")
	.Arguments({
		Arg(XLL_FPX, "v", "is an array of vector elments."),
		})
		.Uncalced()
	.Category("BLAS")
	.FunctionHelp("Return a handle to a BLAS vector.")
);
HANDLEX WINAPI xll_blas_vector_(_FPX* pv)
{
#pragma XLLEXPORT
	HANDLEX result = INVALID_HANDLEX;

	try {
		handle<blas::vector<double>> h(new blas::vector_alloc<double>(size(*pv), pv->array, 1));
		ensure(h);
		result = h.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCTION__ ": unknown exception");
	}

	return result;
}

AddIn xai_blas_matrix(
	Function(XLL_HANDLEX, "xll_blas_matrix", "\\BLAS.MATRIX")
	.Arguments({
		Arg(XLL_FPX, "v", "is an array of matrix elments."),
		Arg(XLL_BOOL, "_trans", "is an optional boolean indicating the matrix is transposed. Default is false.")
		})
	.Uncalced()
	.Category("BLAS")
	.FunctionHelp("Return a handle to a BLAS matrix.")
);
HANDLEX WINAPI xll_blas_matrix(_FPX* pv, BOOL t)
{
#pragma XLLEXPORT
	HANDLEX result = INVALID_HANDLEX;

	try {
		handle<blas::matrix<double>> h(new blas::matrix_alloc<double>(pv->rows, pv->columns, pv->array, t ? CblasTrans : CblasNoTrans));
		ensure(h);
		result = h.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCTION__ ": unknown exception");
	}

	return result;
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
_FPX* WINAPI xll_blas_gemm(_FPX* pa, _FPX* pb)
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

AddIn xai_blas_tpmv(
	Function(XLL_FPX, "xll_blas_tpmv", "BLAS.TPMV")
	.Arguments({
		Arg(XLL_FPX, "a", "is a packed triangular matrix."),
		Arg(XLL_FPX, "x", "is a vector."),
		Arg(XLL_BOOL, "_upper", "indicates a is upper. Default is lower")
		})
		.Category("BLAS")
	.FunctionHelp("Return the matrix product of a and x.")
);
_FPX* WINAPI xll_blas_tpmv(_FPX* pa, _FPX* px, BOOL upper)
{
#pragma XLLEXPORT
	static FPX c;

	try {
		auto n = size(*px);
		c.resize(px->rows, px->columns);
		std::copy(begin(*px), end(*px), c.array());
		auto c_ = fpvector(c.get());
		auto a_ = blas::matrix<double>(n, n, pa->array);
		blas::tpmv(upper ? CblasUpper : CblasLower, a_, c_);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return c.get();
}


AddIn xai_pack(
	Function(XLL_FPX, "xll_pack", "PACK")
	.Arguments({
		Arg(XLL_FPX, "A", "is a square matrix."),
		Arg(XLL_BOOL, "_upper", "is an optional argument indicating upper trangle of A is used. Default is lower.")
		})
	.FunctionHelp("Pack lower or upper triangle of A.")
	.Category(CATEGORY)
	.Documentation(R"(
Pack lower \([a_{ij}\) as \([a_{00}, a_{10}, a_{11}, a_{20}, a_{21}, a_{22},\ldots]\)
and upper as \([a_{00}, a_{01}, a_{11}, a_{02}, a_{12}, a_{22},\ldots]\).
)")
);
_FPX* WINAPI xll_pack(_FPX* pa, BOOL upper)
{
#pragma XLLEXPORT
	static FPX l;

	int n = pa->rows;
	if (n != pa->columns) {
		XLL_ERROR(__FUNCTION__ ": matrix must be square");

		return nullptr;
	}

	l.resize(1, (n * (n + 1)) / 2);

	if (upper) {
		blas::packu(n, pa->array, l.array());
	}
	else {
		blas::packl(n, pa->array, l.array());
	}

	return l.get();
}

AddIn xai_unpack(
	Function(XLL_FPX, "xll_unpack", "UNPACK")
	.Arguments({
		Arg(XLL_FPX, "L", "is a packed matrix."),
		})
		.FunctionHelp("Unpack L into symmetric A.")
	.Category(CATEGORY)
);
_FPX* WINAPI xll_unpack(_FPX* pl)
{
#pragma XLLEXPORT
	static FPX a;
	int m = size(*pl);

	// m = n(n+1)/2
	// n^2 + n - 2m = 0
	// b^2 - 4ac = 1 + 8m
	auto d = sqrt(1 + 8 * m);
	int n = static_cast<int>((-1 + d) / 2);
	a.resize(n, n);

	blas::unpack(n, pl->array, a.array());

	return a.get();
}

#define POTRF_TOPIC "https://software.intel.com/content/www/us/en/develop/documentation/" \
	"onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/" \
	"lapack-linear-equation-computational-routines/matrix-factorization-lapack-computational-routines/potrf.html"

AddIn xai_lapack_potrf(
	Function(XLL_FPX, "xll_lapack_potrf", "LAPACK.POTRF")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_BOOL, "_upper", "is an optional boolean indicating lower decomposition. Default is lower."),
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
_FPX* WINAPI xll_lapack_potrf(_FPX* pa, BOOL upper, BOOL nofill)
{
#pragma XLLEXPORT
	try {
		ensure(pa->columns == pa->rows); // fix up ld???

		auto a = fpmatrix(pa);
		lapack::potrf(upper ? CblasUpper : CblasLower, a);

		if (nofill == FALSE) {
			if (upper) {
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
		// a[upper] = 0 ???
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}

#define PPTRF_TOPIC "https://software.intel.com/content/www/us/en/develop/documentation/" \
	"onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/" \
	"lapack-linear-equation-computational-routines/matrix-factorization-lapack-computational-routines/pptrf.html"

AddIn xai_lapack_pptrf(
	Function(XLL_FPX, "xll_lapack_pptrf", "LAPACK.PPTRF")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_BOOL, "_upper", "is an optional boolean indicating upper packed matrix. Default is lower."),
		})
		.Category(CATEGORY)
	.FunctionHelp("Return the packed Cholesky decomposition of a.")
	//.HelpTopic(PPTRF_TOPIC)
	.Documentation(R"(
This function calculates the Cholesky decomposition of a symmetric positive definite matrix \(A\).
The upper decomposition statisfies \(A = U' U\) and the lower satisifes \(A = L L'\) where
prime indicates matrix transpose. Matrix must be in packed format
)")
);
_FPX* WINAPI xll_lapack_pptrf(_FPX* pa, BOOL upper)
{
#pragma XLLEXPORT
	try {
		int m = size(*pa);

		// m = n(n+1)/2
		// n^2 + n - 2m = 0
		// b^2 - 4ac = 1 + 8m
		auto d = sqrt(1 + 8 * m);
		int n = static_cast<int>((-1 + d) / 2);

		auto a = blas::matrix(n, n, pa->array);
		lapack::pptrf(upper ? CblasUpper : CblasLower, a);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}
#if 0
#endif // 0

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
		lapack::potri(lower ? CblasLower : CblasUpper, a);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pa;
}

AddIn xai_lapack_quad(
	Function(XLL_DOUBLE, "xll_lapack_quad", "LAPACK.QUAD")
	.Arguments({
		Arg(XLL_FPX, "A", "is a symmetric matrix."),
		Arg(XLL_FPX, "x", "is a vector."),
		Arg(XLL_BOOL, "_upper", "is an optional argument indicating upper triangular portion of A should be used."),
		})
	.Category(CATEGORY)
	.FunctionHelp("Return the x'Ax.")
	//.HelpTopic(quad_TOPIC)
	.Documentation(R"(

)")
);
double WINAPI xll_lapack_quad(_FPX* pa, _FPX* px, BOOL upper)
{
#pragma XLLEXPORT
	double result = XLL_NAN;
	try {
		int n = size(*px);
		ensure(n == pa->columns);
		ensure(n == pa->rows);

		result = blas::quad(upper ? CblasUpper : CblasLower, blas::matrix(n,n,pa->array), blas::vector(n, px->array));
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}

	return result;
}