// fms_mkl.h - Intel MKL wrappers
// https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=window&distributions=webdownload&options=offline
#pragma once
#include <valarray>
#include <mkl.h>

namespace mkl {

	// r x c matrix
	template<class X = double>
	class matrix {
		int r, c; // leading dimension???
		X* a = nullptr;
		CBLAS_TRANSPOSE t = CblasNoTrans; // 'N', CblasTrans 'T', CblasConjTrans 'C'
		CBLAS_UPLO ul = (CBLAS_UPLO)0; // CblasUpper 'U', CblasLower 'L'
		CBLAS_DIAG diag = (CBLAS_DIAG)0; //  CblasNonUnit 'N' or CblasUnit 'U'
		bool pack = false;
		int index(int i, int j) const
		{
			if (t == CblasTrans)
				std::swap(i, j);

			if (!pack) {
				if (!ul
					or ul == CblasUpper and i <= j
					or ul == CblasLower and i >= j
					or diag == CblasUnit and i != j) {

					return j + i * c;
				}
			}
			else {
				if (ul == CblasUpper and i <= j)
					return i - 1 + j * (j - 1) / 2;
				else if (ul == CblasUpper and i >= j)
					return i - 1 + (2 * c - j) * (j - 1) / 2;
			}

			return -1;
		}
	public:
		matrix()
			: r(0), c(0), a(nullptr)
		{ }
		matrix(int r, int c)
			: r(r), c(c), a(nullptr)
		{ }
		matrix(int r, int c, X* a)
			: r(r), c(c), a(a)
		{ }
		matrix(const matrix&) = default;
		matrix& operator=(const matrix&) = default;
		matrix(matrix&&) = default;
		matrix& operator=(matrix&&) = default;
		~matrix()
		{ }


		int rows() const
		{
			return t == CblasNoTrans ? r : c;
		}
		int columns() const
		{
			return t == CblasNoTrans ? c : r;
		}
		int size() const
		{
			return r * c;
		}
		X* data()
		{
			return a;
		}
		const X* data() const
		{
			return a;
		}
		CBLAS_TRANSPOSE trans() const
		{
			return t;
		}
		matrix& trans(CBLAS_TRANSPOSE _t)
		{
			t = _t;

			return *this;
		}
		matrix& transpose()
		{
			if (t == CblasNoTrans)
				t = CblasTrans;
			else if (t == CblasTrans)
				t = CblasNoTrans;

			return *this;
		}
		CBLAS_UPLO uplo() const
		{
			return ul;
		}
		CBLAS_DIAG unit() const
		{
			return diag;
		}
		bool packed() const
		{
			return pack;
		}

		X operator()(int i, int j) const
		{
			return a[index(i, j)];
		}
		X& operator()(int i, int j)
		{
			return a[index(i, j)];
		}
	};

	namespace blas {

		// c = alpha a*b + beta c, a m x k, b k x n, c m x n
		inline void gemm(int m, const double* a, int k, const double* b, int n, double* c, 
			double alpha = 1, double beta = 0, 
			CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans)
		{
			cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, alpha, a, k, b, n, beta, c, n);
		}

		template<class X>
		inline matrix<double> mm(const matrix<X>& a, const matrix<X>& b, double* pc)
		{
			matrix<double> c(a.rows(), b.columns(), pc);

			if (!a.packed() and !b.packed()) {
				gemm(a.rows(), a.data(), a.columns(), b.data(), b.columns(), c.data(), 1, 0, a.trans(), b.trans());
			}

			return c;
		}

	}

	namespace lapack {

		// Invert upper triangular matrix
		inline int tptri(int n, double* a, char diag = 'N')
		{
			return LAPACKE_dtptri(LAPACK_ROW_MAJOR, 'U', diag, n, a);
		}

		// Cholesky decomposition a = l'l
		inline int potrf(int n, double* a, int lda = 0)
		{
			if (lda == 0) {
				lda = n;
			}

			return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', n, a, lda);
		}

	}

} // namespace mkl
