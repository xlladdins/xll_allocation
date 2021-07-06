# Allocation

Optimal asset allocation. https://arxiv.org/pdf/2009.10852.pdf

One period model $x\in R^n \mapsto X\colon\Omega\to R^n$, $P$ a probability measure on $\Omega$.

Find minimum variance portfolio $\xi\in R^n$ with initial price 1 having
expected realized return $R$.

$\min_{\xi\in R^n} \xi' V \xi/2 - \lambda(\xi'x - 1) - \mu(\xi' E[X] - R)$ where $V = Cov(X, X)$.

The $i$-th row vector of the Lower Cholesky decomposition of $V$ norm $\sigma_i^2 = \Var(X_i)$.