# Allocation

Optimal asset allocation. https://arxiv.org/pdf/2009.10852.pdf

One period model: $x\in R^n \mapsto X\colon\Omega\to R^n$, $P$ a probability measure on $\Omega$.

Define $R_\xi = \xi'X/\xi'x$ to be the realized return for portfolio $\xi\in R^n$.

Find minimum variance portfolio $\xi\in R^n$ with initial price 1 and
expected realized return $\rho$.

$\min_{\xi\in R^n} \xi' \Sigma \xi/2 - \lambda(\xi'x - 1) - \mu(\xi' E[X] - \rho)$ where $V = Cov(X, X)$.

We may, and do, assume $x = 1 = [1 ... 1]$.

