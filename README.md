# Allocation

Optimal asset allocation. https://arxiv.org/pdf/2009.10852.pdf

One period model: $x\in R^n \mapsto X\colon\Omega\to R^n$, $P$ a probability measure on $\Omega$.

Define $R_\xi = \xi'X/\xi'x$ to be the realized return for portfolio $\xi\in R^n$.

Find minimum variance portfolio $\xi\in R^n$ with initial price 1 and
expected realized return $\rho$.

$\min_{\xi\in R^n} \xi' \Sigma \xi/2 - \lambda(\xi'x - 1) - \mu(\xi' E[X] - \rho)$ where $V = Cov(X, X)$.

We may, and do, assume $x = 1 = [1 ... 1]$.

ER	2	3
Sigma	4	5
	4	0
	0	5
G	1	0.6
	0.6	1
		
	16	12
	12	25
		
	0.09765625	-0.046875
	-0.046875	0.0625
		
x	1	1
V_x	0.05078125	
	0.015625	
V_EX	0.0546875	
	0.09375	
A	0.06640625	
B	0.1484375	
C	0.390625	
D	0.00390625	


