# Notes

$\max \xi'E[X] - \lambda(\xi'x - 1) - \mu/2(\xi'V\xi - \sigma^2)$

$0 = E[X] - \lambda x - \mu V\xi$

$\xi = (1/\mu)(V^{-1}E[X] - \lambda V^{-1}x)$

$1 = \xi'x = (1/\mu)(E[X]'V^{-1}x - \lambda x'V^{-1}x) = (1/\mu)(B - \lamba A)$

$\sigma^2 = \xi'V\xi = \xi'/\mu (E[X] - \lambda x) = (1/\mu^2)(C - 2\lambda B + \lambda^2 A)$

$0 = (C - 2\lambda B + \lambda^2 A) - \sigma^2(B^2 - 2\lambda AB + \lambda^2 A^2)$

$0 = (C - \sigma^2 B^2) - 2(B - \sigma^2 AB)\lambda + (A - \sigma^2 A^2)\lambda^2$