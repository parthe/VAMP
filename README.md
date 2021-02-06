# Vector Approximate Message Passing (VAMP)

Iterative algorithm for solving linear inverse problems based on proximal operations

https://arxiv.org/abs/1610.03082 (Rangan, Schniter, Fletcher)

Currently supports solving linear inverse problems of the form
```math
minimize_x \frac12 ||y-Ax||^2 + \rho(x)
```
