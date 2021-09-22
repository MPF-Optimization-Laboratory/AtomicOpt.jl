using AtomicOpt
using LinearAlgebra
using Printf

n = 500 # dimension of x
m = 250 # number of equality constraints
c = rand(n) + 0.5*ones(n) # create nonnegative cost vector
x0 = abs.(randn(n)) # create random solution vector
M = abs.(randn(m, n)) # create random, nonnegative matrix M
b = M*x0
A = Diagonal(ones(n)./c)*PosSimplex(n, maxrank=m) # create atomic set as scaled positive simplex

x, ~ = level_set(M, b, A, α=0.0, tol=1e-5, maxIts=500000, logger=2)
@printf("objective value = %8.2e, relative infeasibility = %8.2e", c'*x, norm(M*x - b) / max(1.0, norm(b)))

# using Convex
# using SCS
# xcvx = Variable(n)
# problem = minimize(c'*xcvx, [M*xcvx == b, xcvx >= zeros(n)])
# solve!(problem,() -> SCS.Optimizer(verbose=0))
# obj = c'*xcvx.value; obj = obj[1]
# infeas = norm(M*xcvx.value - b) / max(1.0, norm(b)); infeas = infeas[1]
# @printf("objective value = %8.2e, relative infeasibility = %8.2e", obj, infeas)
