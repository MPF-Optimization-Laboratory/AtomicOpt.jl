# --------------------------------------------------------------
# Test level-set algorithm on basis pursuit
# --------------------------------------------------------------
using AtomicOpt
using LinearAlgebra
using Printf
import Random: seed!, randperm
# seed!(0)

m, n, k = 2^8, 2^10, 8            # No. of rows, columns, and nonzeros
M = randn(m, n)                   # ... M is m-by-n
p = randperm(n); p = p[1:k]       # Location of k nonzeros in x
x0 = zeros(n); x0[p] = randn(k)   # k-sparse solution
u = randn(m)/100                  # noise
# u = zeros(m)
b = M*x0 + u                      # The right-hand side corresponding to x0
A = OneBall(n; maxrank = k)       # Atomic set

x, τ = level_set(M, b, A, α = norm(u)^2/2, tol = 1e-9, maxIts=10000)
# @printf("Relative difference between x and x0: %8.2e", norm(x - x0))

using Convex
using SCS

xcvx = Variable(n)
problem = minimize(sum(abs.(xcvx)), [norm(M*xcvx - b) <=  norm(u)])
solve!(problem, SCS.Optimizer(max_iters=3000))

@show norm(x, 1)
@show norm(xcvx.value, 1)
@show norm(M*x - b) / (1 + norm(b))
# @show norm(u)

