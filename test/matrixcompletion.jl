# --------------------------------------------------------------
# Test level-set algorithm on matrix completion
# --------------------------------------------------------------
using AtomicOpt
using LinearAlgebra
using Printf
import Random: seed!, randperm
seed!(0)

n, m, k = 32, 32, 5               # matrix is n-by-m, rank is k
M = randn(800, n*m)               # measurement matrix
u = randn(n, k); v = randn(m, k); 
x0 = vec(u*v')                    # k-rank solution   
b = M*x0                          # The right-hand side corresponding to x0
A = NucBall(n, m; maxrank = k)    # Atomic set

x, normr = @time level_set(M, b, A, maxIts=100000)
@printf("Relative difference between x and x0: %8.2e", norm(x - x0)/ norm(x0))