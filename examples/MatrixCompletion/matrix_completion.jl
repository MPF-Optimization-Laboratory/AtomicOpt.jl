using AtomicOpt
using Printf
using LinearAlgebra
using SparseArrays
import Random: seed!, randperm, shuffle

# config
m, n, r = 100, 100, 3

# generate a random m×n matrix with rank r
X = rand(m, n)
U, S, V = svd(X)
X = U[:,1:r] * Diagonal( S[1:r] ) * V[:,1:r]'

# generate a random mask with nnz ≈ m*n*p
p = 0.5
mask = sprand(Bool, m, n, p); mask = convert(SparseMatrixCSC{Float64, Int64}, mask)

# measurement
B =  mask .* X
# η = rand(nnz(B)); ϵ = 0.01; η .*= (ϵ/norm(η))
b =  B.nzval

# operator
Mop = MaskOP(mask)

# τmax
τmax = 1.2 * sum(S[1:r])
@show τmax

# construct atomic set
A = NucBall(m, n, 2*r)
x, normr = level_set(Mop, b, A, α = 0.0, tol = 1e-3, maxIts=10*length(b))
# x, normr = level_set_bisection(Mop, b, A, τmax, α = 1e-2, tol = 1e-3, maxIts=10*length(b))
@printf "norm of b = %e \n" norm(b)
@show typeof(x)
X_recover = reshape(x, m, n)

@printf "The relative Frobenius norm of (X - X_recover) = %e \n" norm(X - X_recover)/norm(X)