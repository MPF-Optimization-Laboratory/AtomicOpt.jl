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
η = rand(nnz(B)); ϵ = 1.0; η .*= (ϵ/norm(η))
b =  B.nzval + η
α = ϵ^2/2

# operator
Mop = MaskOP(mask)

# construct atomic set
A = NucBall(m, n, 2*r)

# optimal parameters
αopt = ϵ^2/2
τopt = sum(S[1:r])
λopt = support(A, Mop'η)



# solve the problem
# sol = level_set(Mop, b, A, α = αopt, tol = 1e-3, maxIts=10*length(b))
sol = level_set(Mop, b, A, α = αopt, tol = 1e-3, maxIts=10*length(b), rule="bisection", τmax=1.5*τopt)
# sol = conditional_graident(Mop, b, A, τopt, tol = 1e-3, α = αopt, maxIts=10*length(b))
# sol = coordinate_descent(Mop, b, A, λopt, tol = 1e-3, α = αopt, maxIts=10*length(b))

# reconstruct primal variable
x = constructPrimal(sol)
X_recover = reshape(x, m, n)

@printf "norm of b = %e \n" norm(b)
@printf "The relative difference between X and X_recover: %e \n" norm(X - X_recover)/norm(X)