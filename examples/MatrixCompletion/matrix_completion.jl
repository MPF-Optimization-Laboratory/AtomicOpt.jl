using AtomicOpt
using Printf
using LinearAlgebra
using SparseArrays
import Random: seed!, randperm, shuffle

# config
m, n, r = 30, 30, 3

# generate a random m×n matrix with rank r
X = rand(m, n)
U, S, V = svd(X)
X = U[:,1:r] * Diagonal( S[1:r] ) * V[:,1:r]'

# generate a random mask with nnz ≈ m*n*p
p = 0.3
mask = sprand(Bool, m, n, p); mask = convert(SparseMatrixCSC{Float64, Int64}, mask)

# measurement
B =  mask .* X
# η = rand(nnz(B)); ϵ = 0.01; η .*= (ϵ/norm(η))
b =  B.nzval

# operator
Mop = MaskOP(mask)

# construct atomic set
A = NucBall(m, n, r)
x, normr = level_set(Mop, b, A,  α = 0.0, tol = 1e-3, maxIts=10*length(b), logger=1)