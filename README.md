# AtomicOpt.jl
AtomicOpt.jl is a Julia package for solving the following non-convex structured optimization problem:

```
    Find        x ∈ ℝⁿ  
    subject to  ½‖Mx-b‖² ≤ α 
    and         rank(x|A) ≤ k
```
where `M: ℝⁿ -> ℝᵐ` is a linear operator, `b ∈ ℝᵐ` is the observation vector, `A ⊆ ℝⁿ` is a atomic set and `rank(x|A)` measures the complexity of `x` with respect to the atomic set `A`. For example, when `A` is the set of all signed canonical vectors, i.e., `A = {±e₁, ..., ±eₙ}`, then `rank(x|A)` equals to the number of nonzero entries in `x`. When `A` is the set of all normalized rank one matrices, i.e., `A = { uv^T | u ∈ ℝᵐ, v ∈ ℝⁿ, ||u||₂ = ||v||₂ ≤ 1 }`, then `rank(x|A)` equals to the rank of the matrix `x`. Please see our [paper](https://friedlander.io/publications/2019-polar-alignment-atomic-decomp/) for more detailed discussion on atomic sparsity. 

## Installation
To install, just call
```julia
Pkg.add("https://github.com/MPF-Optimization-Laboratory/AtomicOpt.jl.git")
```

## Use AtomicOpt.jl to solve basis pursuit problem
```julia
using AtomicOpt
using LinearAlgebra
using Printf
import Random: seed!, randperm
m, n, k = 2^8, 2^10, 8            # No. of rows, columns, and nonzeros
M = randn(m, n)                   # Measurement operator
p = randperm(n); p = p[1:k]       # Location of k nonzeros in x
u = randn(m)/100                  # Noise
b = M*x0 + u                      # Observation
A = OneBall(n; maxrank = k)       # Atomic set
# Solve the basis pursuit problem
sol = level_set(M, b, A, α = norm(u)^2/2)
x = constructPrimal(sol)
# Report recovery error
@printf("relative difference between x0 and x: .2%f", norm(x - x0)/norm(x0))
```

## Use AtomicOpt.jl to solve matrix completion problem
```julia
using AtomicOpt
using Printf
using LinearAlgebra
using SparseArrays
# generate a random m×n matrix with rank r
m, n, r = 100, 100, 3 
X = rand(m, n)
U, S, V = svd(X)
X0 = U[:,1:r] * Diagonal( S[1:r] ) * V[:,1:r]'
# generate a random mask with nnz ≈ m*n*p
p = 0.5
mask = sprand(Bool, m, n, p); mask = convert(SparseMatrixCSC{Float64, Int64}, mask)
# measurement
B =  mask .* X0
b =  B.nzval
# operator
Mop = MaskOP(mask)
# atomic set
A = NucBall(m, n, r)
# solve the matrix completion problem
sol = level_set(M, b, A, α = 0.0)
x = constructPrimal(sol)
X = reshape(x, m, n)
# Report recovery error
@printf("relative difference between X0 and X: .2%f", norm(X - X0)/norm(X0))
```

## More examples
There are more examples in the folder `examples`.

# Citing this package

If you use AtomicOpt.jl for published work,
we encourage you to cite the software.

Use the following BibTeX citation:

```bibtex
@article{fan2020polar,
  title={Polar Deconvolution of Mixed Signals},
  author={Fan, Zhenan and Jeong, Halyun and Joshi, Babhru and Friedlander, Michael P},
  journal={arXiv preprint arXiv:2010.10508},
  year={2020}
}
```
