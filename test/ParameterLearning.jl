# ----------------------------------------------------------------------
# Test level-set algorithm on parameter learning
# ----------------------------------------------------------------------
using AtomicOpt
using Printf
using LinearAlgebra
import Random: seed!, randperm, shuffle
using ForwardDiff

function randop(n::Integer, family=:orth)
    if family == :orth
        M = Matrix(qr(randn(n,n)).Q)
    end
    return M
end

function randnsparse(n, k)
    p = randperm(n)[1:k]
    x = zeros(n)
    x[p] = randn(k)
    return x
end

n, k = 512, 8
x1 = randnsparse(n, k); Q1 = randop(n); λ1 = 1/norm(x1, 1); x1 = Q1*x1; A1 = Q1*OneBall(n; maxrank = k)
x2 = 2*randnsparse(n, k); Q2 = randop(n); λ2 = 1/norm(x2, 1); x2 = Q2*x2; A2 = Q2*OneBall(n; maxrank = k)
x3 = 3*randnsparse(n, k); Q3 = randop(n); λ3 = 1/norm(x3, 1); x3 = Q3*x3; A3 = Q3*OneBall(n; maxrank = k)
η = randn(n)/n; α = norm(η)^2/2
b = x1 + x2 + x3 + η
λ_ideal = [λ1/(λ1+λ2+λ3); λ2/(λ1+λ2+λ3); λ3/(λ1+λ2+λ3)]

function f(λ::AbstractVector{T}) where T
    x, τ = level_set(I(n), b, λ[1]*A1+λ[2]*A2+λ[3]*A3, α=α, tol = 1e-6, pr=false, maxIts=10000, logger=0)
    return τ
end

g = λ -> ForwardDiff.gradient(f, λ)

function cg(λ, maxiter)
    k = 1
    n = length(λ)
    A = PosSimplex(n)
    gap = Inf
    while k ≤ maxiter
        k += 1
        # @show k
        z = g(λ)
        a = expose(A, z)
        gap = (vec(a) - λ)'*z
        # @show gap
        if gap < 1e-6
            break
        end
        γ = 2/(k+1)
        λ = (1-γ)*λ + γ*vec(a)
    end
    return λ, gap
end

# λ = rand(3); λ ./= sum(λ)
# λ = λ_ideal
# λ_cg, gap = cg(λ, 100)

function projsplx(λ::Vector)
    n = length(λ)
    bget = false
    idx = sortperm(λ, rev=true)
    tsum = 0.0
    @inbounds for i = 1:n-1
        tsum += λ[idx[i]]
        tmax = (tsum - 1.0)/i
        if tmax ≥ λ[idx[i+1]]
            bget = true
            break
        end
    end
    if !bget
        tmax = (tsum + λ[idx[n]] - 1.0) / n
    end
    @inbounds for i = 1:n
        λ[i] = max(λ[i] - tmax, 0)
    end
    return λ
end

function pg(λ, maxiter)
    k = 1
    α = 0.0001
    n = length(λ)
    while k ≤ maxiter
        # @show k
        k += 1
        z = g(λ)
        λ_new = projsplx(λ-α*z)
        if norm(λ_new - λ) < 1e-6
            break
        end
        λ = λ_new
    end
    return λ
end

# λ = rand(3); λ ./= sum(λ)
λ = λ_ideal + rand(3)
λ_pg = pg(λ, 1000)

@show λ_ideal, f(λ_ideal)
# @show λ_cg, f(λ_cg)
@show λ_pg, f(λ_pg);
