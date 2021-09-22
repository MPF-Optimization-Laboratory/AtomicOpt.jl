# ----------------------------------------------------------------------
# Test level-set algorithm on demixing random rotated sparse signals
# ----------------------------------------------------------------------
using AtomicOpt
using Printf
using LinearAlgebra
import Random: seed!, randperm, shuffle
# seed!(3)

"Generate random invertible operator."
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

struct Signal{opT, atomT}
    n::Int64
    k::Int64
    x::Vector{Float64}
    Q::opT
    A::atomT
end

function Signal(n, k; family=:orth)
    x = randnsparse(n, k)
    Q = randop(n, family)
    λ = norm(x, 1)
    A = λ*(Q*OneBall(n; maxrank = k))
    return Signal(n, k, x, Q, A)
end

rotate(s::Signal) = s.Q*s.x

# number of signals, signal length, and sparsity level
N, n, k = 5, 512, 10

# generate collection of `N` signals
signals = ntuple(d->Signal(n,k),N)

# superpose rotated signals: b = Q₁x₁ + Q₂x₂ + ⋯ + Qnxn
b = sum(rotate(s) for s in signals)

# noise
# u = randn(n)/10; 
u = zeros(n)
b = b + u

# cross product of scaled and rotated atomic sets
A = mapreduce(s->s.A, ×, signals)

# demixing
x, τ = @time level_set(I(n), b, A, α = norm(u)^2/2, tol = 1e-6, maxIts=6000)
# x, normr = @trace level_set(I(n), b, A, log=true, maxIts=6000)  modules=[PolarDemixing, AtomicSets]
for (i,(x,s)) in enumerate(zip(x, signals))
    if norm(u) > 1e-6
        err = norm(s.Q'x-s.x)/norm(u)
    else
        err = norm(s.Q'x-s.x)
    end
    @printf("  Relative error in signal %2d: %8.2e\n", i, err)
end
