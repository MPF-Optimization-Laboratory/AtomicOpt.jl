push!(LOAD_PATH, pwd())
using AtomicOpt
using Printf
using LinearAlgebra
import Random: seed!, randperm, shuffle
using Distributed
using BSON: @save, @load

function randop(n::Integer, family=:orth)
    if family == :orth
        Q = Matrix(qr(randn(n,n)).Q)
    end
    return Q
end

function randnsparse(n, nnz)
    p = randperm(n)[1:nnz]
    x = zeros(n)
    x[p] = rand(nnz)
    return x
end

struct Signal{opT, atomT}
    n::Int64
    nnz::Int64
    x::Vector{Float64}
    Q::opT
    A::atomT
end

function Signal(n, nnz; family=:orth)
    x = randnsparse(n, nnz)
    Q = randop(n, family)
    λ = norm(x, 1)
    A = λ*(Q*OneBall(n; maxrank = nnz))
    return Signal(n, nnz, x, Q, A)
end

rotate(s::Signal) = s.Q*s.x

function RecordError(m, n, k, nnz, t)
    signals = ntuple(d->Signal(n, nnz), k)
    M = rand(m, n)
    b = M*sum(rotate(s) for s in signals)
    A = mapreduce(s->s.A, ×, signals)
    x, normr = level_set(M, b, A,  α = 0.0, tol = 1e-3, maxIts=10*length(b), logger=0)
    err = 0
    for (i,(x,s)) in enumerate(zip(x, signals))
        err1 = norm(x-s.Q*s.x); 
        if err1 > err
            err = err1
        end
    end
    @printf("  m = %3d, n = %3d, k = %1d, nnz = %1d, t = %2d, error = %8.2e\n", m, n, k, nnz, t, err)
    flush(stdout)
    return err
end

function RecordError2(m, n, k, nnz, α, t)
    signals = ntuple(d->Signal(n, nnz), k)
    M = randn(m, n)
    b = M*sum(rotate(s) for s in signals)
    u = randn(m); u .*= α / norm(u) 
    b = b + u
    A = mapreduce(s->s.A, ×, signals)
    x, normr = level_set(M, b, A,  α = α^2/2, tol = 1e-3, maxIts=10*length(b), logger=0)
    err = 0
    for (i,(x,s)) in enumerate(zip(x, signals))
        err1 = norm(x-s.Q*s.x); 
        if err1 > err
            err = err1
        end
    end
    @printf("  α = %8.2e, t = %3d\n", α, t)
    flush(stdout)
    return err
end

# function AvgError(m, n, k, nnz)
#     out = @distributed (+) for i = 1:100
#         RecordError(m, n, k, nnz)
#     end
#     @printf("  m = %7d, n = %7d, k = %7d, nnz = %7d, error = %8.2e\n", m, n, k, nnz, out/100)
#     flush(stdout)
#     return out/100
# end

# @show RecordError(200, 200, 4, 5, 1)

# m = 125
# n = 200
# k = 3
# nnz = 5
# J = CartesianIndices((31, 31, 50))
# @printf("Start E2!\n")
# E2 = pmap(i->RecordError(500 - (i[1] - 1)*15, 50+(i[2]-1)*15, 2, nnz, i[3]), J)
# @printf("Start E3!\n")
# E3 = pmap(i->RecordError(500 - (i[1] - 1)*15, 50+(i[2]-1)*15, 3, nnz, i[3]), J)
# @printf("Start E4!\n")
# E4 = pmap(i->RecordError(500 - (i[1] - 1)*15, 50+(i[2]-1)*15, 4, nnz, i[3]), J)
# J = CartesianIndices((200, 50))
# E = pmap(i->RecordError2(m, n, k, nnz, 0.01*i[1], i[2]), J)

# # save
# @save "./examples/PhaseTransition/NoisyMeasurement.bson" E
# @save "./examples/PhaseTransition/NoiselessMeasurement.bson" E2 E3 E4

nnz = 3
n = 1000
J = CartesianIndices((19, 9, 10))
E = pmap(i->RecordError(1000 - (i[1] - 1)*50, n, i[2]+1, nnz, i[3]), J)
@save "./examples/PhaseTransition/NoiselessMeasurement2.bson" E