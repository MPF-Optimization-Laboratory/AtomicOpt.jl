########################################################################
# Atoms of the trace-norm ball.
########################################################################
"""
    TraceBallAtom(n, v)

Atom in the TraceBall of dimension n×n.
"""
struct TraceBallAtom <: AbstractAtom
    n::Int64
    v::Vector{Float64}
    function NucBallAtom(v)
        n = length(v)
        new(n, v)
    end
end

"""
    *(M::AbstractLinearOp, a::TraceBallAtom) -> Y

Multiply a trace-norm atom by a linear map, and return the matrix `Y`.
"""
Base.:(*)(M::AbstractLinearOp, a::TraceBallAtom) = M*vec(a.v*a.v')
Base.:(*)(M::AbstractOperator, a::TraceBallAtom) = M*a.v
Base.vec(a::TraceBallAtom) = vec(a.v*a.v')
Base.length(a::TraceBallAtom) = a.n * a.n


"""
    TraceBall(n)

Atomic set defined by trace-norm ball in `n by n` matrices.
"""
struct TraceBall <: AbstractAtomicSet
    n::Int64
    function TraceBall(n)
        n < 1 && error("n must be a positive integer")
        return new(n)
    end
end

# """
# Multiply the trace-norm ball by a scalar.
# """
# Base.:(*)(τ::Real, A::OneBall) = TraceBall(A.n, τ*A.λ)

gauge(A::TraceBall, x) = tr(x)/A.λ
support(A::TraceBall, z) = real(eigs(z, nev = 1, which=:LR)[1][1])*A.λ

function expose(A::TraceBall, z::Union{AbstractMatrix, LinearMap})
    ~, v = eigs(z, nev = 1, which=:LR);
    return AtomTraceBall(vec(real(v)))
end

"""
    support identification
"""
function support_identification(A::TraceBall, M::AbstractOperator, z::LinearMap)
    r = 3
    ~, V = eigs(z, nev = r, which=:LR)
    mask = M.mask
    (n, m, L) = size(mask)
    W = complex(zeros(n*m*L, r))
    for i = 1:r
        v = V[:, i]
        v = reshape(v, n, m)
        w = broadcast(*, mask, v)
        w = fft(w, (1, 2))
        W[:,i] = vec(w)
    end
    W = W / sqrt(n*m)
    return V, W
end

Base.length(A::TraceBall) = A.n
atom_name(A::TraceBall) = "trace-norm ball"
atom_description(A::TraceBall) = "Normalized psd matrices"
atom_parameters(A::TraceBall) = "n = $(length(A))"

