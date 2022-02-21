########################################################################
# Atoms of the trace-norm ball.
########################################################################
"""
    TraceBallAtom(n, v)

Atom in the TraceBall of dimension n×n.
"""
struct TraceBallAtom{T1<:Int64, T2<:Vector{Float64}} <: AbstractAtom
    n::T1
    v::T2
    function TraceBallAtom(v::Vector{Float64})
        n = length(v)
        new{Int64, Vector{Float64}}(n, v)
    end
end

"""
    *(M::AbstractLinearOp, a::TraceBallAtom) -> Y

Multiply a trace-norm atom by a linear map, and return the matrix `Y`.
"""
Base.:(*)(M::LinearOp, a::TraceBallAtom{Int64, Vector{Float64}}) = M*vec(a.v*a.v')
LinearAlgebra.mul!(Ma::Vector{Float64}, M::LinearOp, a::TraceBallAtom{Int64, Vector{Float64}}) = mul!(Ma, M, vec(a.v*a.v'))
Base.vec(a::TraceBallAtom{Int64, Vector{Float64}}) = vec(a.v*a.v')
Base.length(a::TraceBallAtom{Int64, Vector{Float64}}) = a.n * a.n
Base.size(a::TraceBallAtom{Int64, Vector{Float64}}) = a.n, a.n


"""
    TraceBall(n)

Atomic set defined by trace-norm ball in `n by n` matrices.
"""
struct TraceBall{T<:Int64} <: AbstractAtomicSet
    n::T
    maxrank::T
    function TraceBall(n::Int64, maxrank::Int64)
        n ≥ 1 || throw(DomainError(n,"n must be a positive integer"))
        n ≥ maxrank ≥ 0 || throw(DomainError(maxrank,"maxrank must be ≥ 0"))
        new{Int64}(n, maxrank)
    end
end
TraceBall(n::Int64; maxrank=n) = TraceBall(n, maxrank)

"""
    gauge(A::TraceBall, x::Matrix)

Gives the trace-norm of `x`.
"""
gauge(::TraceBall{Int64}, x::Matrix{Float64}) = tr(x)
gauge(A::TraceBall{Int64}, x::Vector{Float64}) = gauge(A, reshape(x, A.n, A.n))


"""
    support(A::TraceBall, z::AbstractMatrix)

Gives the largest singular value of `z`.
"""
support(::TraceBall{Int64}, z::AbstractMatrix{Float64}) = real(eigs(z, nev = 1, which=:LR)[1][1])
support(A::TraceBall{Int64}, z::AbstractVector{Float64}) = support(A, reshape(z, A.n, A.n))


"""
    expose!(A::TraceBall, z::Matrix, a::TraceBallAtom)

Gives the top eigen vector of `z`.
"""
function expose!(::TraceBall{Int64}, z::AbstractMatrix{Float64}, a::TraceBallAtom{Int64, Vector{Float64}})
    ~, v = eigs(z, nev = 1, which=:LR)
    a.v .= v[:]
    return nothing
end
expose!(A::TraceBall{Int64}, z::AbstractVector{Float64}, a::TraceBallAtom{Int64, Vector{Float64}}) = expose!(A, reshape(z, A.n, A.n), a)

function expose(A::TraceBall{Int64}, z::AbstractMatrix{Float64})
    n = A.n
    a = TraceBallAtom(zeros(n))
    expose!(A, z, a)
    return a
end
expose(A::TraceBall{Int64}, z::AbstractVector{Float64}) = expose(A, reshape(z, A.n, A.n))

Base.length(A::TraceBall{Int64}) = A.n*A.n
Base.size(A::TraceBall{Int64}) = A.n, A.n
rank(A::TraceBall{Int64}) = A.maxrank
atom_name(A::TraceBall{Int64}) = "trace-norm ball"
atom_description(A::TraceBall{Int64}) = "{ vv^T | v ∈ ℝⁿ, ||v||₂ ≤ 1 }"
atom_parameters(A::TraceBall{Int64}) = "n = $(A.n), maxrank = $(A.maxrank)"

########################################################################
# Face of the trace-norm ball.
########################################################################

mutable struct TraceBallFace{T1<:Int64, T2<:Matrix{Float64}} <: AbstractFace
    n::T1
    r::T1
    V::T2
    function TraceBallFace(V::Matrix{Float64})
        n, r = size(V)
        new{Int64, Matrix{Float64}}(n, r, V)
    end
end

"""
    face(A, z) -> F

Return a face `F` of the atomic set `A` exposed by the vector `z`. The dimension of
the exposed face is limited by `k=maxrank(A)`. If the dimension is being limited
by `k`, then the routine returns a set of rank `k` atoms that define the subset
of the exposed face.
"""
function face!(A::TraceBall{Int64}, z::AbstractMatrix{Float64}, F::TraceBallFace{Int64, Matrix{Float64}})
    r = A.maxrank
    if r == 0   
        V = zeros(A.n, 1)
    elseif r < min(A.n) - 1
        ~, V = eigs(z, nev=r, which=:LR)
        F.V .= V
    else
        V = eigvecs(z)
        F.V .= V[:, 1:r]
    end
    return nothing
end
face!(A::TraceBall{Int64}, z::AbstractVector{Float64}, F::TraceBallFace{Int64, Matrix{Float64}}) = face!(A, reshape(z, A.n, A.n), F)

function face(A::TraceBall{Int64}, z::AbstractMatrix{Float64})
    n, r = A.n, max(A.maxrank, 1)
    V = zeros(n, r)
    F = TraceBallFace(V)
    face!(A, z, F)
    return F
end
face(A::TraceBall{Int64}, z::AbstractVector{Float64}) = face(A, reshape(z, A.n, A.n))


"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::TraceBallFace{Int64, Matrix{Float64}}, S::Matrix{Float64}) = vec(F.U*S*F.U')
Base.:(*)(F::TraceBallFace{Int64, Matrix{Float64}}, c::Vector{Float64}) = F*reshape(c, F.r, F.r)

"""
    *(M::LinearOp, F::TraceBallFace)

Return a LinearMap `L` whose forward and adjoing operators
are defined as

    L*p = M*vec(V*matrix(p)*V')
    L'*q = vec(V'*matrix(M'*q)*V)

where the `n`-by-`k` matrice `V` is the left
eigen vectors that define the face `F` of the trace-norm ball.
"""
function Base.:(*)(M::LinearOp, F::TraceBallFace{Int64, Matrix{Float64}}) 
    n, r = F.n, F.r
    f = p->M*vec(F.V*reshape(p, r, r)*F.V')
    fc = q->vec(F.V'*reshape(M'*q, n, n)*F.V)
    return LinearMap(f, fc, size(M, 1), r*r)
end

Base.:(*)(λ::Real, F::TraceBallFace{Int64, Matrix{Float64}}) = λ*vec(F)
Base.length(F::TraceBallFace{Int64, Matrix{Float64}}) = F.n*F.n
Base.size(F::TraceBallFace{Int64, Matrix{Float64}}) = F.n, F.n
rank(F::TraceBallFace{Int64, Matrix{Float64}}) = F.r^2
vec(F::TraceBallFace{Int64, Matrix{Float64}}) = I(F.n*F.n)*F
face_name(F::TraceBallFace{Int64, Matrix{Float64}}) = "face of trace-norm ball"
face_parameters(F::TraceBallFace{Int64, Matrix{Float64}}) = "n = $(F.n), rank = $(rank(F))"
