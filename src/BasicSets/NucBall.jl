########################################################################
# Atoms of the nuclear-norm ball.
########################################################################
"""
    NucBallAtom(m, n, u, v)

Atom in the NucBall of dimension m×n.
"""
struct NucBallAtom{T1<:Int64, T2<:Vector{Float64}} <: AbstractAtom
    m::T1
    n::T1
    u::T2
    v::T2
    function NucBallAtom(u::Vector{Float64}, v::Vector{Float64})
        m = length(u)
        n = length(v)
        new{Int64, Vector{Float64}}(m, n, u, v)
    end
end

"""
    *(M::AbstractLinearOp, a::NucBallAtom) -> Y

Multiply a nuclear-norm atom by a linear map, and return the matrix `Y`.
"""
Base.:(*)(M::LinearOp, a::NucBallAtom) = M*vec(a.u*a.v')
Base.vec(a::NucBallAtom) = vec(a.u*a.v')
Base.length(a::NucBallAtom) = a.m * a.n
Base.size(a::NucBallAtom) = a.m, a.n

########################################################################
# Nuclear-norm ball.
########################################################################
"""
    NucBall(m, n, maxrank=min(m,n))

Atomic set defined by nuclear-norm ball for the space of `m`-by-`n` matrices.
The atomic set takes two optional parameters:

`maxrank` is the maximum dimension of the face exposed by a vector.
"""
struct NucBall{T<:Int64} <: AbstractAtomicSet
    m::T
    n::T
    maxrank::T
    function NucBall(m::Int64, n::Int64, maxrank::Int64)
        m ≥ 1 || throw(DomainError(m,"m must be a positive integer"))
        n ≥ 1 || throw(DomainError(n,"n must be a positive integer"))
        min(m,n) ≥ maxrank ≥ 0 || throw(DomainError(maxrank,"maxrank must be ≥ 0"))
        new{Int64}(m, n, maxrank)
    end
end
NucBall(m::Int64, n::Int64; maxrank=min(m, n)) = NucBall(m, n, maxrank)


"""
    gauge(A::NucBall, x::Matrix)

Gives the nuclear-norm of `x`.
"""
gauge(::NucBall, x::Matrix{Float64}) = sum(svdvals(x))
gauge(A::NucBall, x::Vector{Float64}) = gauge(A, reshape(x, A.m, A.n))


"""
    support(A::NucBall, z::AbstractMatrix)

Gives the largest singular value of `z`.
"""
function support(A::NucBall, z::AbstractMatrix{Float64})
    r = A.maxrank
    if r == 0
        s = [0]
    else
        ~, s, ~ = svds(z, nsv=1)[1]
    end
    return s[1]
end

support(A::NucBall, z::Vector{Float64}) = support(A, reshape(z, A.m, A.n))


"""
    expose(A::NucBall, z::AbstractMatrix)

Gives the top singular vectors of `z`.
"""
function expose(::NucBall, z::AbstractMatrix{Float64})
    u, ~, v = svds(z, nsv=1)[1]
    return NucBallAtom(vec(u), vec(v'))
end

expose(A::NucBall, z::Vector{Float64}) = expose(A, reshape(z, A.m, A.n))

Base.length(A::NucBall) = A.m*A.n
Base.size(A::NucBall) = A.m, A.n
rank(A::NucBall) = A.maxrank
atom_name(A::NucBall) = "nuclear-norm ball"
atom_description(A::NucBall) = "{ uv^T | u ∈ ℝᵐ, v ∈ ℝⁿ, ||u||₂ = ||v||₂ ≤ 1 }"
atom_parameters(A::NucBall) = "m = $(A.m), n = $(A.n), maxrank = $(A.maxrank)"

########################################################################
# Face of the nuclear-norm ball.
########################################################################

struct NucBallFace{T1<:Int64, T2<:Matrix{Float64}} <: AbstractFace
    m::T1
    n::T1
    r::T1
    U::T2
    V::T2
    function NucBallFace(U::Matrix{Float64}, V::Matrix{Float64})
        m = size(U, 1)
        n = size(V, 1)
        r = size(U, 2)
        new{Int64, Matrix{Float64}}(m, n, r, U, V)
    end
end

"""
    face(A, z) -> F

Return a face `F` of the atomic set `A` exposed by the vector `z`. The dimension of
the exposed face is limited by `k=maxrank(A)`. If the dimension is being limited
by `k`, then the routine returns a set of rank `k` atoms that define the subset
of the exposed face.
"""
function face(A::NucBall, z::AbstractMatrix{Float64})
    r = A.maxrank
    if r == 0   
        U = zeros(A.m, 1)
        V = zeros(A.n, 1)
    elseif r < min(A.m, A.n) - 1
        # Arpack won't accept nsv too large
        U, _, V = svds(z, nsv=r+1)[1]
        V = convert(typeof(U), V)
    else
        U, _, V = svd(z)
        U = U[:, 1:r]
        V = V[:, 1:r]
    end
    return NucBallFace(U, V)
end

face(A::NucBall, z::Vector{Float64}) = face(A, reshape(z, A.m, A.n))

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::NucBallFace, S::AbstractMatrix{Float64}) = vec(F.U*S*F.V')
Base.:(*)(F::NucBallFace, c::Vector{Float64}) = F*reshape(c, F.r, F.r)

"""
    *(M::LinearOp, F::NucBallFace)

Return a LinearMap `L` whose forward and adjoing operators
are defined as

    L*p = M*vec(U*matrix(p)*V')
    L'*q = vec(U'*matrix(M'*q)*V)

where the `n`-by-`k` matrices `U` and `V` are the left and right
singular vectors that define the face `F` of the nuclear-norm ball.
"""
function Base.:(*)(M::LinearOp, F::NucBallFace) 
    m, n, r = F.m, F.n, F.r
    f = p->M*vec(F.U*reshape(p, r, r)*F.V')
    fc = q->vec(F.U'*reshape(M'*q, m, n)*F.V)
    return LinearMap(f, fc, size(M, 1), r*r)
end

Base.:(*)(λ::Real, F::NucBallFace) = λ*vec(F)
Base.length(F::NucBallFace) = F.m*F.n
Base.size(F::NucBallFace) = F.m, F.n
rank(F::NucBallFace) = F.r
vec(F::NucBallFace) = I(F.m*F.n)*F
face_name(F::NucBallFace) = "face of nuclear-norm ball"
face_parameters(F::NucBallFace) = "m = $(F.m), n = $(F.n), rank = $(rank(F))"
