########################################################################
# Atoms of the nuclear-norm ball.
########################################################################
"""
    NucBallAtom(n, m, u, v)

Atom in the NucBall of dimension n×m.
"""
struct NucBallAtom <: AbstractAtom
    n::Int64
    m::Int64
    u::Vector{Float64}
    v::Vector{Float64}
    function NucBallAtom(u, v)
        n = length(u)
        m = length(v)
        new(n, m, u, v)
    end
end

"""
    *(M::AbstractLinearOp, a::NucBallAtom) -> Y

Multiply a nuclear-norm atom by a linear map, and return the matrix `Y`.
"""
Base.:(*)(M::AbstractLinearOp, a::NucBallAtom) = M*vec(a.u*a.v')
Base.vec(a::NucBallAtom) = vec(a.u*a.v')
Base.length(a::NucBallAtom) = a.n * a.m
Base.size(a::NucBallAtom) = a.n, a.m

########################################################################
# Nuclear-norm ball.
########################################################################
"""
    NucBall(n, m, maxrank=n)

Atomic set defined by nuclear-norm ball for the space of `m`-by-`n` matrices.
The atomic set takes two optional parameters:

`maxrank` is the maximum dimension of the face exposed by a vector.
"""
struct NucBall <: AbstractAtomicSet
    n::Int64
    m::Int64
    maxrank::Int64
    function NucBall(n, m, maxrank)
        n ≥ 1 || throw(DomainError(n,"n must be a positive integer"))
        m ≥ 1 || throw(DomainError(m,"m must be a positive integer"))
        min(m,n) ≥ maxrank ≥ 0 || throw(DomainError(maxrank,"maxrank must be ≥ 0"))
        new(n, m, maxrank)
    end
end
NucBall(n, m; maxrank=min(n, m)) = NucBall(n, m, maxrank)


"""
    gauge(A::NucBall, x::Matrix)

Gives the nuclear-norm of `x`.
"""
gauge(A::NucBall, x::Matrix) = sum(svdvals(x))
gauge(A::NucBall, x::Vector) = gauge(A, reshape(x, A.n, A.m))


"""
    support(A::NucBall, z::Matirx)

Gives the largest singular value of `z`.
"""
function support(A::NucBall, z::Union{Matrix, SparseMatrixCSC})
    r = A.maxrank
    if r == 0
        s = [0]
    else
        ~, s, ~ = svds(z, nsv=1)[1]
    end
    return s[1]
end

support(A::NucBall, z::Vector) = support(A, reshape(z, A.n, A.m))


"""
    expose(A::NucBall, z::Matrix)

Gives the top singular vectors of `z`.
"""
function expose(::NucBall, z::Union{Matrix, SparseMatrixCSC})
    u, ~, v = svds(z, nsv=1)[1]
    return NucBallAtom(vec(u), vec(v))
end

expose(A::NucBall, z::Vector) = expose(A, reshape(z, A.n, A.m))

Base.length(A::NucBall) = A.n*A.m
rank(A::NucBall) = A.maxrank
atom_name(A::NucBall) = "nuclear-norm ball"
atom_description(A::NucBall) = "{ x ∈ ℝⁿ | ||x||₁ ≤ 1 }"
atom_parameters(A::NucBall) = "n×m = $(length(A)); maxrank = $(A.maxrank)"

########################################################################
# Face of the nuclear-norm ball.
########################################################################

struct NucBallFace <: AbstractFace
    n::Int64
    m::Int64
    k::Int64
    U::Matrix{Float64}
    V::Matrix{Float64}
    function NucBallFace(U, V)
        n = size(U, 1)
        m = size(V, 1)
        k = size(U, 2)
        new(n, m, k, U, V)
    end
end

"""
    face(A, z) -> F

Return a face `F` of the atomic set `A` exposed by the vector `z`. The dimension of
the exposed face is limited by `k=maxrank(A)`. If the dimension is being limited
by `k`, then the routine returns a set of rank `k` atoms that define the subset
of the exposed face.
"""
function face(A::NucBall, z::Union{Matrix, SparseMatrixCSC}; rTol=0.1)
    r = A.maxrank
    if r == 0   
        U = zeros(A.n, 1)
        V = zeros(A.m, 1)
    elseif r < min(A.m, A.n) - 1
        # Arpack won't accept nsv too large
        U, _, V = svds(z, nsv=r+1)[1]
    else
        U, _, V = svd(z)
        U = U[:, 1:r]
        V = V[:, 1:r]
    end
    return NucBallFace(U, V)
end

face(A::NucBall, z::Vector; kwargs...) = face(A, reshape(z, A.n, A.m); kwargs...)

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::NucBallFace, S::Union{Matrix, SparseMatrixCSC}) = vec(F.U*S*F.V')
Base.:(*)(F::NucBallFace, c::Vector) = F*reshape(c, F.k, F.k)

"""
    *(M::AbstractLinearOp, F::NucBallFace)

Return a LinearMap `L` whose forward and adjoing operators
are defined as

    L*p = M*vec(U*matrix(p)*V')
    L'*q = vec(U'*matrix(M'*q)*V)

where the `n`-by-`k` matrices `U` and `V` are the left and right
singular vectors that define the face `F` of the nuclear-norm ball.
"""
function Base.:(*)(M::AbstractLinearOp, F::NucBallFace) 
    n, m, k = F.n, F.m, F.k
    f = p->M*vec(F.U*reshape(p, k, k)*F.V')
    fc = q->vec(F.U'*reshape(M'*q, n, m)*F.V)
    return LinearMap(f, fc, size(M, 1), k*k)
end
Base.:(*)(λ::Real, F::NucBallFace) = λ*vec(F)
Base.length(F::NucBallFace) = F.n*F.m
rank(F::NucBallFace) = F.k
vec(F::NucBallFace) = I(F.m*F.n)*F
face_name(F::NucBallFace) = "nuclear-norm ball"
face_parameters(F::NucBallFace) = "rank = $(rank(F)); n = $(length(F))"
