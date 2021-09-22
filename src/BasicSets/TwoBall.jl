########################################################################
# Atoms of the 2-norm ball.
########################################################################

"""
    TwoBallAtom(n, z)

Atom in the TwoBall of dimension n.
"""
struct TwoBallAtom <: AbstractAtom
    n::Int64
    z::Vector{Float64}
    function TwoBallAtom(z)
        n = length(z)
        return new(n, z)
    end
end

"""
Multiply a two-norm atom by a linear map.
"""
Base.:(*)(M::AbstractLinearOp, a::TwoBallAtom) = M*a.z
Base.vec(a::TwoBallAtom) = a.z
Base.length(a::TwoBallAtom) = a.n

########################################################################
# Atomic set for the 1-norm.
########################################################################

"""
    TwoBall(n, maxrank=n)

Atomic set defined by 2-norm ball in `n` variables.
The atomic set takes two optional parameters:

`maxrank` is the maximum dimension of the face exposed by a vector.
"""
struct TwoBall <: AbstractAtomicSet
    n::Int64
    maxrank::Int64
    function TwoBall(n, maxrank)
        n ≥ 1 || throw(DomainError(n,"n must be a positive integer"))
        n ≥ maxrank ≥ 1 || throw(DomainError(maxrank,"maxrank must be ≥ 1"))
        return new(n, maxrank)
    end
end
TwoBall(n; maxrank=n) = TwoBall(n, maxrank)

"""
    gauge(A::TwoBall, x::Vector)

Gives the 2-norm of `x`.
"""
gauge(A::TwoBall, x::Vector) = norm(x)
gauge(A::TwoBall, x::Matrix) = gauge(A, vec(x))

"""
    support(A::TwoBall, z::Vector)

Gives the 2-norm of `z`.
"""
support(A::TwoBall, z::Vector) = norm(z)

"""
    expose(A::TwoBall, z::Vector)

A non-overwriting version of [`expose!`](@ref).
"""
expose(A::TwoBall, z; kwargs...) = expose!(A::TwoBall, copy(z); kwargs...)

"""
    expose!(A::TwoBall, z::Vector; tol=1e-12)

Obtain an atom in the face exposed by the vector `z`.
The vector `z` is overwritten. If `norm(z)<tol`, then
`z` is returned untouched.
"""
function expose!(A::TwoBall, z::Vector; tol=1e-12)
    znorm = norm(z)
    if znorm < tol
        return TwoBallAtom(zero(z))
    end
    return TwoBallAtom(z ./= znorm)
end

Base.length(A::TwoBall) = A.n
rank(A::TwoBall) = A.maxrank
atom_name(A::TwoBall) = "2-norm ball"
atom_description(A::TwoBall) = "{ x ∈ ℝⁿ | ||x||_2 ≤ 1 }"
atom_parameters(A::TwoBall) = "n = $(length(A)); maxrank = $(A.maxrank)"


########################################################################
# Face of the 1-norm ball.
########################################################################

struct TwoBallFace <: AbstractFace
    n::Int64
    k::Int64
    S::LinearMap{Float64}
end

TwoBallFace(S::LinearMap) = TwoBallFace(size(S)...,S)

"""
    face(A, z)

Return a face of the atomic set `A` exposed by the vector `z`. The dimension of
the exposed face is limited by `k=maxrank(A)`. If the dimension is being limited
by `k`, then the routine returns a set of rank `k` atoms that define the subset
of the exposed face.
"""
function face(A::TwoBall, z::Vector; rTol=1e-12)
    @assert length(A) == length(z)
    a = expose(A, z, tol=rTol)
    S = LinearMap(reshape(a.z, length(A), 1))
    return TwoBallFace(S)
end

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::TwoBallFace, c::Vector) = F.S*c
Base.:(*)(M::AbstractLinearOp, F::TwoBallFace) = M*F.S
Base.:(*)(λ::Real, F::TwoBallFace) = λ*F.S
Base.length(F::TwoBallFace) = F.n
rank(F::TwoBallFace) = F.k
vec(F::TwoBallFace) = F.S
face_name(F::TwoBallFace) = "2-norm ball"
face_parameters(F::TwoBallFace) = "rank = $(rank(F)); n = $(length(F))"
