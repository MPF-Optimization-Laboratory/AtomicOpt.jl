########################################################################
# Atoms of the 1-norm ball.
########################################################################

"""
    OneBallAtom(n, z)

Atom in the OneBall of dimension n.
"""
struct OneBallAtom <: AbstractAtom
    n::Int64
    z::Vector{Float64}
    function OneBallAtom(z)
        n = length(z)
        return new(n, z)
    end
end

"""
Multiply a one-norm atom by a linear map.
"""
Base.:(*)(M::AbstractLinearOp, a::OneBallAtom) = M*a.z
Base.vec(a::OneBallAtom) = a.z
Base.length(a::OneBallAtom) = a.n

########################################################################
# Atomic set for the 1-norm.
########################################################################

"""
    OneBall(n, maxrank=n)

Atomic set defined by 1-norm ball in `n` variables.
The atomic set takes two optional parameters:

`maxrank` is the maximum dimension of the face exposed by a vector.
"""
struct OneBall <: AbstractAtomicSet
    n::Int64
    maxrank::Int64
    function OneBall(n, maxrank)
        n ≥ 1 || throw(DomainError(n,"n must be a positive integer"))
        n ≥ maxrank ≥ 1 || throw(DomainError(maxrank,"maxrank must be ≥ 1"))
        return new(n, maxrank)
    end
end
OneBall(n; maxrank=n) = OneBall(n, maxrank)

"""
    gauge(A::OneBall, x::Vector)

Gives the 1-norm of `x`.
"""
gauge(A::OneBall, x::Vector) = norm(x,1)
gauge(A::OneBall, x::Matrix) = gauge(A, vec(x))

"""
    support(A::OneBall, z::Vector)

Gives the inf-norm of `z`.
"""
support(A::OneBall, z::Vector) = norm(z,Inf)

"""
    expose(A::OneBall, z::Vector)

A non-overwriting version of [`expose!`](@ref).
"""
expose(A::OneBall, z; kwargs...) = expose!(A::OneBall, copy(z); kwargs...)

"""
    expose!(A::OneBall, z::Vector; tol=1e-12)

Obtain an atom in the face exposed by the vector `z`.
The vector `z` is overwritten. If `norm(z,Inf)<tol`, then
`z` is returned untouched.
"""
function expose!(A::OneBall, z::Vector; tol=1e-1)
    zmax = norm(z,Inf)
    if zmax < 1e-12
        return OneBallAtom(zero(z))
    end
    nnz = 0
    for i in eachindex(z)
        val = z[i]
        if abs(zmax - abs(val)) < tol*zmax
            z[i] = sign(val)
            nnz += 1
        else
            z[i] = zero(eltype(z))
        end
    end
    return OneBallAtom(z ./= nnz)
end

Base.length(A::OneBall) = A.n
rank(A::OneBall) = A.maxrank
atom_name(A::OneBall) = "1-norm ball"
atom_description(A::OneBall) = "{ x ∈ ℝⁿ | ||x||₁ ≤ 1 }"
atom_parameters(A::OneBall) = "n = $(length(A)); maxrank = $(A.maxrank)"


########################################################################
# Face of the 1-norm ball.
########################################################################

struct OneBallFace <: AbstractFace
    n::Int64
    k::Int64
    S::LinearMap{Float64}
end

OneBallFace(S::LinearMap) = OneBallFace(size(S)...,S)

"""
    face(A, z)

Return a face of the atomic set `A` exposed by the vector `z`. The dimension of
the exposed face is limited by `k=maxrank(A)`. If the dimension is being limited
by `k`, then the routine returns a set of rank `k` atoms that define the subset
of the exposed face.
"""
function face(A::OneBall, z::Vector; rTol=1e-1)
    @assert length(A) == length(z)
    if rank(A) < length(A)
        idx = partialsortperm(abs.(z), 1:rank(A), rev=true)
    else 
        t = support(A, z)
        a = expose(A, z, tol=rTol)
        idx = findall(!iszero, vec(a))
    end
    k = length(idx)
    val = [sign(z[i]) for i in idx]
    n = length(A)
    S = sparse(idx, collect(1:k), val, n, k)
    S = LinearMap(S)
    return OneBallFace(S)
end

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::OneBallFace, c::Vector) = F.S*c
Base.:(*)(M::AbstractLinearOp, F::OneBallFace) = M*F.S
Base.:(*)(λ::Real, F::OneBallFace) = λ*F.S
Base.length(F::OneBallFace) = F.n
rank(F::OneBallFace) = F.k
vec(F::OneBallFace) = F.S
face_name(F::OneBallFace) = "1-norm ball"
face_parameters(F::OneBallFace) = "rank = $(rank(F)); n = $(length(F))"
