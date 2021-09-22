########################################################################
# Mapped atoms.
########################################################################

"Mapped atom."
struct MappedAtom <: AbstractAtom
    n::Int64
    z::Vector{Float64}
    function MappedAtom(z)
        n = length(z)
        new(n, z)
    end
end

Base.length(a::MappedAtom) = a.n
Base.vec(a::MappedAtom) = a.z
Base.:(*)(M::AbstractLinearOp, a::MappedAtom) = M*a.z

########################################################################
# Mapped atomic set.
########################################################################

"""
    MappedAtomicSet(n::Integer, A::AbstractAtomicSet, M::LinearMap)

Linear transformation another atomic set, including scaling.
"""
struct MappedAtomicSet{matT<:AbstractLinearOp, atomT<:AbstractAtomicSet} <: AbstractAtomicSet
    n::Int64
    child::atomT
    M::matT
    function MappedAtomicSet(child::atomT, M::matT) where {matT,atomT}
        n = size(M, 2)
        n == length(child) || error("size(M) incompatible with atomic set A")
        return new{matT,atomT}(n, child, M)
    end
end


"""
    *(M, A::AbstractAtomicSet)

Apply the linear operator `M` on the atomic set `A`. Results in
another atomic set (MA) with the defining property

    MA = { Ma | a ∈ A }.
"""
function Base.:(*)(M::matT,
                   A::atomT) where {matT<:AbstractLinearOp,atomT<:AbstractAtomicSet}
    return MappedAtomicSet(A, M)
end


"""
    gauge(A::MappedAtomicSet, x::Vector)

Gauge value of a vector with respect to a scaled atomic set.
"""
gauge(A::MappedAtomicSet, x::Vector) = gauge(A.child, A.M\x)


"""
    support(A::MappedAtomicSet, z::Vector)

Support value of a gauge with respect to a scaled atomic set.
"""
support(A::MappedAtomicSet, z::Vector) = support(A.child, A.M'*z)


"""
    expose(A::MappedAtomicSet, z::Vector)

Obtain an atom in the face of the face of the scaled atomic set exposed by
the vector `z`.
"""
function expose(A::MappedAtomicSet, z; kwargs... )
    Ma = A.M*expose(A.child, A.M'*z; kwargs...)
    return MappedAtom(Ma)
end

Base.length(A::MappedAtomicSet) = A.n
atom_name(A::MappedAtomicSet) = "Map of $(atom_name(A.child))"
atom_description(A::MappedAtomicSet) = "M ⋅ $(atom_description(A.child))"
atom_parameters(A::MappedAtomicSet) = "n = $(length(A))"

########################################################################
# Face of the Mapped Atomic Set
########################################################################

struct MappedFace <: AbstractFace
    n::Int64
    k::Int64
    S::LinearMap{Float64}
end

MappedFace(S::LinearMap) = MappedFace(size(S)...,S)

"""
    face(A, z)

Return a face of the atomic set `A` exposed by the vector `z`. 
"""
function face(A::MappedAtomicSet, z::Vector; rTol=1e-1)
    F = face(A.child, A.M'*z, rTol = rTol)
    S = A.M*F
    return MappedFace(S)
end

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::MappedFace, c::Vector) = F.S*c
Base.:(*)(M::AbstractLinearOp, F::MappedFace) = M*F.S
Base.:(*)(λ::Real, F::MappedFace) = λ*F.S
Base.length(F::MappedFace) = F.n
rank(F::MappedFace) = F.k
vec(F::MappedFace) = F.S
face_name(F::MappedFace) = "Mapped Atomic Set"
face_parameters(F::MappedFace) = "rank = $(rank(F)); n = $(length(F))"
