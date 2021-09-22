########################################################################
# Sum of atoms.
########################################################################

"""
    SumAtom(n, atoms::Vector{AbstractAtom})

Create an atom as the sum of individual atoms in a vector.
"""
struct SumAtom{As<:Tuple{Vararg{AbstractAtom}}} <: AbstractAtom
    n::Int64
    atoms::As
    function SumAtom(atoms::As) where As
        n = checklength(atoms, "all atoms must be the same length")
        return new{As}(n, atoms)
    end
end

Base.length(a::SumAtom) = a.n
Base.vec(a::SumAtom) = mapreduce(vec, +, a.atoms)
Base.:(*)(M::AbstractLinearOp, a::SumAtom) = mapreduce(a->M*a, +, a.atoms)

# Add a unit test if resurrected.
# scale(a::SumAtom, α::Vector) =
#     SumAtom([α*a for (a,α) = zip(a,α)])

########################################################################
# Sum of atomic sets.
########################################################################

"""
    SumAtomicSet(A::Vector{AbstractAtomicSet})

Create an atomic set from the Minkowski sum of atomic sets.
"""
struct SumAtomicSet{As<:Tuple{Vararg{AbstractAtomicSet}}} <: AbstractAtomicSet
    n::Int64
    sets::As
    function SumAtomicSet(sets::As) where As
        n = checklength(sets, "all atomic sets must be the same length")
        return new{As}(n, sets)
    end
end


"""
    sum(A, B)
    +(A, B)

Create the sum of the atomic sets A and B.
"""
function Base.:(+)(A::AbstractAtomicSet, B::AbstractAtomicSet)
    return SumAtomicSet(tuple(A, B))
end
function Base.:(+)(A::SumAtomicSet, B::AbstractAtomicSet)
    return SumAtomicSet(tuple(A.sets..., B))
end
function Base.:(+)(A::AbstractAtomicSet, B::SumAtomicSet)
    return SumAtomicSet(tuple(A, B.sets...))
end
function Base.:(+)(A::SumAtomicSet, B::SumAtomicSet)
    return SumAtomicSet(tuple(A.sets..., B.sets...))
end


"""
    support(A::SumAtomicSet, z)

Support function of a sum of atomic sets.
"""
function support(A::SumAtomicSet, z::Vector)
    f = Ai->support(Ai, z)
    return mapreduce(f, +, A.sets)
end


"""
    expose(A::SumAtomicSet, z; kwargs...)

Expose atoms in the sum of atomic sets. For two atomic sets
A and B, and a vector z,

    expose((A+B),z) = expose(A,z) + expose(B,z)

"""
function expose(A::SumAtomicSet, z; kwargs...)
    f = Ai->expose(Ai, z; kwargs...)
    return SumAtom(map(f, A.sets))
end

Base.length(A::SumAtomicSet) = A.n
atom_name(A::SumAtomicSet) = "Sum of atomic sets"
atom_description(A::SumAtomicSet) = "A₁ + A₂ + ⋯"
atom_parameters(A::SumAtomicSet) = "$(length(A.sets)) sets; n = $(A.n)"

########################################################################
# Face of sum of atomic sets.
########################################################################

struct SumFace{As<:Tuple{Vararg{AbstractFace}}} <: AbstractFace
    n::Int64
    faces::As
    function SumFace(faces::As) where As
        n = checklength(faces, "all faces must be the same length")
        return new{As}(n, faces)
    end
end

"""
    face(A1 + ... + Ak, z)

Return collection of faces of the atomic sets `Ai`'s exposed by the vector `z`.
"""
function face(A::SumAtomicSet, z::Vector; rTol=1e-1)
    f = Ai -> face(Ai, z, rTol = rTol)
    faces = map(f, A.sets)
    return SumFace(faces)
end
"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::SumFace, c::Vector) = vec(F)*c
Base.:(*)(M::AbstractLinearOp, F::SumFace) = mapreduce(Fi->M*Fi, hcat, F.faces)
Base.:(*)(λ::Real, F::SumFace) = mapreduce(Fi->λ*Fi, hcat, F.faces)
Base.length(F::SumFace) = F.n
rank(F::SumFace) = map(rank, F.faces)
vec(F::SumFace) = mapreduce(vec, hcat, F.faces)
face_name(F::SumFace) = "Face of the sum of atomic sets"
face_parameters(F::SumFace) = "rank = $(rank(F)); n = $(length(F))"
