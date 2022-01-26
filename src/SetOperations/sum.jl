########################################################################
# Sum of atoms.
########################################################################

"""
    SumAtom(n, atoms::Vector{AbstractAtom})

Create an atom as the sum of individual atoms in a vector.
"""
mutable struct SumAtom{T1<:Int64, T2<:Tuple{Vararg{AbstractAtom}}} <: AbstractAtom
    n::T1
    atoms::T2
    function SumAtom(atoms::Tuple{Vararg{AbstractAtom}}) 
        n = checklength(atoms, "all atoms must be the same length")
        new{Int64, Tuple{Vararg{AbstractAtom}}}(n, atoms)
    end
end

Base.length(a::SumAtom) = a.n
Base.vec(a::SumAtom) = mapreduce(vec, hcat, a.atoms)
Base.:(*)(M::AbstractLinearOp, a::SumAtom) = mapreduce(a->M*a, hcat, a.atoms)
LinearAlgebra.mul!(Ma::AbstractMatrix{Float64}, M::LinearOp, a::SumAtom) = pmap(i->mul!(view(Ma,:,i), M, a.atoms[i]), collect(1:length(a.atoms)))


########################################################################
# Sum of atomic sets.
########################################################################

"""
    SumAtomicSet(A::Vector{AbstractAtomicSet})

Create an atomic set from the Minkowski sum of atomic sets.
"""
struct SumAtomicSet{T1<:Int64, T2<:Tuple{Vararg{AbstractAtomicSet}}} <: AbstractAtomicSet
    n::T1
    sets::T2
    function SumAtomicSet(sets::Tuple{Vararg{AbstractAtomicSet}}) 
        n = checklength(sets, "all atomic sets must be the same length")
        new{Int64, Tuple{Vararg{AbstractAtomicSet}}}(n, sets)
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
function support(A::SumAtomicSet, z::Vector{Float64})
    return mapreduce(Ai->support(Ai, z), +, A.sets)
end


"""
    expose(A::SumAtomicSet, z)

Expose atoms in the sum of atomic sets. For two atomic sets
A and B, and a vector z,

    expose((A+B),z) = expose(A,z) + expose(B,z)

"""
function expose!(A::SumAtomicSet, z::Vector{Float64}, a::SumAtom)
    k = length(A.sets)
    pmap(i->expose!(A.sets[i], z, a.atoms[i]), collect(1:k))
    return nothing
end

function expose(A::SumAtomicSet, z::Vector{Float64})
    atoms = pmap(Ai->expose(Ai, z), A.sets)
    return SumAtom(atoms)
end 

Base.length(A::SumAtomicSet) = A.n
atom_name(A::SumAtomicSet) = "Sum of atomic sets"
atom_description(A::SumAtomicSet) = "A₁ + A₂ + ⋯"
atom_parameters(A::SumAtomicSet) = "$(length(A.sets)) sets; n = $(A.n)"

########################################################################
# Face of sum of atomic sets.
########################################################################
mutable struct SumFace{T1<:Int64, T2<:Tuple{Vararg{AbstractFace}}} <: AbstractFace
    n::T1
    faces::T2
    function SumFace(faces::Tuple{Vararg{AbstractFace}})
        n = checklength(faces, "all faces must be the same length")
        return new{Int64, Tuple{Vararg{AbstractFace}}}(n, faces)
    end
end

"""
    face(A1 + ... + Ak, z)

Return collection of faces of the atomic sets `Ai`'s exposed by the vector `z`.
"""
function face!(A::SumAtomicSet, z::Vector{Float64}, F::SumFace)
    k = length(A.sets)
    pmap(i->face!(A.sets[i], z, F.faces[i]), collect(1:k))
    return nothing
end

function face(A::SumAtomicSet, z::Vector{Float64})
    faces = pmap(Ai->face(Ai, z), A.sets)
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
