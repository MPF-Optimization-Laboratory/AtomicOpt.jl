########################################################################
# Cross product of atoms.
########################################################################

"""
    CrossProductAtom(n, atoms::Vector{AbstractAtom})

Create an atom as the cross product of individual atoms.
"""
struct CrossProductAtom{As<:Tuple{Vararg{AbstractAtom}}} <: AbstractAtom
    n::Int64
    atoms::As
    function CrossProductAtom(atoms::As) where As
        n = checklength(atoms, "all atoms must be the same length")
        return new{As}(n, atoms)
    end
end

Base.length(a::CrossProductAtom) = a.n
Base.vec(a::CrossProductAtom) = mapreduce(vec, hcat, a.atoms)
Base.:(*)(M::AbstractLinearOp, a::CrossProductAtom) =
    mapreduce(a->M*a, hcat, a.atoms)
Base.getindex(a::CrossProductAtom, i::Integer) = a.atoms[i]

########################################################################
# Cross product of atomic sets.
########################################################################

"""
    CrossProductSet(A::Vector{AbstractAtomicSet})

Create an atomic set from the cross product of atomic sets.
"""
struct CrossProductSet{As<:Tuple{Vararg{AbstractAtomicSet}}} <: AbstractAtomicSet
    n::Int64
    sets::As
    function CrossProductSet(sets::As) where As
        n = checklength(sets,"all atomic sets must be the same length")
        return new{As}(n, sets)
    end
end


"""
    cross(A, B)
    ×(A, B)

Create the cross product of the atomic sets A and B.
"""
function cross(A::AbstractAtomicSet, B::AbstractAtomicSet)
    return CrossProductSet(tuple(A,B))
end
function cross(A::CrossProductSet, B::AbstractAtomicSet)
    return CrossProductSet(tuple(A.sets..., B))
end
function cross(A::AbstractAtomicSet, B::CrossProductSet)
    return CrossProductSet(tuple(A, B.sets...))
end
function cross(A::CrossProductSet, B::CrossProductSet)
    return CrossProductSet(tuple(A.sets..., B.sets...))
end


"""
    support(A::CrossProductSet, z)

Support function of the cross product of atomic sets.
"""
function support(A::CrossProductSet, z)
    f = Ai->support(Ai, z)
    return mapreduce(f, +, A.sets)
end

"""
    expose(A::CrossProductSet, z; kwargs...)

Expose atoms in the cartesian product of atomic sets.
For two atomic sets A and B, and a vector z,

    expose(A×B,z) = (expose(A,z), expose(B,z))
"""
function expose(A::CrossProductSet, z; kwargs...)
    f = Ai->expose(Ai, z; kwargs...)
    return CrossProductAtom(map(f, A.sets))
end

Base.length(A::CrossProductSet) = A.n
atom_name(A::CrossProductSet) = "Cross product of atomic sets"
atom_description(A::CrossProductSet) = "A₁ × A₂ × ⋯"
atom_parameters(A::CrossProductSet) = "$(length(A.sets)) sets; n = $(A.n)"

Base.getindex(A::CrossProductSet, i::Integer) = A.sets[i]

########################################################################
# Face of product of atomic sets.
########################################################################

struct CrossProductFace{As<:Tuple{Vararg{AbstractFace}}} <: AbstractFace
    n::Int64
    faces::As
    function CrossProductFace(faces::As) where As
        n = checklength(faces, "all faces must be the same length")
        return new{As}(n, faces)
    end
end

"""
    face(A1 × ... × Ak, z)

Return collection of faces of the atomic sets `Ai`'s exposed by the vector `z`. 
"""
function face(A::CrossProductSet, z::Vector; rTol=1e-1)
    f = Ai -> face(Ai, z, rTol = rTol)
    faces = map(f, A.sets)
    return CrossProductFace(faces)
end

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
function Base.:(*)(F::CrossProductFace, c::Vector)
    x = []
    start = 1
    for (Fi,ri) in zip(F.faces, rank(F))
        push!(x, Fi*c[start:start+ri-1])
        start += ri
    end
    return x
end

# function TEMP(F::CrossProductFace, c::Vector)
#     r = cumsum(rank(F))
#     rng = map( t->(:)(t[1]+1,t[2]), zip((0,r[1:end-1]...), r))
#     cF = ( c->view(c,rng) for rng in rng)
#     return map(*, F,)
# end

# """
# Create a tuple from the cumulative sum of the input tuple, i.e.,input

#     accumulate((3,2,1,4)) == (3,5,6,10)
# """
# cumsum(itr::Tuple) = foldl((x,y)->(x...,last(x)+y), itr)

Base.:(*)(M::AbstractLinearOp, F::CrossProductFace) = mapreduce(Fi->M*Fi, hcat, F.faces)
Base.:(*)(λ::Real, F::CrossProductFace) = mapreduce(Fi->λ*Fi, hcat, F.faces)
Base.length(F::CrossProductFace) = F.n
rank(F::CrossProductFace) = map(rank, F.faces)
vec(F::CrossProductFace) = mapreduce(vec, hcat, F.faces)
face_name(F::CrossProductFace) = "Face of the product of atomic sets"
face_parameters(F::CrossProductFace) = "rank = $(rank(F)); n = $(length(F))"
Base.getindex(A::CrossProductFace, i::Integer) = A.faces[i]
