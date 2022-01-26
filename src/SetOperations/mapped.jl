########################################################################
# Mapped atoms.
########################################################################

"Mapped atom."
struct MappedAtom{T1<:Int64, T2<:AbstractLinearOp, T3<:AbstractAtom} <: AbstractAtom
    n::T1
    map::T2
    atom::T3
    function MappedAtom(map::AbstractLinearOp, atom::AbstractAtom)
        size(map, 2) == length(atom) || error("size(M) incompatible with atom")
        n = size(map, 1)
        new{Int64, AbstractLinearOp, AbstractAtom}(n, map, atom)
    end
end

Base.length(a::MappedAtom) = a.n
Base.vec(a::MappedAtom) = a.map * a.atom
Base.:(*)(M::AbstractLinearOp, a::MappedAtom) = M*vec(a)
LinearAlgebra.mul!(Ma::AbstractVector{Float64}, M::LinearOp, a::MappedAtom) = mul!(Ma, M, vec(a))

########################################################################
# Mapped atomic set.
########################################################################

"""
    MappedAtomicSet(n::Integer, A::AbstractAtomicSet, M::LinearMap)

Linear transformation another atomic set, including scaling.
"""
struct MappedAtomicSet{T1<:Int64, T2<:AbstractLinearOp, T3<:AbstractAtomicSet} <: AbstractAtomicSet
    n::T1
    map::T2
    set::T3
    function MappedAtomicSet(map::AbstractLinearOp, set::AbstractAtomicSet) 
        size(map, 2) == length(set) || error("size(M) incompatible with atomic set")
        n = size(map, 1)
        return new{Int64, AbstractLinearOp, AbstractAtomicSet}(n, map, set)
    end
end


"""
    *(M, A::AbstractAtomicSet)

Apply the linear operator `M` on the atomic set `A`. Results in
another atomic set (MA) with the defining property

    MA = { Ma | a ∈ A }.
"""
function Base.:(*)(M::AbstractLinearOp, A::AbstractAtomicSet) 
    return MappedAtomicSet(M, A)
end

function Base.:(*)(λ::Float64, A::AbstractAtomicSet) 
    return MappedAtomicSet( LinearMaps.UniformScalingMap(λ, length(A)), A)
end


"""
    support(A::MappedAtomicSet, z::Vector)

Support value of a gauge with respect to a scaled atomic set.
"""
support(A::MappedAtomicSet, z::Vector{Float64}) = support(A.set, A.map'*z)


"""
    expose(A::MappedAtomicSet, z::Vector)

Obtain an atom in the face of the face of the scaled atomic set exposed by
the vector `z`.
"""
function expose!(A::MappedAtomicSet, z::Vector{Float64}, a::MappedAtom)
    expose!(A.set, A.map'*z, a.atom)
    return nothing
end

function expose(A::MappedAtomicSet, z::Vector{Float64})
    atom = expose(A.set, A.map'*z)
    return MappedAtom(A.map, atom)
end

Base.length(A::MappedAtomicSet) = A.n
atom_name(A::MappedAtomicSet) = "Map of $(atom_name(A.set))"
atom_description(A::MappedAtomicSet) = "M ⋅ $(atom_description(A.set))"
atom_parameters(A::MappedAtomicSet) = "n = $(length(A))"

########################################################################
# Face of the Mapped Atomic Set
########################################################################

struct MappedFace{T1<:Int64, T2<:AbstractLinearOp, T3<:AbstractFace} <: AbstractFace
    n::T1
    k::T1
    map::T2
    fa::T3
    function MappedFace(map::AbstractLinearOp, fa::AbstractFace)
        size(map, 2) == length(fa) || error("size(M) incompatible with face")
        n = size(map, 1)
        k = rank(fa)
        new{Int64, AbstractLinearOp, AbstractFace}(n, k, map, fa)
    end
end

"""
    face(A, z)

Return a face of the atomic set `A` exposed by the vector `z`. 
"""
function face!(A::MappedAtomicSet, z::Vector{Float64}, F::MappedFace)
    face!(A.set, A.map'*z, F.fa)
    return nothing
end

function face(A::MappedAtomicSet, z::Vector{Float64})
    fa = face(A.set, A.map'*z)
    return MappedFace(A.map, fa)
end

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::MappedFace, c::Vector) = (F.map*F.fa)*c
Base.:(*)(M::AbstractLinearOp, F::MappedFace) = M*F.map*F.fa
Base.:(*)(λ::Real, F::MappedFace) = λ*F.map*F.fa
Base.length(F::MappedFace) = F.n
rank(F::MappedFace) = F.k
vec(F::MappedFace) = F.map*F.fa
face_name(F::MappedFace) = "Mapped Atomic Set"
face_parameters(F::MappedFace) = "rank = $(rank(F)); n = $(length(F))"
