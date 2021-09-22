########################################################################
# Scaled atoms.
########################################################################

"Scaled atom."
struct ScaledAtom{T <: AbstractAtom} <: AbstractAtom
    child::T
    λ::Float64
    function ScaledAtom(child::T, λ::Real) where T
        λ < 0 && error("λ must be nonnegative")
        return new{T}(child, convert(Float64,λ))
    end
end

Base.length(a::ScaledAtom) = length(a.child)
Base.vec(a::ScaledAtom) = a.λ*vec(a.child)
Base.:(*)(M::AbstractLinearOp, a::ScaledAtom) = a.λ*(M*a.child)

"""
     (*)(λ::Real, a::AbstractAtom)
     (*)(a::AbstractAtom, λ::Real)

Return a scaled atom.
"""
Base.:(*)(λ::Real, a::AbstractAtom) = ScaledAtom(a, λ)
Base.:(*)(a::AbstractAtom, λ::Real) = ScaledAtom(a, λ)

########################################################################
# Scaled atomic set.
########################################################################

"""
    ScaledAtomicSet(A::AbstractAtomicSet, λ::Real)

Nonnegative scalar multiple of another atomic set.
"""
struct ScaledAtomicSet{T <: AbstractAtomicSet} <: AbstractAtomicSet
    child::T
    λ::Float64
    function ScaledAtomicSet(child::T, λ::Real) where T
        λ < 0 && error("λ must be nonnegative")
        return new{T}(child, convert(Float64,λ))
    end
end


"""
        *(λ::Real, A::AbstractAtomicSet)
        *(A::AbstractAtomicSet, λ::Real)

Scale an atomic set by a scalar.
"""
Base.:(*)(λ::Real, A::AbstractAtomicSet) = ScaledAtomicSet(A, λ)
Base.:(*)(A::AbstractAtomicSet, λ::Real) = ScaledAtomicSet(A, λ)


"""
    gauge(A::ScaledAtomicSet, x::Vector)

Gauge value of a vector with respect to a scaled atomic set.
"""
gauge(A::ScaledAtomicSet, x::Vector) = gauge(A.child, x)/A.λ


"""
    support(A::ScaledAtomicSet, x::Vector)

Support value of a gauge with respect to a scaled atomic set.
"""
support(A::ScaledAtomicSet, z::Vector) = support(A.child, z)*A.λ


"""
    expose!(A::ScaledAtomicSet, z::Vector)

Obtain an atom in the face of the face of the scaled atomic set exposed by
the vector `z`.
"""
function expose!(A::ScaledAtomicSet, z::Vector)
    a = expose(A.child, z)
    return ScaledAtom(a, A.λ)
end

expose(A::ScaledAtomicSet, z::Vector) = expose!(A, copy(z))

Base.length(A::ScaledAtomicSet) = length(A.child)
atom_name(A::ScaledAtomicSet) = "Scaling of $(atom_name(A.child))"
atom_description(A::ScaledAtomicSet) = "λ ⋅ $(atom_description(A.child))"
atom_parameters(A::ScaledAtomicSet) = "λ = $(A.λ); $(atom_parameters(A.child))"

########################################################################
# Face of the Scaled Atomic Set
########################################################################

struct ScaledFace <: AbstractFace
    n::Int64
    k::Int64
    S::LinearMap{Float64}
end

ScaledFace(S::LinearMap) = ScaledFace(size(S)...,S)

"""
    face(A, z)

Return a face of the atomic set `A` exposed by the vector `z`.
"""
function face(A::ScaledAtomicSet, z::Vector; rTol=1e-1)
    F = face(A.child, z, rTol = rTol)
    S = A.λ*F
    return ScaledFace(S)
end

"""
Given a face and a set of weights, reveal the corresponding
point on the face.
"""
Base.:(*)(F::ScaledFace, c::Vector) = F.S*c
Base.:(*)(M::AbstractLinearOp, F::ScaledFace) = M*F.S
Base.:(*)(λ::Real, F::ScaledFace) = λ*F.S
Base.length(F::ScaledFace) = F.n
rank(F::ScaledFace) = F.k
vec(F::ScaledFace) = F.S
face_name(F::ScaledFace) = "Scaled Atomic Set"
face_parameters(F::ScaledFace) = "rank = $(rank(F)); n = $(length(F))"
